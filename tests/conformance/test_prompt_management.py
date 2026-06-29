"""Run every spec prompt-management conformance fixture against the real subpackage.

The fixtures (``spec/prompt-management/conformance/``) describe
backend + manager behavior in terms of in-process mock backends and
``PromptManager`` operations. Unlike the llm-provider fixtures
(which mock a remote wire), the prompt-management harness instantiates
real ``PromptManager``s and runs them against ``MockPromptBackend``s —
no I/O or network involved.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from openarmature.llm.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from openarmature.prompts import (
    ChatPrompt,
    ContentSegment,
    MappingLabelResolver,
    PlaceholderSegment,
    Prompt,
    PromptError,
    PromptGroup,
    PromptManager,
    PromptNotFound,
    PromptResult,
    PromptStoreUnavailable,
    SamplingConfig,
    TextPrompt,
)

from ._deferral import skip_if_deferred
from .harness.loader import CONFORMANCE_ROOT
from .harness.prompt_management import (
    FixtureBackendSpec,
    FixtureCall,
    FixtureExpectedResultEquivalence,
    FixtureManagerSpec,
    PromptManagementFixture,
)

_CAPABILITY_DIR = CONFORMANCE_ROOT / "prompt-management" / "conformance"


def _fixture_paths() -> list[Path]:
    return sorted(_CAPABILITY_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


# ---------------------------------------------------------------------------
# Fixture YAML mapping helpers (chat prompts — proposal 0046)
# ---------------------------------------------------------------------------


def _segment_from_fixture(entry: dict[str, Any]) -> Any:
    """Map one ``chat_template`` entry from a fixture YAML to an OA
    ChatSegment.  Uses ``model_construct`` to bypass construction-time
    validators — the harness exists to test render-time behavior,
    including fixtures that intentionally build prompts violating
    construction-time invariants (placeholder regex, role-block
    compat).  Render-time enforcement (the normative trigger)
    still runs; only the construction-time ergonomic-only check is
    bypassed.

    Supported shapes:

    - ``{role, content}`` where ``content`` is a string → ContentSegment
      with a text-template content.
    - ``{role, content}`` where ``content`` is a list of block dicts →
      ContentSegment with a content-blocks-template.  Block dicts use
      ``{type: text, text: ...}`` / ``{type: image, source: {...}}`` /
      ``{type: image_url, url: ...}`` / ``{type: image_inline, ...}``.
    - ``{placeholder: <name>}`` → PlaceholderSegment.
    """
    from openarmature.prompts import (
        ImageInlineBlockTemplate,
        ImageURLBlockTemplate,
        TextBlockTemplate,
    )

    if "placeholder" in entry:
        return PlaceholderSegment.model_construct(
            type="placeholder",
            placeholder=cast("str", entry["placeholder"]),
        )
    role = cast("str", entry["role"])
    content = entry["content"]
    if isinstance(content, str):
        return ContentSegment.model_construct(
            type="content",
            role=cast("Any", role),
            content=content,
        )
    # list of blocks
    blocks: list[Any] = []
    for block in cast("list[dict[str, Any]]", content):
        block_type = block.get("type", "text")
        if block_type == "text":
            blocks.append(TextBlockTemplate(text=cast("str", block["text"])))
        elif block_type == "image":
            # Spec §3.1 / llm-provider §3.1.2 shape: ``{type: image,
            # source: {type: url|inline, url?: ..., base64_data?: ...},
            # media_type?: ..., detail?: ...}`` — ``source`` carries
            # the discriminator + scheme-specific fields; ``media_type``
            # and ``detail`` live at the block level (``media_type`` is
            # required for inline sources, ignored for URL sources per
            # llm-provider §3.1.2).
            source = cast("dict[str, Any]", block["source"])
            source_type = source.get("type")
            if source_type == "url":
                blocks.append(
                    ImageURLBlockTemplate(
                        url=cast("str", source["url"]),
                        detail=block.get("detail"),
                    )
                )
            elif source_type == "inline":
                blocks.append(
                    ImageInlineBlockTemplate(
                        base64_data=cast("str", source["base64_data"]),
                        media_type=cast("str", block.get("media_type", "")),
                        detail=block.get("detail"),
                    )
                )
            else:
                raise AssertionError(f"unsupported image source type: {source_type!r}")
        elif block_type == "image_url":
            blocks.append(
                ImageURLBlockTemplate(
                    url=cast("str", block["url"]),
                    detail=block.get("detail"),
                )
            )
        elif block_type == "image_inline":
            blocks.append(
                ImageInlineBlockTemplate(
                    base64_data=cast("str", block["base64_data"]),
                    media_type=cast("str", block["media_type"]),
                    detail=block.get("detail"),
                )
            )
        else:
            raise AssertionError(f"unsupported content-block type: {block_type!r}")
    return ContentSegment.model_construct(
        type="content",
        role=cast("Any", role),
        content=blocks,
    )


def _message_from_fixture(entry: dict[str, Any]) -> Message:
    """Map one fixture placeholder-list entry to an OA ``Message``.

    Placeholder injection carries caller-supplied ``Message`` lists
    so all four llm-provider roles are valid here (``system`` /
    ``user`` / ``assistant`` / ``tool``).  Unknown or misspelled
    roles raise rather than silently coerce to user — fail-closed
    posture symmetric to the Langfuse backend's mapper.
    """
    role = cast("str", entry["role"])
    content = entry["content"]
    if role == "system":
        return SystemMessage(content=cast("str", content))
    if role == "assistant":
        return AssistantMessage(content=cast("str", content))
    if role == "user":
        return UserMessage(content=content)
    if role == "tool":
        return ToolMessage(
            content=cast("str", content),
            tool_call_id=cast("str", entry["tool_call_id"]),
        )
    raise AssertionError(f"unsupported placeholder message role: {role!r}")


# ---------------------------------------------------------------------------
# MockPromptBackend — backend stand-in for the fixtures
# ---------------------------------------------------------------------------


class MockPromptBackend:
    """In-process PromptBackend matching the fixture ``backends[i]`` shape.

    Each instance carries a ``name`` (used for fallback-order tracing
    and for ``backend_call_counts`` assertions), an optional
    ``simulate_unavailable`` flag that makes every fetch raise
    ``PromptStoreUnavailable``, and a list of canned prompts keyed
    by ``(name, label)``.

    ``call_count`` is incremented on every ``fetch`` entry so
    fixtures 008 and 009 can assert how many times each backend's
    fetch was actually invoked.
    """

    def __init__(self, spec: FixtureBackendSpec) -> None:
        self.name = spec.name
        self._simulate_unavailable = spec.simulate_unavailable
        self._prompts: dict[tuple[str, str], Prompt] = {}
        now = datetime.now(UTC)
        for ps in spec.prompts:
            # Sampling sub-record (fixture 013): flatten the fixture's
            # `extras:` sub-block into top-level kwargs so caller-
            # supplied vendor knobs land in SamplingConfig's extras-
            # allow bag rather than as a literal `extras` key.
            sampling: SamplingConfig | None = None
            if ps.sampling is not None:
                flat: dict[str, Any] = {k: v for k, v in ps.sampling.items() if k != "extras"}
                extras = ps.sampling.get("extras")
                if isinstance(extras, dict):
                    for k, v in cast(dict[str, Any], extras).items():
                        flat.setdefault(k, v)
                sampling = SamplingConfig(**flat)
            if ps.chat_template is not None:
                # Proposal 0046: chat-prompt variant.  Map fixture
                # YAML segment dicts to OA ChatSegment entries via
                # ``model_construct`` to bypass construction-time §11
                # validators — fixtures 028 / 030 intentionally build
                # prompts that violate construction-time invariants
                # (duplicate placeholder names, invalid placeholder
                # regex) to verify the render-time error path.  The
                # spec-normative render-time checks still run.
                chat_segments = [_segment_from_fixture(entry) for entry in ps.chat_template]
                self._prompts[(ps.name, ps.label)] = ChatPrompt.model_construct(
                    kind="chat",
                    name=ps.name,
                    version=ps.version,
                    label=ps.label,
                    chat_template=chat_segments,
                    template_hash=ps.template_hash,
                    fetched_at=now,
                    sampling=sampling,
                    observability_entities=(
                        dict(ps.observability_entities) if ps.observability_entities is not None else None
                    ),
                )
            else:
                assert ps.template is not None, (
                    f"prompt {ps.name!r}/{ps.label!r} must declare either ``template`` or ``chat_template``"
                )
                self._prompts[(ps.name, ps.label)] = TextPrompt(
                    name=ps.name,
                    version=ps.version,
                    label=ps.label,
                    template=ps.template,
                    template_hash=ps.template_hash,
                    fetched_at=now,
                    sampling=sampling,
                    observability_entities=(
                        dict(ps.observability_entities) if ps.observability_entities is not None else None
                    ),
                )
        self.call_count = 0
        # Proposal 0072 (conformance-adapter §6.8) caching-primitive
        # state. ``source_read_count`` counts only source reads (cache
        # miss / bypass / staleness), distinct from ``call_count`` (every
        # fetch). The clock is controllable via ``advance_clock``.
        self._caching = spec.caching
        self._clock_seconds = 0
        self._cache_entry_time: dict[tuple[str, str], int] = {}
        self.source_read_count = 0

    def advance_clock(self, seconds: int) -> None:
        """Advance the controllable clock."""
        self._clock_seconds += seconds

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        self.call_count += 1
        if self._simulate_unavailable:
            raise PromptStoreUnavailable(
                f"mock backend {self.name!r} simulating unavailable for ({name!r}, {label!r})"
            )
        key = (name, label)
        if key not in self._prompts:
            raise PromptNotFound(
                f"mock backend {self.name!r} has no prompt for ({name!r}, {label!r})",
                name=name,
                label=label,
                backend=self.name,
            )
        if self._caching and not self._serves_from_cache(key, cache_ttl_seconds):
            # Source read: count it and (re)stamp the cache entry's age.
            self.source_read_count += 1
            self._cache_entry_time[key] = self._clock_seconds
        return self._prompts[key]

    def _serves_from_cache(self, key: tuple[str, str], cache_ttl_seconds: int | None) -> bool:
        # Proposal 0072 read-side control: 0 bypasses the cache (always a
        # source read); a missing entry is a source read; None serves any
        # cached entry; N > 0 serves only while the entry is younger than
        # N seconds (else a fresh source read).
        if cache_ttl_seconds == 0 or key not in self._cache_entry_time:
            return False
        if cache_ttl_seconds is None:
            return True
        age = self._clock_seconds - self._cache_entry_time[key]
        return age < cache_ttl_seconds


# ---------------------------------------------------------------------------
# Fixture runner
# ---------------------------------------------------------------------------


async def _run_call(
    call: FixtureCall,
    backends: dict[str, MockPromptBackend],
    manager: PromptManager | None,
    captures: dict[str, Any],
) -> tuple[Any, BaseException | None]:
    """Execute one fixture call, returning ``(result, raised)``.

    Exactly one of ``result`` / ``raised`` is populated.
    """
    target = call.target
    operation = call.operation

    try:
        if target == "construct_prompt_group":
            # Synthetic op — assemble a PromptGroup from captured
            # PromptResults.
            assert call.group_name is not None
            assert call.members_refs is not None
            members = [captures[ref] for ref in call.members_refs]
            return PromptGroup(group_name=call.group_name, members=members), None

        if isinstance(target, str) and target in {"manager", "secondary_manager", "tertiary_manager"}:
            # All three manager targets dispatch to the currently-active
            # manager in the per-pair iteration loop. The naming exists
            # only to keep fixture YAML self-describing under a
            # multi-manager shape (e.g., fixture 015).
            assert manager is not None
            if operation == "fetch":
                assert call.name is not None
                return (
                    await manager.fetch(call.name, call.label, cache_ttl_seconds=call.cache_ttl_seconds),
                    None,
                )
            if operation == "render":
                # Either inline fetched_prompt or a ref to a capture.
                if call.fetched_prompt_ref is not None:
                    prompt = captures[call.fetched_prompt_ref]
                else:
                    assert call.fetched_prompt is not None
                    fetched = await manager.fetch(call.fetched_prompt["name"], call.fetched_prompt["label"])
                    prompt = fetched
                placeholders = (
                    {k: [_message_from_fixture(m) for m in v] for k, v in call.placeholders.items()}
                    if call.placeholders is not None
                    else None
                )
                return (
                    manager.render(
                        prompt,
                        call.variables or {},
                        placeholders=placeholders,
                    ),
                    None,
                )
            if operation == "get":
                assert call.name is not None
                placeholders = (
                    {k: [_message_from_fixture(m) for m in v] for k, v in call.placeholders.items()}
                    if call.placeholders is not None
                    else None
                )
                return (
                    await manager.get(
                        call.name,
                        call.label,
                        call.variables or {},
                        placeholders=placeholders,
                        cache_ttl_seconds=call.cache_ttl_seconds,
                    ),
                    None,
                )
            raise AssertionError(f"unsupported manager operation: {operation!r}")

        # ``target: {backend: <name>}`` — direct backend op.
        assert not isinstance(target, str)
        backend = backends[target.backend]
        if operation == "fetch":
            assert call.name is not None and call.label is not None
            return await backend.fetch(call.name, call.label, cache_ttl_seconds=call.cache_ttl_seconds), None
        if operation == "advance_clock":
            # Proposal 0072 §6.8: advance the caching backend's clock.
            assert call.seconds is not None
            backend.advance_clock(call.seconds)
            return None, None
        raise AssertionError(f"unsupported backend operation: {operation!r}")
    except PromptError as exc:
        return None, exc


# ---------------------------------------------------------------------------
# Expectation assertions
# ---------------------------------------------------------------------------


def _assert_per_call(
    call: FixtureCall,
    result: Any,
    raised: BaseException | None,
    backends: dict[str, MockPromptBackend],
) -> None:
    if call.expected is None:
        return

    # Call-count assertions apply regardless of raise / success path —
    # a future fixture asserting short-circuit on a non-raising call
    # (e.g., first-backend hit means second-backend isn't consulted)
    # would have its expectation silently ignored if these checks
    # only ran inside the ``raises`` branch.
    if call.expected.secondary_backend_call_count is not None:
        assert backends["secondary"].call_count == call.expected.secondary_backend_call_count, (
            f"expected secondary call_count={call.expected.secondary_backend_call_count}, "
            f"got {backends['secondary'].call_count}"
        )
    if call.expected.backend_call_counts is not None:
        for name, count in call.expected.backend_call_counts.items():
            actual_count = backends[name].call_count
            assert actual_count == count, f"expected {name} call_count={count}, got {actual_count}"

    if call.expected.raises is not None:
        assert raised is not None, (
            f"expected raise of category {call.expected.raises.category!r}, got result {result!r}"
        )
        actual = getattr(raised, "category", None)
        assert actual == call.expected.raises.category, (
            f"expected category {call.expected.raises.category!r}, got {actual!r} ({raised!r})"
        )
        carries = call.expected.raises.carries
        if carries is not None:
            for key, expected_value in carries.items():
                if key == "description_mentions":
                    description = getattr(raised, "description", "") or str(raised)
                    assert expected_value in description, (
                        f"expected description to mention {expected_value!r}, got {description!r}"
                    )
                    continue
                actual_attr = getattr(raised, key, None)
                assert actual_attr == expected_value, (
                    f"expected {key}={expected_value!r}, got {actual_attr!r}"
                )
        return

    assert raised is None, f"unexpected raise: {raised!r}"

    if call.expected.prompt is not None:
        assert isinstance(result, (TextPrompt, ChatPrompt)), f"expected Prompt, got {type(result).__name__}"
        expected = call.expected.prompt.model_dump(exclude_none=True)
        for key, value in expected.items():
            actual_attr = getattr(result, key)
            assert actual_attr == value, f"prompt.{key}: expected {value!r}, got {actual_attr!r}"

    if call.expected.prompt_result is not None:
        assert isinstance(result, PromptResult), f"expected PromptResult, got {type(result).__name__}"
        expected = call.expected.prompt_result.model_dump(exclude_none=True)
        for key, value in expected.items():
            if key == "rendered_hash_present":
                if value:
                    assert result.rendered_hash, "expected rendered_hash present"
                continue
            if key == "rendered_hash_non_empty_string":
                if value:
                    assert isinstance(result.rendered_hash, str)
                    assert len(result.rendered_hash) > 0
                continue
            if key == "messages":
                expected_messages: list[dict[str, Any]] = value
                actual_messages = [m.model_dump(exclude_none=True) for m in result.messages]
                # Drop fields the fixture doesn't constrain.
                normalized: list[dict[str, Any]] = []
                for m in actual_messages:
                    normalized.append({k: m[k] for k in m if k in {"role", "content"}})
                assert normalized == expected_messages, (
                    f"messages: expected {expected_messages!r}, got {normalized!r}"
                )
                continue
            actual_attr = getattr(result, key)
            assert actual_attr == value, f"prompt_result.{key}: expected {value!r}, got {actual_attr!r}"


def _assert_result_equivalence(
    eq: FixtureExpectedResultEquivalence,
    captures: dict[str, Any],
) -> None:
    overlap = set(eq.fields_must_match) & set(eq.fields_may_differ)
    assert not overlap, (
        f"fixture inconsistency: fields {sorted(overlap)} appear in both "
        f"fields_must_match and fields_may_differ"
    )
    members = [captures[ref] for ref in eq.of]
    first = members[0]
    for other in members[1:]:
        for field in eq.fields_must_match:
            assert getattr(first, field) == getattr(other, field), (
                f"result_equivalence: field {field!r} differs across {eq.of!r}"
            )
        for field in eq.fields_must_differ:
            assert getattr(first, field) != getattr(other, field), (
                f"result_equivalence: field {field!r} matched across {eq.of!r} but MUST differ"
            )


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------


def _build_manager(
    spec: FixtureManagerSpec,
    backends_map: dict[str, MockPromptBackend],
    resolvers_map: dict[str, MappingLabelResolver],
) -> PromptManager:
    ordered = [backends_map[name] for name in spec.backends]
    resolver: MappingLabelResolver | None = None
    if spec.label_resolver_ref is not None:
        if spec.label_resolver_ref not in resolvers_map:
            raise AssertionError(f"unknown label_resolver_ref: {spec.label_resolver_ref!r}")
        resolver = resolvers_map[spec.label_resolver_ref]
    return PromptManager(*ordered, label_resolver=resolver)


def _assert_capture_attrs(capture_name: str, actual: Any, expected: dict[str, Any]) -> None:
    # Walk fixture-supplied expected attributes against a captured
    # Prompt / PromptResult. Handles sampling (flatten extras + dump),
    # *_absent flags, and dict-typed observability_entities.
    for key, expected_value in expected.items():
        if key == "sampling_absent":
            if expected_value:
                actual_sampling = getattr(actual, "sampling", None)
                assert actual_sampling is None, (
                    f"{capture_name}.sampling: expected absent, got {actual_sampling!r}"
                )
            continue
        if key == "observability_entities_absent":
            if expected_value:
                actual_oe = getattr(actual, "observability_entities", None)
                assert actual_oe is None, (
                    f"{capture_name}.observability_entities: expected absent, got {actual_oe!r}"
                )
            continue
        if key == "sampling":
            actual_sampling = getattr(actual, "sampling", None)
            assert actual_sampling is not None, f"{capture_name}.sampling: expected present, got None"
            # Spec sidecar convention nests vendor extras under
            # `extras:`; SamplingConfig.model_dump() flattens them to
            # the top level (extra="allow"). Normalize the expected
            # shape before equality compare.
            expected_flat = {k: v for k, v in expected_value.items() if k != "extras"}
            if isinstance(expected_value.get("extras"), dict):
                expected_flat.update(expected_value["extras"])
            actual_flat = actual_sampling.model_dump(exclude_none=True)
            assert actual_flat == expected_flat, (
                f"{capture_name}.sampling: expected {expected_flat!r}, got {actual_flat!r}"
            )
            continue
        actual_value = getattr(actual, key)
        # Proposal 0046: messages may be Message instances when the
        # fixture expects dict-shapes.  Dump for structural compare.
        if key == "messages" and isinstance(actual_value, list):
            actual_value = [
                _message_to_dict_for_compare(cast("Message", m)) for m in cast("list[Any]", actual_value)
            ]
        assert actual_value == expected_value, (
            f"{capture_name}.{key}: expected {expected_value!r}, got {actual_value!r}"
        )


def _message_to_dict_for_compare(message: Message) -> dict[str, Any]:
    """Dump a Message to a plain dict for structural equality against
    a fixture YAML expected value.  Mirrors the documented
    Message shape: ``{role, content}`` with optional extras."""
    dumped = message.model_dump(exclude_none=True)
    # Normalize content-blocks shape: drop pydantic internal
    # discriminator literal where fixtures don't carry it.
    return dumped


# Fixtures whose implementation lands in a later PR. Skip-marked so a
# green test run at this commit means "everything we claim to implement
# passes." Each subsequent PR drops its own rows as it lands the
# underlying support.
_DEFERRED_FIXTURES: dict[str, str] = {
    # Proposal 0047 (implicit prefix-cache wire-byte stability, spec
    # v0.39.0) adds an ``expected_shared_prefix`` directive — multi-
    # render byte-equality check on the shared template prefix.
    # Queued for v0.13.0 LLM provider hardening batch.
    "032-cross-variable-substring-stability": (
        "Proposal 0047 wire-byte stability (expected_shared_prefix directive); queued for v0.13.0"
    ),
    # ----- v0.16.0 spec-pin bump (v0.70.1 -> v0.84.0) -------------------
    # Proposal 0080 (PromptGroup arity enforcement, spec v0.75.0) -- fixture
    # 035 uses a cases-only shape (no backends) the PM fixture model doesn't
    # accept, and asserts the construct-time prompt_group_invalid raise that
    # python does not yet implement. Defers until a later v0.16.0 PR.
    "035-prompt-group-arity-rejection": ("Proposal 0080 PromptGroup arity enforcement; not implemented"),
    # Proposal 0086 (PromptManager default cache_ttl_seconds, spec v0.79.0)
    # -- fixture 036 uses the manager default-cache-ttl construction slot
    # python does not yet implement. Defers until a later v0.16.0 PR.
    "036-prompt-manager-default-cache-ttl": (
        "Proposal 0086 PromptManager default cache_ttl_seconds; not implemented"
    ),
}


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_prompt_management_fixture(fixture_path: Path) -> None:
    fixture_id = _fixture_id(fixture_path)
    skip_if_deferred(fixture_id, _DEFERRED_FIXTURES)
    raw: Any = yaml.safe_load(fixture_path.read_text())
    fixture = PromptManagementFixture.model_validate(raw)

    backends: dict[str, MockPromptBackend] = {spec.name: MockPromptBackend(spec) for spec in fixture.backends}

    # Named LabelResolvers; managers reference them by their fixture-
    # top-level key name via ``label_resolver_ref``.
    resolvers_map: dict[str, MappingLabelResolver] = {}
    if fixture.label_resolver is not None:
        resolvers_map["label_resolver"] = MappingLabelResolver(fixture.label_resolver.mapping)
    if fixture.tertiary_label_resolver is not None:
        resolvers_map["tertiary_label_resolver"] = MappingLabelResolver(
            fixture.tertiary_label_resolver.mapping
        )

    captures: dict[str, Any] = {}

    # Fixture 015 introduces secondary/tertiary manager+calls slots
    # that run independently with shared backends. Run each (manager,
    # calls) pair in declaration order; the captures dict is shared
    # so cross-manager assertions on capture names still work.
    manager_pairs = [
        (fixture.manager, fixture.calls),
        (fixture.secondary_manager, fixture.secondary_calls),
        (fixture.tertiary_manager, fixture.tertiary_calls),
    ]
    for manager_spec, manager_calls in manager_pairs:
        # ``manager_spec is None`` with direct-backend calls is the
        # proposal 0072 fixtures 033/034 shape (no manager; calls target
        # backends directly). Build a manager only when one is declared;
        # direct-backend calls don't need it.
        manager = _build_manager(manager_spec, backends, resolvers_map) if manager_spec is not None else None
        for call in manager_calls:
            result, raised = await _run_call(call, backends, manager, captures)
            _assert_per_call(call, result, raised, backends)
            if call.capture_as is not None and raised is None:
                captures[call.capture_as] = result

    # Cases-form fixtures (016) split into independent sub-cases that
    # share the backends but use their own per-case manager + calls.
    cases = raw.get("cases")
    if cases:
        for case in cases:
            # Strip case-level metadata (``name``, ``description``)
            # that PromptManagementFixture doesn't model; the runner
            # doesn't need them.
            case_payload = {
                **{k: v for k, v in raw.items() if k not in {"cases", "expected"}},
                **{k: v for k, v in case.items() if k not in {"name", "description"}},
            }
            case_fixture = PromptManagementFixture.model_validate(case_payload)
            case_manager_pairs = [
                (case_fixture.manager, case_fixture.calls),
                (case_fixture.secondary_manager, case_fixture.secondary_calls),
                (case_fixture.tertiary_manager, case_fixture.tertiary_calls),
            ]
            for manager_spec, manager_calls in case_manager_pairs:
                if manager_spec is None:
                    continue
                manager = _build_manager(manager_spec, backends, resolvers_map)
                for call in manager_calls:
                    result, raised = await _run_call(call, backends, manager, captures)
                    _assert_per_call(call, result, raised, backends)
                    if call.capture_as is not None and raised is None:
                        captures[call.capture_as] = result
            if case_fixture.expected is not None:
                _apply_top_level_expected(case_fixture.expected, captures)

    # Proposal 0072: per-backend end-state (e.g. source_read_count from
    # the caching primitive).
    if fixture.expected_backend_state is not None:
        for backend_name, state in fixture.expected_backend_state.items():
            backend = backends[backend_name]
            for attr, want in state.items():
                got = getattr(backend, attr)
                assert got == want, f"backend {backend_name!r} {attr}: got {got!r}, expected {want!r}"

    if fixture.expected is None:
        return

    _apply_top_level_expected(fixture.expected, captures)


def _apply_top_level_expected(expected: Any, captures: dict[str, Any]) -> None:
    if expected.prompt_group is not None:
        pg_expected = expected.prompt_group
        group = captures[pg_expected.of]
        assert isinstance(group, PromptGroup)
        assert group.group_name == pg_expected.group_name
        assert len(group.members) == pg_expected.member_count
        if pg_expected.member_names is not None:
            assert [m.name for m in group.members] == pg_expected.member_names

    if expected.result_equivalence is not None:
        _assert_result_equivalence(expected.result_equivalence, captures)
    for eq in expected.result_equivalences:
        _assert_result_equivalence(eq, captures)

    for pair in expected.rendered_hash_equal:
        a, b = pair
        assert captures[a].rendered_hash == captures[b].rendered_hash, (
            f"rendered_hash differs between {a!r} and {b!r} but fixture expects equal"
        )
    for pair in expected.rendered_hash_different:
        a, b = pair
        assert captures[a].rendered_hash != captures[b].rendered_hash, (
            f"rendered_hash matches between {a!r} and {b!r} but fixture expects different"
        )

    # Fixtures 013-016 use capture-name-keyed top-level expected
    # entries instead of the per-call expected:{prompt|prompt_result}
    # shape. Walk those via pydantic's model_extra (FixtureExpectedTopLevel
    # is permissive) and assert each capture matches the supplied
    # attribute dict.
    model_extra: dict[str, Any] = expected.model_extra or {}
    for capture_name, expected_attrs in model_extra.items():
        if not isinstance(expected_attrs, dict):
            continue
        if capture_name not in captures:
            raise AssertionError(f"expected capture {capture_name!r} not found in captures")
        _assert_capture_attrs(capture_name, captures[capture_name], cast(dict[str, Any], expected_attrs))
