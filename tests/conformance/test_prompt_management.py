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
from typing import Any

import pytest
import yaml

from openarmature.prompts import (
    Prompt,
    PromptError,
    PromptGroup,
    PromptManager,
    PromptNotFound,
    PromptResult,
    PromptStoreUnavailable,
)

from .harness.loader import CONFORMANCE_ROOT
from .harness.prompt_management import (
    FixtureBackendSpec,
    FixtureCall,
    FixtureExpectedResultEquivalence,
    PromptManagementFixture,
)

_CAPABILITY_DIR = CONFORMANCE_ROOT / "prompt-management" / "conformance"


def _fixture_paths() -> list[Path]:
    return sorted(_CAPABILITY_DIR.glob("[0-9][0-9][0-9]-*.yaml"))


def _fixture_id(path: Path) -> str:
    return path.stem


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
            self._prompts[(ps.name, ps.label)] = Prompt(
                name=ps.name,
                version=ps.version,
                label=ps.label,
                template=ps.template,
                template_hash=ps.template_hash,
                fetched_at=now,
            )
        self.call_count = 0

    async def fetch(self, name: str, label: str = "production") -> Prompt:
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
        return self._prompts[key]


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

        if isinstance(target, str) and target == "manager":
            assert manager is not None
            if operation == "fetch":
                assert call.name is not None and call.label is not None
                return await manager.fetch(call.name, call.label), None
            if operation == "render":
                # Either inline fetched_prompt or a ref to a capture.
                if call.fetched_prompt_ref is not None:
                    prompt = captures[call.fetched_prompt_ref]
                else:
                    assert call.fetched_prompt is not None
                    fetched = await manager.fetch(call.fetched_prompt["name"], call.fetched_prompt["label"])
                    prompt = fetched
                return manager.render(prompt, call.variables or {}), None
            if operation == "get":
                assert call.name is not None and call.label is not None
                return await manager.get(call.name, call.label, call.variables or {}), None
            raise AssertionError(f"unsupported manager operation: {operation!r}")

        # ``target: {backend: <name>}`` — direct backend op.
        assert not isinstance(target, str)
        backend = backends[target.backend]
        if operation == "fetch":
            assert call.name is not None and call.label is not None
            return await backend.fetch(call.name, call.label), None
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
        assert isinstance(result, Prompt), f"expected Prompt, got {type(result).__name__}"
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


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_prompt_management_fixture(fixture_path: Path) -> None:
    raw: Any = yaml.safe_load(fixture_path.read_text())
    fixture = PromptManagementFixture.model_validate(raw)

    backends: dict[str, MockPromptBackend] = {spec.name: MockPromptBackend(spec) for spec in fixture.backends}
    manager: PromptManager | None = None
    if fixture.manager is not None:
        ordered = [backends[name] for name in fixture.manager.backends]
        manager = PromptManager(*ordered)

    captures: dict[str, Any] = {}
    for call in fixture.calls:
        result, raised = await _run_call(call, backends, manager, captures)
        _assert_per_call(call, result, raised, backends)
        if call.capture_as is not None and raised is None:
            captures[call.capture_as] = result

    if fixture.expected is None:
        return

    if fixture.expected.prompt_group is not None:
        pg_expected = fixture.expected.prompt_group
        group = captures[pg_expected.of]
        assert isinstance(group, PromptGroup)
        assert group.group_name == pg_expected.group_name
        assert len(group.members) == pg_expected.member_count
        if pg_expected.member_names is not None:
            assert [m.name for m in group.members] == pg_expected.member_names

    if fixture.expected.result_equivalence is not None:
        _assert_result_equivalence(fixture.expected.result_equivalence, captures)
    for eq in fixture.expected.result_equivalences:
        _assert_result_equivalence(eq, captures)

    for pair in fixture.expected.rendered_hash_equal:
        a, b = pair
        assert captures[a].rendered_hash == captures[b].rendered_hash, (
            f"rendered_hash differs between {a!r} and {b!r} but fixture expects equal"
        )
    for pair in fixture.expected.rendered_hash_different:
        a, b = pair
        assert captures[a].rendered_hash != captures[b].rendered_hash, (
            f"rendered_hash matches between {a!r} and {b!r} but fixture expects different"
        )
