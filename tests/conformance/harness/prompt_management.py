"""Typed YAML models for prompt-management conformance fixtures.

Fixture shape (different from the llm-provider / graph shapes):

- ``backends:`` — list of mock backend specs (each with ``name``,
  optional ``simulate_unavailable``, and a list of ``prompts``).
- ``manager:`` — optional manager composition (a list of backend
  names, in fallback order).
- ``calls:`` — list of operations to drive. Each call has a
  ``target`` (``{backend: <name>}`` for direct backend operations,
  or ``manager`` for manager operations), an ``operation``, inputs,
  optional ``capture_as`` (binds the operation's result to a name
  usable by later calls / final expectations), and optional
  per-call ``expected``.
- ``expected:`` — optional top-level expectation block for
  PromptGroup shape or cross-call result-equivalence assertions
  that need access to ``capture_as`` bindings.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PermissiveModel(BaseModel):
    """For fixture sub-shapes that vary across fixtures and don't
    warrant a per-shape enumeration."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Backend / manager configuration
# ---------------------------------------------------------------------------


class FixturePromptSpec(_StrictModel):
    name: str
    label: str
    version: str
    # Text-prompt path uses ``template``; chat-prompt path uses
    # ``chat_template`` (proposal 0046, v0.38.0).  Exactly one is
    # set per fixture prompt.
    template: str | None = None
    chat_template: list[dict[str, Any]] | None = None
    template_hash: str
    # Proposal 0033: optional typed sub-record + observability-entities
    # mapping the mock backend attaches to the returned Prompt.
    sampling: dict[str, Any] | None = None
    observability_entities: dict[str, Any] | None = None


class FixtureBackendSpec(_StrictModel):
    name: str
    prompts: list[FixturePromptSpec] = []
    simulate_unavailable: bool = False
    # Proposal 0072 (conformance-adapter §6.8): when true, the backend is
    # a caching primitive — it caches by (name, label), counts source
    # reads, and honors cache_ttl_seconds via a controllable clock.
    caching: bool = False


class FixtureLabelResolverSpec(_StrictModel):
    # Mapping shape per spec §7 informative example: `"default"` is
    # the resolver's default-override (step 2 of the fallback chain);
    # any other key is a per-name override (step 1).
    mapping: dict[str, str]


class FixtureManagerSpec(_StrictModel):
    # backends is optional: fixture 036's manager block omits it and
    # defaults to all declared backends in order (proposal 0086).
    backends: list[str] | None = None
    label_resolver_ref: str | None = None
    # Proposal 0086: the manager's service-wide default cache_ttl_seconds.
    default_cache_ttl_seconds: int | None = None


# ---------------------------------------------------------------------------
# Call targets, operations, and expectations
# ---------------------------------------------------------------------------


class BackendTarget(_StrictModel):
    backend: str


class ManagerTarget(_StrictModel):
    # Proposal 0086: fixture 036 routes fetches through the manager via the
    # dict form ``target: {manager: true}`` (mirroring ``{backend: <name>}``),
    # alongside the bare-string ``manager`` the other fixtures use. Fixed to
    # ``true`` -- ``{manager: false}`` is nonsensical and rejected at parse.
    manager: Literal[True]


CallTarget = (
    BackendTarget
    | ManagerTarget
    | Literal[
        "manager",
        "secondary_manager",
        "tertiary_manager",
        "construct_prompt_group",
    ]
)


class FixtureExpectedRaises(_PermissiveModel):
    category: str
    # Optional extra carries. Fixture 005 surfaces
    # ``description_mentions`` / ``name`` / ``version`` / ``label``
    # here. Permissive on this shape so fixtures evolve;
    # per-backend call-count assertions live on the parent
    # ``FixtureExpectedPerCall`` (see ``secondary_backend_call_count``
    # and ``backend_call_counts`` below), not in ``carries``.
    carries: dict[str, Any] | None = None


class FixtureExpectedPrompt(_PermissiveModel):
    """Per-call ``expected.prompt`` shape (fetch ops)."""


class FixtureExpectedPromptResult(_PermissiveModel):
    """Per-call ``expected.prompt_result`` shape (render / get ops)."""


class FixtureExpectedPerCall(_StrictModel):
    prompt: FixtureExpectedPrompt | None = None
    prompt_result: FixtureExpectedPromptResult | None = None
    raises: FixtureExpectedRaises | None = None
    # Fixture 008's extra: assert how many times the secondary
    # backend's fetch was called. Lives alongside ``raises``.
    secondary_backend_call_count: int | None = None
    # Fixture 009's extra: assert per-backend call counts (named
    # backends → expected call count) after a fetch that exhausts
    # all of them.
    backend_call_counts: dict[str, int] | None = None


class FixtureCall(_StrictModel):
    target: CallTarget
    # ``operation`` is required for fetch / render / get calls. The
    # ``construct_prompt_group`` shape uses the target as the operation
    # indicator (no separate operation field on the call).
    operation: Literal["fetch", "render", "get", "advance_clock"] | None = None
    name: str | None = None
    # Proposal 0072: per-fetch read-side cache control, and the
    # advance_clock control op's step (in seconds).
    cache_ttl_seconds: int | None = None
    seconds: int | None = None
    # `label` is optional per spec §6 v0.26.0: omitting it triggers
    # the configured LabelResolver (step 2) or the spec fallback
    # `"production"` (step 3). Distinct from ``label: null`` which
    # YAML elides; pydantic still maps both to ``None``.
    label: str | None = None
    variables: dict[str, Any] | None = None
    # Proposal 0046: render-only ``placeholders`` kwarg — caller-
    # supplied message-list injections keyed by placeholder name.
    placeholders: dict[str, list[dict[str, Any]]] | None = None
    # Render-only inputs — either an inline ``fetched_prompt`` (which
    # the harness fetches first, then renders) or a ``fetched_prompt_ref``
    # pointing at an earlier ``capture_as``.
    fetched_prompt: dict[str, str] | None = None
    fetched_prompt_ref: str | None = None
    # construct_prompt_group-only inputs.
    group_name: str | None = None
    members_refs: list[str] | None = None
    capture_as: str | None = None
    expected: FixtureExpectedPerCall | None = None


# ---------------------------------------------------------------------------
# Top-level expected
# ---------------------------------------------------------------------------


class FixtureExpectedPromptGroup(_PermissiveModel):
    """Top-level ``expected.prompt_group`` shape (fixture 011)."""

    of: str
    group_name: str
    member_count: int
    member_order_preserved: bool | None = None
    member_names: list[str] | None = None


class FixtureExpectedResultEquivalence(_PermissiveModel):
    """Top-level ``expected.result_equivalence`` shape (fixtures 006,
    010, 012). Asserts equality across two or more captured results on
    a configurable set of fields."""

    of: list[str]
    fields_must_match: list[str]
    fields_may_differ: list[str] = []
    # fixture 012 — assert two different captures have a DIFFERENT
    # value on a given field.
    fields_must_differ: list[str] = []


class FixtureExpectedTopLevel(_PermissiveModel):
    """Top-level expected block.

    Most fixtures set one or more of the typed sub-blocks below
    (``prompt_group``, ``result_equivalence``, ``rendered_hash_*``).
    Fixture 015 (label-resolver) introduces a capture-name-keyed
    shape where each top-level key under ``expected:`` is a capture
    name and the value is a dict of Prompt/PromptResult attributes
    the harness MUST verify against the corresponding capture. Those
    keys arrive on ``model_extra`` since the typed fields below don't
    cover them; the runner walks ``model_extra`` to apply per-capture
    assertions.
    """

    prompt_group: FixtureExpectedPromptGroup | None = None
    result_equivalence: FixtureExpectedResultEquivalence | None = None
    # Some fixtures (012) have multiple result-equivalence blocks; keep
    # a plural list-form too. Empty by default.
    result_equivalences: list[FixtureExpectedResultEquivalence] = []
    # Fixture 012's per-pair rendered_hash equality / inequality
    # assertions. Each entry is a 2-element list of capture names; the
    # pair MUST share (resp. differ on) ``rendered_hash``.
    rendered_hash_equal: list[list[str]] = []
    rendered_hash_different: list[list[str]] = []


# ---------------------------------------------------------------------------
# Fixture root
# ---------------------------------------------------------------------------


class PromptManagementFixture(_StrictModel):
    # Optional: fixture 035 is cases-only with per-case backends (and an
    # empty-group case that needs no backend at all), so the top level
    # declares none (proposal 0080).
    backends: list[FixtureBackendSpec] = []
    # Fixture 016 uses a top-level ``cases:`` list to split into
    # independent sub-cases that share the backends declaration but
    # each have their own manager + calls. The runner walks the list
    # and runs each case in declaration order; the per-case shape is
    # the same as the top-level fixture (manager, calls, expected).
    cases: list[dict[str, Any]] | None = None
    manager: FixtureManagerSpec | None = None
    calls: list[FixtureCall] = []
    # Named LabelResolver specs; managers reference them by key name
    # via ``label_resolver_ref``. Fixture 015 introduces three named
    # slots — the primary `label_resolver` plus a `tertiary_label_resolver`
    # for the no-default branch. Future fixtures MAY add more slots
    # here; the harness resolves refs by attribute lookup.
    label_resolver: FixtureLabelResolverSpec | None = None
    tertiary_label_resolver: FixtureLabelResolverSpec | None = None
    # Fixture 015's multi-manager shape. Each `<prefix>_manager` /
    # `<prefix>_calls` pair runs independently with shared backends.
    secondary_manager: FixtureManagerSpec | None = None
    secondary_calls: list[FixtureCall] = []
    tertiary_manager: FixtureManagerSpec | None = None
    tertiary_calls: list[FixtureCall] = []
    expected: FixtureExpectedTopLevel | None = None
    # Proposal 0072: per-backend end-state assertions (e.g.
    # source_read_count) for the caching primitive.
    expected_backend_state: dict[str, dict[str, Any]] | None = None
