"""Typed directive sub-models — the shapes referenced inside fixtures.

Typing strategy: model every key in every fixture, but use
``dict[str, Any]`` for genuinely polymorphic payloads (notably the inner
``state``/``nodes``/``edges`` of recursive subgraph definitions, and the
update payloads themselves which are arbitrary state-shaped dicts). The
load-bearing invariant is that every TOP-LEVEL and DIRECTIVE key is
known — a fixture introducing a new directive that we haven't modelled
fails parsing immediately, and that's exactly what we want.

The submodels here (``NodeSpec``, ``MiddlewareSpec``, etc.) are referenced
from :mod:`fixtures` and :mod:`expectations`. The split is for readability;
all of these could live in one file but the file would push 800 lines.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class _ForbidExtras(BaseModel):
    """Strict — used for the structural skeleton (state schema, node primary
    directive set, edges, observer registration, middleware config split).
    Catches new fixture directives at the load-bearing places."""

    model_config = ConfigDict(extra="forbid")


class _AllowExtras(BaseModel):
    """Permissive — used for payload-shape models (mock LLM responses,
    middleware-specific params, flaky/fan-out config). Validates KNOWN
    keys' types but doesn't reject unknown ones — these payloads evolve
    frequently and modelling every parameter exhaustively
    creates churn without proportional value. The strictness
    contract sits at the directive STRUCTURE level (above), not the
    parameter-bag level (here)."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# State schema (state.fields)
# ---------------------------------------------------------------------------


class StateFieldSpec(_ForbidExtras):
    """A single state field declaration.

    The ``alt_reducer`` knob exists only for ``graph-engine/007-compile-errors``'s
    ``conflicting_reducers`` case — fixtures intentionally declare two reducers
    on one field to verify the engine fails compile with the right category.

    The ``required`` knob (used by the state-migration deserialization-
    failure fixture 044) marks a field as having no default — Pydantic's
    natural "required" shape. The default-when-omitted falls through
    via the ``default`` field above.
    """

    type: str
    default: Any = None
    reducer: str | None = None
    alt_reducer: str | None = None
    required: bool = False


class StateSchema(_ForbidExtras):
    fields: dict[str, StateFieldSpec]
    # User-facing state-schema version per pipeline-utilities §10.2
    # (proposal 0014). The state-migration fixtures (039-046) declare
    # this on each case's ``state`` block; non-migration fixtures
    # omit it (defaults to empty-string sentinel).
    schema_version: str = ""


# ---------------------------------------------------------------------------
# Edge specs
# ---------------------------------------------------------------------------


class EdgeSpec(_AllowExtras):
    """One edge in a graph definition.

    Edges come in static (``from``/``to``) and conditional
    (``from``/``condition``) forms; observability/011 also uses a
    ``when``-shaped predicate. Schema is permissive here so all forms
    parse — the engine retrofit interprets each shape against
    the engine's edge model.
    """

    from_: str = Field(alias="from")
    to: str | None = None
    condition: dict[str, Any] | None = None
    when: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Node directive sub-models
# ---------------------------------------------------------------------------


class FailureSpec(_AllowExtras):
    """One entry in a flaky node's ``failure_sequence``. ``transient: true``
    + ``category`` triggers a transient retry-classifier-friendly raise;
    ``transient: false`` raises a non-transient instead."""

    transient: bool
    category: str | None = None
    message: str | None = None


class FlakySpec(_AllowExtras):
    """Base flaky directive shapes.

    Two known sub-shapes share the ``flaky:`` key:

    1. Sequence form (pipeline-utilities/007 etc.): ``failure_sequence`` of
       per-attempt failures + ``success_update`` for the success state.
    2. Compact form (pipeline-utilities/029): ``fail_first_invocation_only``
       boolean + ``on_success`` state update — used in checkpoint fixtures
       where the failure is keyed to first invocation.

    ``flaky_resume_aware`` (pipeline-utilities/027) is a *separate* node
    directive even though it lives under a node named ``flaky`` in that
    fixture — see :class:`FlakyResumeAwareSpec`.
    """

    failure_sequence: list[FailureSpec | None] | None = None
    success_update: dict[str, Any] | None = None
    fail_first_invocation_only: bool | None = None
    on_success: dict[str, Any] | None = None


class FlakyByIndexSpec(_AllowExtras):
    """Fan-out variant: failure depends on ``fan_out_index``.

    Two presence patterns:

    - ``fail_when_idx`` (int) — only that index fails.
    - ``fail_count_per_idx`` (int) — every index fails this many attempts
      before succeeding.

    Both come with ``category`` (transient category) and ``success_compute``
    (the success state shape).
    """

    fail_when_idx: int | None = None
    fail_count_per_idx: int | None = None
    category: str | None = None
    success_compute: dict[str, Any]


class FlakyPerIndexSpec(_AllowExtras):
    """Checkpoint-resume variant. Two failure-injection shapes:

    - ``fail_first_run_indices``: indices fail on the first invocation
      only; everyone succeeds on subsequent runs (used by 048, 049,
      050, 051, 053, 054 to simulate "abort, then resume succeeds").
    - ``always_fail_indices``: indices fail on EVERY invocation
      (deterministic failure). Used by 052 (collect mode): the failure
      gets recorded as an error contribution and the instance is
      ``completed`` on the saved record; on resume the recorded error
      rolls forward verbatim without re-running.

    Both forms share ``success_compute`` for the success-path state
    update.
    """

    fail_first_run_indices: list[int] | None = None
    always_fail_indices: list[int] | None = None
    success_compute: dict[str, Any]


class FlakyInstanceOnlySpec(_AllowExtras):
    """Instance-middleware variant: each fan-out instance fails its first
    ``fail_count_per_instance`` whole-instance invocations, then succeeds."""

    fail_count_per_instance: int
    category: str
    success_compute: dict[str, Any]


class FlakyResumeAwareSpec(_AllowExtras):
    """Checkpoint-resume + retry variant: fails N times on the first
    invocation, then on resume (any later invocation) fails M times before
    succeeding. Used to verify ``attempt_index`` resets on resume."""

    fail_first_invocation_count: int
    fail_resumed_invocation_count: int
    category: str
    on_success: dict[str, Any]


class UpdateFromFieldSpec(_ForbidExtras):
    """Mock computation: result_field = input_field × multiplier.

    Used by fan-out fixtures to give instances a deterministic,
    parameterizable computation without needing real LLM calls. The harness
    mock interprets this directive at runtime.
    """

    # Free-form: some fixtures use ``{result: x, multiplier: 2}``, others
    # ``{score: item}`` with no multiplier. The fan-out runtime reads
    # whichever keys are present.
    model_config = ConfigDict(extra="allow")


class FanOutSpec(_AllowExtras):
    """A fan-out node's configuration.

    Two mutually exclusive modes:

    - ``items_field`` mode — instance count = ``len(parent_state[items_field])``;
      each instance's input is ``items_field[i]`` projected into ``item_field``.
    - ``count`` mode — instance count = ``count`` (literal int OR callable);
      no per-item data.

    Cross-cutting: ``concurrency`` (default 10), ``error_policy`` (default
    ``fail_fast``; alternative ``collect``), ``on_empty`` (default ``raise``;
    alternative ``noop``), ``count_field`` (writes resolved count to this
    parent field), ``errors_field`` (for ``collect`` mode), and
    ``instance_middleware`` (whole-instance retry seam).
    """

    subgraph: str
    # Mode A — items.
    items_field: str | None = None
    item_field: str | None = None
    # Mode B — count. Permissive ``Any`` because fixtures express
    # callable counts as e.g. ``{callable: state_field, field: workers}``.
    count: Any = None
    count_field: str | None = None
    # Common. ``concurrency`` accepts the same shapes as ``count``.
    collect_field: str | None = None
    target_field: str | None = None
    concurrency: Any = None
    error_policy: Literal["fail_fast", "collect"] | None = None
    on_empty: Literal["raise", "noop"] | None = None
    errors_field: str | None = None
    instance_middleware: list[MiddlewareSpec] | None = None
    # proposal-0009 fixtures (048-054): ``concurrent_mode: serial``
    # forces deterministic completion ordering for resume-correctness
    # assertions. The adapter translates this into ``concurrency=1``;
    # other modes (the implicit "concurrent" default) fall through to
    # the configured ``concurrency`` value.
    concurrent_mode: Literal["serial", "concurrent"] | None = None
    # proposal-0009 fixture 052 (collect mode): ``abort_after_instance: N``
    # is a harness directive — after instance N's completion save fires,
    # the harness aborts to simulate a crash with that prefix flushed.
    # The engine never sees this directive; the conformance test driver
    # interprets it.
    abort_after_instance: int | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_instance_middleware(cls, data: Any) -> Any:
        """Normalize the two YAML shapes for ``instance_middleware`` entries.

        Existing fixtures (e.g., 021) use the explicit-tag form::

            instance_middleware:
              - type: retry
                max_attempts: 3

        proposal-0009 fixture 053 uses the key-as-tag form::

            instance_middleware:
              - retry:
                  max_attempts: 3

        Both are valid YAML descriptions of the same configuration.
        Rewrite the second shape into the first BEFORE the
        discriminated union validates so downstream parsing sees a
        uniform shape.
        """
        if not isinstance(data, dict):
            return data
        data_dict: dict[str, Any] = cast("dict[str, Any]", data)
        im: Any = data_dict.get("instance_middleware")
        if not isinstance(im, list):
            return data_dict
        normalized: list[Any] = []
        for entry in cast("list[Any]", im):
            if isinstance(entry, dict):
                entry_dict = cast("dict[str, Any]", entry)
                if "type" not in entry_dict and len(entry_dict) == 1:
                    only_key = next(iter(entry_dict))
                    inner = entry_dict[only_key]
                    if isinstance(inner, dict):
                        inner_dict = cast("dict[str, Any]", inner)
                        flat: dict[str, Any] = {"type": only_key}
                        flat.update(inner_dict)
                        normalized.append(flat)
                        continue
            normalized.append(entry)
        data_dict["instance_middleware"] = normalized
        return data_dict


class ParallelBranchSpec(_AllowExtras):
    """One entry inside a ``parallel_branches.branches`` mapping.

    A branch's work is given by exactly one of ``subgraph`` (a compiled
    subgraph referenced by name, with optional ``inputs`` / ``outputs``)
    or ``call`` (an inline node-behavior directive — ``update`` / ``flaky``
    / ``raises`` — run as a function over the parent state, proposal 0075).
    An optional ``when`` directive (``{field: <name>}``) skips the branch
    at dispatch. Permissive on extras because fixtures may carry extra
    knobs (e.g., per-branch annotations the harness ignores).
    """

    subgraph: str | None = None
    call: dict[str, Any] | None = None
    when: dict[str, Any] | None = None
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    middleware: list[MiddlewareSpec] | None = None


class ParallelBranchesSpec(_AllowExtras):
    """``parallel_branches:`` block on a NodeSpec.

    Mirrors :class:`FanOutSpec` but topology-driven: M heterogeneous
    branches, each referencing a different compiled subgraph by name
    against the case's top-level ``subgraphs:`` block. Branch insertion
    order is preserved.
    """

    branches: dict[str, ParallelBranchSpec]
    error_policy: Literal["fail_fast", "collect"] | None = None
    errors_field: str | None = None


class RuntimeConfigSpec(_AllowExtras):
    """``calls_llm.config`` block — mirrors ``RuntimeConfig``. Used by
    observability fixtures 016-018 (request-parameter and
    extras emission) and by the GenAI semconv set.

    Each field maps one-to-one to ``openarmature.llm.response.RuntimeConfig``
    on the source side. ``extras`` is the ``extra="allow"`` pass-through
    bag for provider-specific parameters (frequency_penalty, etc.).
    """

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    # Three declared fields promoted by proposal 0032 (spec v0.24.0).
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop_sequences: list[str] | None = None
    extras: dict[str, Any] | None = None


class CallsLlmSpec(_AllowExtras):
    """LLM-using node: sends ``messages`` to the harness's mock provider
    and stores the response (assistant content) in ``stores_response_in``.
    Used by observability fixtures to verify LLM-provider span emission.

    ``config`` (fixtures 016-018) carries the optional
    ``RuntimeConfig`` field set for the call — temperature, max_tokens,
    top_p, seed, and a provider-specific ``extras`` bag.
    """

    messages: list[dict[str, Any]]
    stores_response_in: str
    config: RuntimeConfigSpec | None = None


class EmitsLogSpec(_AllowExtras):
    """Additive companion: the node emits a log record alongside its
    state update. Verified by observability fixture 010 (Logs Bridge)."""

    message: str
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class GlobalTracerSpec(_AllowExtras):
    """Additive companion: the node ALSO emits a span via the OTel global
    tracer (in addition to whatever it does normally). Used by
    observability fixture 005 to verify private-tracer isolation."""

    span_name: str


class NodeSpec(_ForbidExtras):
    """A single node's directive.

    Exactly one *primary* directive must be set:

    - ``update`` / ``update_pure`` / ``update_pure_from_state`` /
      ``update_from_field`` — state-update flavours (the latter is a mock
      computation interpreted by the fan-out harness).
    - ``raises`` — node raises with the given message; pair with optional
      ``error_category``.
    - ``subgraph`` — references a top-level ``subgraph``/``subgraphs``
      definition by name. Companions: ``inputs``, ``outputs`` for explicit
      mapping.
    - ``fan_out`` — see :class:`FanOutSpec`.
    - ``flaky`` and the four ``flaky_*`` variants — harness mocks for
      retry/checkpoint behaviours.
    - ``calls_llm`` — see :class:`CallsLlmSpec`.

    Companion modifiers (additive, may combine with most primaries):

    - ``emits_log`` — fires a log record with the node's update.
    - ``also_emits_via_global_tracer`` — fires a span on the OTel global
      provider (used to verify isolation).
    - ``middleware`` — per-node middleware list.
    """

    # Primary directives — exactly one of these must be set.
    update: dict[str, Any] | None = None
    update_pure: dict[str, Any] | None = None
    update_pure_from_state: dict[str, Any] | None = None
    update_from_field: UpdateFromFieldSpec | None = None
    raises: str | dict[str, Any] | None = None
    subgraph: str | None = None
    fan_out: FanOutSpec | None = None
    parallel_branches: ParallelBranchesSpec | None = None
    flaky: FlakySpec | None = None
    flaky_by_index: FlakyByIndexSpec | None = None
    flaky_per_index: FlakyPerIndexSpec | None = None
    flaky_instance_only: FlakyInstanceOnlySpec | None = None
    flaky_resume_aware: FlakyResumeAwareSpec | None = None
    calls_llm: CallsLlmSpec | None = None

    # Companions — additive.
    inputs: dict[str, str] | None = None
    outputs: dict[str, str] | None = None
    middleware: list[MiddlewareSpec] | None = None
    emits_log: EmitsLogSpec | None = None
    also_emits_via_global_tracer: GlobalTracerSpec | None = None
    # Pair with ``raises`` to specify the error category (graph-engine §4).
    error_category: str | None = None
    # Parallel-branches fixtures (033, 037): the node sleeps this many
    # milliseconds before its update fires. Used to force deterministic
    # branch-completion ordering (037 — different branches finish at
    # different wall-clock times yet final state must be insertion-order
    # deterministic per §11.8) and to slow a third branch so fail-fast
    # cancellation has time to land before it finishes (033).
    sleep_ms: int | None = None

    _PRIMARY_FIELDS = (
        "update",
        "update_pure",
        "update_pure_from_state",
        "update_from_field",
        "raises",
        "subgraph",
        "fan_out",
        "parallel_branches",
        "flaky",
        "flaky_by_index",
        "flaky_per_index",
        "flaky_instance_only",
        "flaky_resume_aware",
        "calls_llm",
    )

    @model_validator(mode="after")
    def _exactly_one_primary(self) -> NodeSpec:
        set_primaries = [field for field in self._PRIMARY_FIELDS if getattr(self, field) is not None]
        if len(set_primaries) == 0:
            raise ValueError(f"node has no primary directive (one of: {list(self._PRIMARY_FIELDS)})")
        if len(set_primaries) > 1:
            raise ValueError(f"node has multiple primary directives: {set_primaries}; exactly one is allowed")
        return self


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class RetryMiddleware(_AllowExtras):
    type: Literal["retry"]
    max_attempts: int
    classifier: dict[str, Any] | None = None
    backoff: dict[str, Any] | None = None


class TimingMiddleware(_AllowExtras):
    type: Literal["timing"]
    # ``on_complete`` shape varies: 014 uses a string label
    # (``"capture"``); 012 uses a dict (``{capture_to: timing_records}``)
    # to point at a state field. Permissive Any covers both.
    on_complete: Any = None


class ErrorRecoveryMiddleware(_AllowExtras):
    """Test-seam middleware (006): catches exceptions, returns a synthetic
    state update instead of re-raising."""

    type: Literal["error_recovery"]
    catch_categories: list[str] | None = None
    on_error_update: dict[str, Any] | None = None


class ShortCircuitMiddleware(_AllowExtras):
    """Test-seam middleware (004): bypasses ``next()`` and returns a
    state update directly, never invoking the wrapped node."""

    type: Literal["short_circuit"]
    update: dict[str, Any] | None = None


class TraceRecorderMiddleware(_AllowExtras):
    """Test-seam middleware (002): records the order of pre/post-node
    callbacks for composition-ordering assertions. Carries free-form
    parameters for the harness mock — typical fixtures supply
    ``name``, ``pre_marker``, ``post_marker`` to label the recorded
    entries."""

    type: Literal["trace_recorder"]


class FailureIsolationMiddleware(_AllowExtras):
    """Canonical failure-isolation middleware (fixtures 058-063).
    Catches an exception escaping the inner chain and
    returns a configured degraded partial update, emitting a distinct
    ``FailureIsolatedEvent``."""

    type: Literal["failure_isolation"]
    # Static partial-update mapping, or the callable encoding
    # ``{callable: state_derived, template, target_field}`` (fixture 059).
    degraded_update: dict[str, Any]
    event_name: str
    # Optional ``{matches_category: <category>}`` predicate (fixture 060).
    predicate: dict[str, Any] | None = None
    # Optional on_caught hook
    # ``{kind, increment_field, capture_message_field}`` (fixture 062).
    on_caught: dict[str, Any] | None = None


MiddlewareSpec = Annotated[
    RetryMiddleware
    | TimingMiddleware
    | ErrorRecoveryMiddleware
    | ShortCircuitMiddleware
    | TraceRecorderMiddleware
    | FailureIsolationMiddleware,
    Field(discriminator="type"),
]


class MiddlewareConfig(_ForbidExtras):
    """Top-level ``middleware:`` block — registers middlewares per-graph
    and/or per-node. ``per_graph`` wraps every node; ``per_node`` is a map
    of node name to per-node list."""

    per_graph: list[MiddlewareSpec] | None = None
    per_node: dict[str, list[MiddlewareSpec]] | None = None


# Resolve the forward references on NodeSpec/FanOutSpec.
NodeSpec.model_rebuild()
FanOutSpec.model_rebuild()


# ---------------------------------------------------------------------------
# Mock provider / mock LLM responses (llm-provider + observability)
# ---------------------------------------------------------------------------


class MockResponse(_AllowExtras):
    """One canned response from the harness mock LLM provider.

    Common fields:

    - ``status`` (int) + ``body`` (dict) — successful HTTP response.
    - ``raises_category`` (str) + ``cause`` — error-categories fixtures.
    - ``connection_failure`` (bool) — network failure simulation
      (llm-provider/004 connection_failure case).

    Permissive shape because the body's content mirrors OpenAI's wire
    format which is wide and evolving; modelling every field would
    duplicate the OpenAI schema. The OpenAI wire format is the
    authoritative shape.
    """

    status: int | None = None
    body: dict[str, Any] | None = None
    raises_category: str | None = None
    cause: dict[str, Any] | None = None
    connection_failure: bool | None = None


class MockProviderConfig(_AllowExtras):
    """``mock_provider:`` block.

    - ``responses`` is consumed in order: each ``complete()`` call pops
      the next entry.
    - ``health_endpoint`` configures the mock for ``ready()`` checks
      (llm-provider/007); the harness exposes a separate health-probe
      response distinct from the ``responses`` queue.

    Permissive shape — fixture-specific config knobs (e.g. retry
    intervals, simulated latencies) may appear without breaking parse.
    """

    responses: list[MockResponse] | None = None
    health_endpoint: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Observers (graph-engine §6)
# ---------------------------------------------------------------------------


class ObserverSpec(_ForbidExtras):
    """One observer registration.

    - ``attach`` is ``graph`` (graph-attached, persists across invocations)
      or ``invocation`` (passed to one ``invoke`` call).
    - ``target`` is ``outer`` (outermost graph) or a subgraph name.
    - ``behavior`` is ``record`` (capture events for assertion) or
      ``raise`` (raise to verify error isolation).
    - ``phases`` (optional) — subset of ``{"started", "completed"}``
      for per-observer phase subscription.
    - ``sleep_ms_per_event`` — the slow-observer directive. An int
      means a constant sleep per
      event; a dict with ``first_invocation`` / ``subsequent_invocations``
      keys selects per invocation index (used by fixture 024 to slow
      only the first invocation).
    """

    name: str
    attach: Literal["graph", "invocation"]
    target: str
    behavior: Literal["record", "raise"]
    phases: list[Literal["started", "completed"]] | None = None
    sleep_ms_per_event: int | dict[str, int] | None = None


# ---------------------------------------------------------------------------
# LlmProviderFixture's `calls`
# ---------------------------------------------------------------------------


class LlmCallSpec(_AllowExtras):
    """One call against the mock provider.

    ``operation`` is ``complete`` (with ``messages`` + optional ``tools``)
    or ``ready`` (no inputs). Other call params (temperature, max_tokens,
    top_p, seed, etc.) may appear and are passed through to the
    underlying provider call. ``expected`` is checked against the result.
    """

    operation: Literal["complete", "ready"]
    messages: list[dict[str, Any]] | None = None
    tools: list[dict[str, Any]] | None = None
    # Optional — when missing, the case-level ``expected:`` carries
    # the assertion (the per-call vs per-case split).
    expected: dict[str, Any] | None = None


__all__ = [
    "CallsLlmSpec",
    "EdgeSpec",
    "EmitsLogSpec",
    "ErrorRecoveryMiddleware",
    "FailureIsolationMiddleware",
    "FailureSpec",
    "FanOutSpec",
    "FlakyByIndexSpec",
    "FlakyInstanceOnlySpec",
    "FlakyPerIndexSpec",
    "FlakyResumeAwareSpec",
    "FlakySpec",
    "GlobalTracerSpec",
    "LlmCallSpec",
    "MiddlewareConfig",
    "MiddlewareSpec",
    "MockProviderConfig",
    "MockResponse",
    "NodeSpec",
    "ObserverSpec",
    "RetryMiddleware",
    "RuntimeConfigSpec",
    "ShortCircuitMiddleware",
    "StateFieldSpec",
    "StateSchema",
    "TimingMiddleware",
    "TraceRecorderMiddleware",
    "UpdateFromFieldSpec",
]
