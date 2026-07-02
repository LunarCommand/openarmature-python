"""Adapter: spec conformance YAML fixtures → openarmature.graph constructs.

The fixture format is documented in
`openarmature-spec/spec/graph-engine/conformance/README.md`. This module
parses one fixture (or one sub-case from the table-style 007 fixture) into a
state class, a compiled graph, and an execution-order trace, so the
parametrized tests in `test_conformance.py` can drive the engine and assert
against the fixture's `expected` block.
"""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, cast

from pydantic import Field, create_model

from openarmature.graph import (
    END,
    BranchSpec,
    CompiledGraph,
    EndSentinel,
    ExplicitMapping,
    FanOutNode,
    FieldNameMatching,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    ParallelBranchesNode,
    ProjectionStrategy,
    Reducer,
    State,
    SubgraphNode,
    append,
    concat_flatten,
    last_write_wins,
    merge,
    merge_all,
)
from openarmature.graph.observer import Observer

if TYPE_CHECKING:
    from openarmature.graph.observer import _InvocationContext

REDUCERS: dict[str, Reducer] = {
    "last_write_wins": last_write_wins,
    "append": append,
    "merge": merge,
    "concat_flatten": concat_flatten,
    "merge_all": merge_all,
}


def _parse_type(s: str) -> Any:
    s = s.strip()
    if s == "string":
        return str
    if s == "int":
        return int
    if s == "float":
        return float
    if s == "bool":
        return bool
    # ``any`` admits the callable-degrade null slot (proposal 0066 fixture
    # 065 Case 3): a fan-out collection whose degraded instance omits
    # collect_field gets a null entry, so the element type must permit None.
    if s == "any":
        return Any
    # proposal-0063 tool fixtures (092-098): ``record`` is the state slot
    # a tool result is stored in — an opaque, language-idiomatic value
    # (often a mapping) with a ``null`` default. ``Any`` admits both the
    # null default and whatever shape the tool produced.
    if s == "record":
        return Any
    # Unparameterized container types — parallel-branches fixtures
    # 034/035/037 use ``dict`` and ``list<dict>`` as state-field types
    # for accumulator slots (branch_errors, merged_dict, collected_labels)
    # where the element shape is heterogeneous across branches. The
    # proposal-0036 reducer fixtures (026/027) use bare ``list`` /
    # ``dict`` deliberately so the reducer (not the typed-state layer)
    # is the gatekeeper for the list-of-lists / list-of-mappings shape.
    if s == "dict":
        return dict[str, Any]
    if s == "list":
        return list[Any]
    if s == "list<dict>":
        return list[dict[str, Any]]
    # proposal-0009 fixture 052: ``error_entry`` is the spec's shorthand
    # for the per-instance error record contributed to ``errors_field``
    # under collect mode. The exact shape is implementation-defined per
    # §9.5; the engine ships dict[str, str] with at least
    # ``fan_out_index`` and ``category`` keys.
    if s == "list<error_entry>":
        return list[dict[str, str]]
    if s.startswith("list<") and s.endswith(">"):
        return list[_parse_type(s[5:-1])]
    if s.startswith("dict<") and s.endswith(">"):
        k, _, v = s[5:-1].partition(",")
        return dict[_parse_type(k), _parse_type(v)]
    raise ValueError(f"unknown fixture type {s!r}")


def build_state_cls(model_name: str, fields_spec: Mapping[str, Mapping[str, Any]]) -> type[State]:
    """Translate a fixture's `state.fields` block into a Pydantic State subclass.

    The `alt_reducer` key (used only by the 007 conflicting_reducers case) is
    treated as a second declared reducer so the resulting field carries two
    reducers in its Annotated metadata — exactly the shape `field_reducers`
    inspects.
    """

    field_defs: dict[str, Any] = {}
    for fname, spec in fields_spec.items():
        py_type = _parse_type(spec["type"])
        reducers = [REDUCERS[spec[k]] for k in ("reducer", "alt_reducer") if k in spec]
        annotation: Any = Annotated[py_type, *reducers] if reducers else py_type

        if "default" in spec:
            raw_default: Any = spec["default"]
            if isinstance(raw_default, list | dict):
                # Mutable default → use a factory so each instance gets its own copy.
                snapshot = copy.deepcopy(cast(Any, raw_default))
                field_defs[fname] = (
                    annotation,
                    Field(default_factory=lambda v=snapshot: copy.deepcopy(v)),
                )
            else:
                field_defs[fname] = (annotation, raw_default)
        else:
            field_defs[fname] = (annotation, ...)

    return create_model(model_name, __base__=State, **field_defs)


def _resolve_target(target: str) -> str | EndSentinel:
    return END if target == "END" else target


def _make_update_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    snapshot = dict(update)

    async def fn(_state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        return copy.deepcopy(snapshot)

    return fn


_RAISES_EXCEPTION_KINDS: dict[str, type[Exception]] = {
    "ValueError": ValueError,
    "RuntimeError": RuntimeError,
    "TypeError": TypeError,
    "KeyError": KeyError,
}


def _make_raising_fn(
    node_name: str,
    raises_spec: str | Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    # Two shapes: a bare message string (fixture 006) raises RuntimeError;
    # a ``{message, exception_kind}`` dict (fixture 063) raises the named
    # exception type with that message (an uncategorized error, so a
    # wrapping failure-isolation event reports a null category).
    if isinstance(raises_spec, Mapping):
        message = str(raises_spec.get("message", ""))
        kind = str(raises_spec.get("exception_kind", "RuntimeError"))
        if kind not in _RAISES_EXCEPTION_KINDS:
            raise ValueError(f"unsupported raises exception_kind: {kind}")
        exc_type = _RAISES_EXCEPTION_KINDS[kind]
    else:
        message = raises_spec
        exc_type = RuntimeError

    async def fn(_state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        raise exc_type(message)

    return fn


def _resolve_success_compute(success_compute: Mapping[str, Any], state: Any) -> dict[str, Any]:
    """Translate a ``success_compute`` mapping into a partial update.

    For each key/value: if the value is a string AND matches a state
    field, treat it as a field reference (read state.<value>). Otherwise
    treat it as a literal — bool/int/str-not-a-field/etc. Used by
    flaky_by_index and flaky_instance_only test seams.
    """
    state_cls = cast("type[State]", type(state))
    model_fields = cast("dict[str, Any]", state_cls.model_fields)
    state_fields: set[str] = set(model_fields.keys())
    result: dict[str, Any] = {}
    for tgt, val in success_compute.items():
        if isinstance(val, str) and val in state_fields:
            result[tgt] = getattr(state, val)
        else:
            result[tgt] = val
    return result


class _CategorizedException(Exception):
    """A test exception carrying a `category` attribute so the default
    retry classifier can match it."""

    def __init__(self, message: str, category: str) -> None:
        super().__init__(message)
        self.category = category


# Conformance-adapter §5.1 ``cause`` (proposal 0070): a failure mock's raised
# error MAY chain to an originating cause, recursively, so a consumer walking
# the cause chain (pipeline-utilities §6.3 failure isolation) observes each
# link's category / message.
def _build_mock_cause(cause_spec: Mapping[str, Any] | None) -> Exception | None:
    """Build the chained originating exception from a failure mock's ``cause``
    directive. ``cause: {category, message, cause: {...}}`` nests recursively;
    each link becomes a ``_CategorizedException`` (or a bare ``Exception`` when
    its category is null) linked via ``__cause__``. Returns ``None`` when no
    cause is configured."""
    if cause_spec is None:
        return None
    inner = _build_mock_cause(cause_spec.get("cause"))
    message = str(cause_spec.get("message", ""))
    category = cause_spec.get("category")
    exc: Exception = (
        _CategorizedException(message=message, category=category)
        if isinstance(category, str) and category
        else Exception(message)
    )
    if inner is not None:
        exc.__cause__ = inner
    return exc


def _make_pure_update_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`update_pure` test seam — applies a fixed update.

    Two shapes coexist across the spec fixtures:

    - Literal values (e.g. ``update_pure: {a_ran: true, count: 0}``)
      — most common, the snapshot is the partial verbatim.
    - Field references (e.g. fixture 050 ``update_pure: {stage1: input}``)
      — when a value is a string AND the state has a field of that
      name, treat the string as a field-name reference and resolve
      to ``state.<input>`` at call time. This handles fixtures that
      use ``update_pure`` to copy one inner field to another without
      a ``multiplier`` (which would route through ``update_from_field``).

    The disambiguation is deliberately lax: a literal-string update
    (e.g. ``update_pure: {label: "foo"}``) accidentally matching a
    state field name would resolve incorrectly. Real fixtures don't
    exercise this overlap; if a future fixture needs both shapes
    disambiguated, prefer ``update_pure_from_state`` for the
    field-reference case and keep ``update_pure`` strictly literal.
    """
    snapshot = dict(update)

    async def fn(state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        resolved: dict[str, Any] = {}
        state_cls = cast("type[Any]", type(state))
        model_fields = cast("dict[str, Any]", getattr(state_cls, "model_fields", {}))
        state_field_names = set(model_fields.keys())
        for k, v in snapshot.items():
            if isinstance(v, str) and v in state_field_names:
                resolved[k] = getattr(state, v)
            else:
                resolved[k] = copy.deepcopy(v)
        return resolved

    return fn


def _make_update_from_field_fn(
    node_name: str,
    update: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`update_from_field` test seam — assigns ``state.<source> * multiplier``
    to ``target``. The fixture format is ``{<target>: <source>, multiplier: N}``;
    e.g. ``{result: x, multiplier: 2}`` means ``result = state.x * 2``.
    Used by the doubler / scorer subgraphs in the fan-out fixtures."""
    cfg = dict(update)
    multiplier = int(cfg.pop("multiplier", 1))
    # The remaining single key→value pair is target_field → source_field.
    if len(cfg) != 1:
        raise ValueError(
            f"update_from_field for {node_name!r} expects exactly one "
            f"target→source mapping plus an optional `multiplier`; got {update!r}"
        )
    target_field, source_field = next(iter(cfg.items()))

    async def fn(state: Any) -> Mapping[str, Any]:
        trace.append(node_name)
        source_value = getattr(state, source_field)
        return {target_field: source_value * multiplier}

    return fn


def _make_flaky_by_index_fn(
    node_name: str,
    cfg: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`flaky_by_index` test seam used by fan-out fixtures.

    Two config shapes exist in the spec fixtures:

    - ``fail_count_per_idx: N`` — fixture 020. Each fan-out instance
      fails its first N ATTEMPTS with a transient exception (so retry
      middleware exercises the retry path). Counter is per instance —
      reconstructed per node activation since each fan-out instance
      gets its own subgraph invocation, so the closure is fresh.
    - ``fail_when_idx: N`` — fixture 019. The instance whose
      ``state.idx`` (item value) equals N fails; others succeed. No
      retry logic; this is the collect-policy test seam.

    Both shapes share ``category`` and ``success_compute``.
    """
    category = cfg.get("category", "provider_unavailable")
    success_compute = dict(cfg.get("success_compute", {}))
    fail_count_per_idx = cfg.get("fail_count_per_idx")
    fail_when_idx = cfg.get("fail_when_idx")
    # Fixture-global attempt counter. Truly per-instance semantics
    # would require an identifier stable across an instance's retries
    # AND distinct across fan-out instances. id(state) works for
    # NODE-LEVEL retry (state stable across retries) but not for
    # INSTANCE-LEVEL retry (each instance retry constructs a fresh
    # subgraph state). A state-field key would work in principle but
    # the field name is fixture-specific. The current fixtures
    # (020 node-level retry, 019 collect mode) pass with this global
    # counter because the timing aligns; documenting the limitation
    # here so a future fixture exercising the gap surfaces a real
    # failure rather than silently miscounting.
    attempt_counter = [0]

    async def fn(state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        if idx == 0:
            trace.append(node_name)
        # fail_when_idx mode: fail when the instance's item value (read
        # from state.idx by convention) equals the configured int.
        if fail_when_idx is not None:
            instance_idx = getattr(state, "idx", None)
            if instance_idx == int(fail_when_idx):
                raise _CategorizedException(
                    message=f"flaky_by_index fail_when_idx={fail_when_idx}",
                    category=category,
                )
        # fail_count_per_idx mode: fail the first N attempts (counted
        # globally across the fan-out).
        elif fail_count_per_idx is not None and idx < int(fail_count_per_idx):
            raise _CategorizedException(
                message=f"flaky_by_index failure at attempt {idx}",
                category=category,
            )
        if success_compute:
            return _resolve_success_compute(success_compute, state)
        return {}

    return fn


def _make_flaky_instance_only_fn(
    node_name: str,
    cfg: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """`flaky_instance_only` test seam — fails on the first call,
    succeeds thereafter. Same fixture-global counter as
    flaky_by_index; see that function's note on the per-instance vs
    per-fixture keying tradeoff. The current fixtures (021
    instance-level retry) align with the global counter at runtime
    even though strict per-instance semantics would need a richer
    identifier."""
    fail_first_only = bool(cfg.get("fail_first_only", True))
    category = cfg.get("category", "provider_unavailable")
    success_compute = dict(cfg.get("success_compute", {}))
    attempt_counter = [0]

    async def fn(state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        if idx == 0:
            trace.append(node_name)
        if fail_first_only and idx == 0:
            raise _CategorizedException(
                message="flaky_instance_only first-attempt failure",
                category=category,
            )
        if success_compute:
            return _resolve_success_compute(success_compute, state)
        return {}

    return fn


def _make_flaky_per_index_fn(
    node_name: str,
    cfg: Mapping[str, Any],
    trace: list[str],
    *,
    instance_attempt_recorder: dict[int, list[int]] | None = None,
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Build a flaky-per-index node body. Two failure-injection shapes
    (per proposal-0009 fixture set 048-054):

    - ``fail_first_run_indices: [int, ...]`` — instances with these
      indices fail on the FIRST CALL EVER (the first-run path); all
      subsequent calls (resume) succeed. The closure tracks "have I
      ever failed" via a shared flag that flips on the first raise.

    - ``always_fail_indices: [int, ...]`` — instances with these
      indices fail on EVERY call. Used by collect-mode fixtures (052)
      where the failure becomes an error contribution on the saved
      record and rolls forward verbatim on resume.

    Both forms share ``success_compute`` for the success-path state
    update.

    Reads ``current_fan_out_index()`` to determine which fan-out
    instance is currently executing. Returns the success_compute output
    for non-failing indices.

    ``instance_attempt_recorder`` (optional): when supplied, the closure
    appends each call's ``current_attempt_index()`` to
    ``instance_attempt_recorder[idx]`` so the test driver can later
    assert per-instance retry-count expectations
    (``instance_N_attempt_index_on_resume`` /
    ``instance_N_resume_attempt_count`` directives).
    """
    from openarmature.observability.correlation import (  # noqa: PLC0415
        current_attempt_index,
        current_fan_out_index,
    )

    fail_first_run_indices = set(cfg.get("fail_first_run_indices") or [])
    always_fail_indices = set(cfg.get("always_fail_indices") or [])
    success_compute = dict(cfg.get("success_compute", {}))
    # Per-index tracking of which ``fail_first_run_indices`` instances
    # have already failed once. The earlier single-flag shape failed
    # only the first index in dispatch order when the list named
    # multiple indices; per-index tracking matches the directive's
    # "fail on FIRST CALL EVER" wording.
    already_failed_indices: set[int] = set()
    traced = [False]

    async def fn(state: Any) -> Mapping[str, Any]:
        if not traced[0]:
            trace.append(node_name)
            traced[0] = True
        idx = current_fan_out_index()
        if idx is None:
            # Defensive — flaky_per_index only makes sense inside a
            # fan-out instance. Surface as a categorized failure so
            # mismatched fixture wiring is loud rather than silent.
            raise _CategorizedException(
                message=f"flaky_per_index({node_name}) called outside a fan-out instance",
                category="node_exception",
            )
        if instance_attempt_recorder is not None:
            instance_attempt_recorder.setdefault(idx, []).append(current_attempt_index())
        if idx in always_fail_indices:
            raise _CategorizedException(
                message=f"flaky_per_index({node_name}) always-fail at idx={idx}",
                category="node_exception",
            )
        if idx in fail_first_run_indices and idx not in already_failed_indices:
            already_failed_indices.add(idx)
            raise _CategorizedException(
                message=f"flaky_per_index({node_name}) first-run failure at idx={idx}",
                category="node_exception",
            )
        if success_compute:
            return _resolve_success_compute(success_compute, state)
        return {}

    return fn


def _make_flaky_fn(
    node_name: str,
    flaky: Mapping[str, Any],
    trace: list[str],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Build a flaky node body. Two shapes:

    - ``failure_sequence: [...]`` — per-attempt failures keyed to
      attempt index. Used by fixture 007 + retry middleware tests.
      When the sequence is exhausted, subsequent attempts return
      ``success_update``.
    - ``fail_first_invocation_only: true`` — fails on the first call,
      then promotes itself; subsequent calls return
      ``on_success``. Used by checkpoint resume fixtures (025, 029,
      031): the engine's first invoke aborts on this node; the
      resumed invoke succeeds. No retry middleware is wrapped around
      these nodes (any wrapping would bypass the resume path), so
      "fail-once-then-succeed" matches the resume contract directly.
    """
    sequence = list(flaky.get("failure_sequence", []))
    success_update = dict(flaky.get("success_update", {}))
    attempt_counter = [0]
    fail_first_invocation_only = bool(flaky.get("fail_first_invocation_only"))
    has_failed_once = [False]
    on_success = dict(flaky.get("on_success", {}))

    async def fn(_state: Any) -> Mapping[str, Any]:
        idx = attempt_counter[0]
        attempt_counter[0] = idx + 1
        # `execution_order` is engine-step-scoped, not per-attempt — only
        # append on the first attempt so retry middleware re-invocations
        # don't double-count the node.
        if idx == 0:
            trace.append(node_name)
        if fail_first_invocation_only:
            # Promote-on-first-failure model: the failure is what
            # marks "first invocation has happened." Subsequent calls
            # (which arrive only after a resume — the abort tore
            # down the original invoke) return on_success.
            if not has_failed_once[0]:
                has_failed_once[0] = True
                raise _CategorizedException(
                    message=f"flaky({node_name}) first-invocation failure",
                    category="node_exception",
                )
            return copy.deepcopy(on_success)
        if idx < len(sequence):
            entry = sequence[idx]
            if entry is None:
                return copy.deepcopy(success_update)
            # An entry MAY carry a recursive ``cause`` (proposal 0070 §5.1)
            # that chains the raised error to an originating cause.
            cause_exc = _build_mock_cause(entry.get("cause"))
            exc = _CategorizedException(
                message=entry.get("message", "flaky"),
                category=entry.get("category", "provider_unavailable"),
            )
            if cause_exc is not None:
                raise exc from cause_exc
            raise exc
        return copy.deepcopy(success_update)

    return fn


def _wrap_with_sleep(
    fn: Callable[[Any], Awaitable[Mapping[str, Any]]],
    sleep_ms: int,
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    # ``sleep_ms`` companion modifier on a NodeSpec — sleep that many
    # milliseconds before the wrapped body fires. Used by parallel-branches
    # fixtures 033 (slow third branch for fail-fast cancellation) and 037
    # (randomized completion timing to verify insertion-order determinism).
    delay = sleep_ms / 1000.0

    async def fn_with_sleep(state: Any) -> Mapping[str, Any]:
        await asyncio.sleep(delay)
        return await fn(state)

    return fn_with_sleep


def _wrap_with_execution_recorder(
    fn: Callable[[Any], Awaitable[Mapping[str, Any]]],
    node_name: str,
    recorders: dict[str, dict[int, list[int]]],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Wrap a node body so that, when it runs inside a fan-out instance, it
    records the executing instance's ``current_fan_out_index()`` into
    ``recorders`` (keyed by node name then index). Lets the checkpoint resume
    driver tell which fan-out instances executed vs. rolled forward for a
    plain-node fan-out (e.g. the crash_injection fixture 067), where no
    ``flaky_per_index`` body records execution. Records at body entry, so an
    instance whose body ran counts as executed even if it then fails."""
    from openarmature.observability.correlation import (  # noqa: PLC0415
        current_attempt_index,
        current_fan_out_index,
    )

    async def fn_recording(state: Any) -> Mapping[str, Any]:
        idx = current_fan_out_index()
        if idx is not None:
            recorders.setdefault(node_name, {}).setdefault(idx, []).append(current_attempt_index())
        return await fn(state)

    return fn_recording


@dataclass(frozen=True)
class _TracingFanOutNode(FanOutNode[State, State]):
    """Conformance helper: a FanOutNode that appends its name to a shared
    trace list when the engine runs it. Same role as _TracingSubgraphNode
    for subgraphs — a fan-out node is one engine step from the parent's
    POV, so it should contribute exactly one trace entry."""

    trace_list: list[str] = field(default_factory=list[str])

    async def run_with_context(
        self,
        state: State,
        context: _InvocationContext,
        *,
        pre_resolved_count: int | None = None,
        pre_resolved_concurrency: tuple[int | None] | None = None,
    ) -> Mapping[str, Any]:
        self.trace_list.append(self.name)
        return await super().run_with_context(
            state,
            context,
            pre_resolved_count=pre_resolved_count,
            pre_resolved_concurrency=pre_resolved_concurrency,
        )


@dataclass(frozen=True)
class _TracingParallelBranchesNode(ParallelBranchesNode[State]):
    """Conformance helper: a ParallelBranchesNode that appends its name
    to the shared trace list once when the engine runs it. The
    parallel-branches dispatcher itself counts as one engine step from
    the parent's POV, mirroring the fan-out tracing wrapper."""

    trace_list: list[str] = field(default_factory=list[str])

    async def run_with_context(
        self,
        state: State,
        context: _InvocationContext,
    ) -> Mapping[str, Any]:
        self.trace_list.append(self.name)
        return await super().run_with_context(state, context)


@dataclass(frozen=True)
class _TracingSubgraphNode(SubgraphNode[State, State]):
    """Conformance helper: a SubgraphNode that appends its name to a shared
    trace list when the engine runs it.

    Lets the conformance adapter use real SubgraphNode (so observer-context
    threading works for fixture 013, and compile-time projection validation
    works for the mapping_references_undeclared_field 007 case) while still
    supporting `execution_order` assertions that include the wrapper name —
    the engine itself doesn't dispatch an event for the wrapper per fixture
    013's spec.
    """

    trace_list: list[str] = field(default_factory=list[str])

    async def run(
        self,
        state: State,
        context: _InvocationContext | None = None,
    ) -> Mapping[str, Any]:
        self.trace_list.append(self.name)
        return await super().run(state, context=context)


def _make_conditional_fn(
    if_field: str,
    equals: Any,
    then: str,
    else_: str,
) -> Callable[[Any], str | EndSentinel]:
    then_target = _resolve_target(then)
    else_target = _resolve_target(else_)

    def fn(state: Any) -> str | EndSentinel:
        return then_target if getattr(state, if_field) == equals else else_target

    return fn


@dataclass
class BuiltGraph:
    """Result of translating a fixture into runnable engine constructs."""

    state_cls: type[State]
    builder: GraphBuilder[State]
    trace: list[str]

    def initial_state(self, overrides: Mapping[str, Any]) -> State:
        return self.state_cls(**overrides)


def _projection_for(node_spec: Mapping[str, Any]) -> ProjectionStrategy[State, State]:
    """Pick the projection strategy declared on a subgraph node spec.

    `inputs:` and/or `outputs:` in the YAML → `ExplicitMapping`. Both absent →
    the default `FieldNameMatching`.
    """

    inputs = node_spec.get("inputs")
    outputs = node_spec.get("outputs")
    if inputs is None and outputs is None:
        return FieldNameMatching[State, State]()
    return ExplicitMapping[State, State](inputs=inputs, outputs=outputs)


def build_graph(
    spec: Mapping[str, Any],
    *,
    subgraphs: Mapping[str, CompiledGraph[State]] | None = None,
    trace: list[str] | None = None,
    model_name: str = "FixtureState",
    node_middleware: Mapping[str, Sequence[Any]] | None = None,
    graph_middleware: Sequence[Any] | None = None,
    fan_out_instance_middleware: Mapping[str, Sequence[Any]] | None = None,
    parallel_branches_branch_middleware: Mapping[str, Mapping[str, Sequence[Any]]] | None = None,
    flaky_per_index_attempt_recorders: dict[str, dict[int, list[int]]] | None = None,
    instance_execution_recorders: dict[str, dict[int, list[int]]] | None = None,
) -> BuiltGraph:
    """Translate a graph-shaped fixture block into a `BuiltGraph`.

    `spec` is the top-level fixture mapping for plain fixtures, or the inner
    `graph:` block for the table-style 007 cases. `subgraphs` is the registry
    used by 006-style fixtures to look up a compiled subgraph by its declared
    name.

    `node_middleware` (mapping node name to ordered middleware list) and
    `graph_middleware` (ordered middleware list applied to every node)
    are middleware hooks. The translation from a fixture's
    `middleware:` block into actual instances lives in the
    pipeline-utilities test driver.

    Subgraph references in `spec.nodes` resolve to `_TracingSubgraphNode`
    (a SubgraphNode subclass) so the engine threads observer context through
    AND the conformance adapter's `execution_order` trace gets the wrapper
    name appended when it runs.
    """

    state_cls = build_state_cls(model_name, spec["state"]["fields"])
    builder = GraphBuilder(state_cls)
    if "entry" in spec:
        builder.set_entry(spec["entry"])

    trace = trace if trace is not None else []
    subgraphs = subgraphs or {}
    node_middleware = node_middleware or {}
    fan_out_instance_middleware = fan_out_instance_middleware or {}
    parallel_branches_branch_middleware = parallel_branches_branch_middleware or {}

    for mw in graph_middleware or ():
        builder.add_middleware(mw)

    for node_name, node_spec in spec.get("nodes", {}).items():
        per_node_mw = tuple(node_middleware.get(node_name, ()))
        if "subgraph" in node_spec:
            sub_name = node_spec["subgraph"]
            compiled = subgraphs[sub_name]
            projection = _projection_for(node_spec)
            if node_name in builder._nodes:
                raise ValueError(f"node {node_name!r} already declared")
            builder._nodes[node_name] = _TracingSubgraphNode(
                name=node_name,
                compiled=compiled,
                projection=projection,
                trace_list=trace,
                middleware=per_node_mw,
                subgraph_identity=sub_name,
            )
            continue
        if "fan_out" in node_spec:
            _add_fan_out_node(
                builder,
                node_name,
                node_spec["fan_out"],
                subgraphs,
                trace,
                instance_middleware=fan_out_instance_middleware.get(node_name, ()),
            )
            continue
        if "parallel_branches" in node_spec:
            _add_parallel_branches_node(
                builder,
                node_name,
                node_spec["parallel_branches"],
                subgraphs,
                trace,
                branch_middleware=parallel_branches_branch_middleware.get(node_name, {}),
            )
            continue

        body: Callable[[Any], Awaitable[Mapping[str, Any]]]
        if "raises" in node_spec:
            body = _make_raising_fn(node_name, node_spec["raises"], trace)
        elif "flaky" in node_spec:
            body = _make_flaky_fn(node_name, node_spec["flaky"], trace)
        elif "flaky_by_index" in node_spec:
            body = _make_flaky_by_index_fn(node_name, node_spec["flaky_by_index"], trace)
        elif "flaky_instance_only" in node_spec:
            body = _make_flaky_instance_only_fn(node_name, node_spec["flaky_instance_only"], trace)
        elif "flaky_per_index" in node_spec:
            recorder: dict[int, list[int]] | None = None
            if flaky_per_index_attempt_recorders is not None:
                recorder = flaky_per_index_attempt_recorders.setdefault(node_name, {})
            body = _make_flaky_per_index_fn(
                node_name,
                node_spec["flaky_per_index"],
                trace,
                instance_attempt_recorder=recorder,
            )
        elif "update" in node_spec:
            body = _make_update_fn(node_name, node_spec["update"], trace)
        elif "update_pure" in node_spec:
            body = _make_pure_update_fn(node_name, node_spec["update_pure"], trace)
        elif "update_from_field" in node_spec:
            body = _make_update_from_field_fn(node_name, node_spec["update_from_field"], trace)
        else:
            raise ValueError(
                f"node {node_name!r} has no recognized directive "
                "(update / update_pure / update_from_field / raises / flaky / "
                "flaky_by_index / flaky_instance_only / flaky_per_index / fan_out / "
                "parallel_branches / subgraph)"
            )

        sleep_ms = node_spec.get("sleep_ms")
        if sleep_ms is not None:
            body = _wrap_with_sleep(body, int(sleep_ms))

        # Record per-instance execution for plain-node fan-outs so the
        # checkpoint resume driver can tell executed from rolled-forward
        # instances. flaky_per_index records its own per-instance attempts,
        # so it is skipped here; this covers the rest.
        if instance_execution_recorders is not None and "flaky_per_index" not in node_spec:
            body = _wrap_with_execution_recorder(body, node_name, instance_execution_recorders)

        builder.add_node(node_name, body, middleware=per_node_mw)

    for edge_spec in spec.get("edges", []):
        source = edge_spec["from"]
        if "to" in edge_spec:
            builder.add_edge(source, _resolve_target(edge_spec["to"]))
        elif "condition" in edge_spec:
            cond = edge_spec["condition"]
            builder.add_conditional_edge(
                source,
                _make_conditional_fn(cond["if_field"], cond["equals"], cond["then"], cond["else"]),
            )
        else:
            raise ValueError(f"edge from {source!r} has neither `to` nor `condition`")

    return BuiltGraph(state_cls=state_cls, builder=builder, trace=trace)


# ---------------------------------------------------------------------------
# Observer fixture support (spec v0.3.0 §6, fixtures 012–015)
# ---------------------------------------------------------------------------


@dataclass
class ObserverFixture:
    """Captured per-observer state for assertion against an observer fixture.

    Built once per observer declared in a fixture's `observers:` block. The
    observer callable produced by `make_observer_fn` records every event it
    receives into `events` and (if behavior == "raise") raises after
    recording.

    `phases` is the optional subscription set parsed from the fixture's
    YAML. None means "no `phases:` key was present" — the harness leaves
    the engine to default to both phases.

    `sleep_ms_per_event` configures the slow-observer directive. When
    `None`, the observer runs at full
    speed. An int means a constant sleep per event. A dict with
    `first_invocation` / `subsequent_invocations` keys is invocation-
    counter-aware: the first invocation through this observer uses the
    `first_invocation` value, every subsequent invocation uses the
    `subsequent_invocations` value. `invocation_counter` is bumped by the
    harness between invocations.
    """

    name: str
    attach: str  # "graph" | "invocation"
    target: str  # "outer" | <subgraph name>
    behavior: str  # "record" | "raise"
    phases: frozenset[str] | None = None
    sleep_ms_per_event: int | Mapping[str, int] | None = None
    invocation_counter: list[int] = field(default_factory=lambda: [0])
    events: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


def _record_event(event: NodeEvent) -> dict[str, Any]:
    """Convert a NodeEvent into a dict matching the YAML expected shape."""
    rec: dict[str, Any] = {
        "step": event.step,
        "phase": event.phase,
        "node_name": event.node_name,
        "namespace": list(event.namespace),
        "pre_state": event.pre_state.model_dump(),
        "parent_states": [ps.model_dump() for ps in event.parent_states],
        "attempt_index": event.attempt_index,
    }
    if event.post_state is not None:
        rec["post_state"] = event.post_state.model_dump()
    if event.error is not None:
        rec["error"] = event.error.category
    if event.fan_out_index is not None:
        rec["fan_out_index"] = event.fan_out_index
    if event.branch_name is not None:
        rec["branch_name"] = event.branch_name
    return rec


def _resolve_sleep_ms(fixture: ObserverFixture) -> int:
    """Resolve the per-event sleep duration in ms for the slow-observer
    directive. `None` and `0` mean no sleep;
    an int form is constant; a dict form selects by `invocation_counter`.
    """
    spec = fixture.sleep_ms_per_event
    if spec is None:
        return 0
    if isinstance(spec, int):
        return spec
    # Dict form with first_invocation / subsequent_invocations keys —
    # used by fixture 024 to slow only the first invocation so the
    # second drain runs cleanly. `invocation_counter[0]` is bumped by
    # the harness between `compiled.invoke()` calls.
    if fixture.invocation_counter[0] == 0:
        return int(spec.get("first_invocation", 0))
    return int(spec.get("subsequent_invocations", 0))


def make_observer_fn(
    fixture: ObserverFixture,
    delivery: list[tuple[str, int, str]],
) -> Observer:
    """Build the async observer callable for an `ObserverFixture`.

    Records every event into `fixture.events` and appends
    `(name, step, phase)` to the shared `delivery` list (the order
    observers are called in across the whole invocation, used to assert
    `delivery_order`). Raising observers record + append before raising,
    so the engine's error isolation can be verified by checking that
    subsequent observers/events still get through.

    Honors `fixture.sleep_ms_per_event` per the slow-observer
    directive: each event awaits `asyncio.sleep(ms / 1000)`
    BEFORE recording, so a drain timeout that cancels mid-sleep leaves
    the event unrecorded and the counter shows it as undelivered.
    """

    async def observer(event: ObserverEvent) -> None:
        if not isinstance(event, NodeEvent):
            return
        sleep_ms = _resolve_sleep_ms(fixture)
        if sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000.0)
        delivery.append((fixture.name, event.step, event.phase))
        fixture.events.append(_record_event(event))
        if fixture.behavior == "raise":
            raise RuntimeError(f"{fixture.name} raised on event at step {event.step}")

    return observer


def normalize_expected_event(ev: Mapping[str, Any]) -> dict[str, Any]:
    """Fill in defaults for keys the YAML omits, so equality with the
    recorded event dict works as-is. Fixtures don't repeat the
    `attempt_index: 0` and `parent_states: []` defaults for every event;
    the engine emits both unconditionally, so backfill them here.
    """
    e = dict(ev)
    e.setdefault("parent_states", [])
    e.setdefault("attempt_index", 0)
    return e


def _add_fan_out_node(
    builder: GraphBuilder[Any],
    node_name: str,
    cfg: Mapping[str, Any],
    subgraphs: Mapping[str, CompiledGraph[State]],
    trace: list[str],
    *,
    instance_middleware: Sequence[Any] = (),
) -> None:
    """Translate a fixture's ``fan_out:`` block into a builder.add_fan_out_node
    call.

    Resolves callable forms of ``count`` and ``concurrency``:

    - ``state_field_read`` — read an int from a parent state field.
    - ``queue_chunk`` — ``max(1, len(state.<field>) // chunk_size)``.

    These are the only callable shapes the in-scope fan-out fixtures
    use. Adding more is straightforward.
    """
    sub_name = cfg["subgraph"]
    sub_compiled = subgraphs[sub_name]

    count_raw = cfg.get("count")
    count: int | Callable[[Any], int] | None = None
    if isinstance(count_raw, dict):
        count = _resolve_callable_int_resolver(cast("dict[str, Any]", count_raw))
    elif count_raw is not None:
        count = int(count_raw)

    # ``concurrent_mode: serial`` (proposal-0009 fixture set 048-054)
    # is harness sugar for ``concurrency=1`` — forces deterministic
    # per-instance completion ordering for resume-correctness assertions.
    # Takes precedence over an explicit ``concurrency`` value if both
    # are present.
    concurrent_mode = cfg.get("concurrent_mode")
    if concurrent_mode == "serial":
        conc_raw: Any = 1
    else:
        conc_raw = cfg.get("concurrency", 10)
    conc: int | Callable[[Any], int | None] | None
    if isinstance(conc_raw, dict):
        conc = cast(
            "Callable[[Any], int | None]",
            _resolve_callable_int_resolver(cast("dict[str, Any]", conc_raw)),
        )
    elif conc_raw is None:
        conc = None
    else:
        conc = int(conc_raw)

    builder.add_fan_out_node(
        node_name,
        subgraph=sub_compiled,
        collect_field=cfg["collect_field"],
        target_field=cfg["target_field"],
        items_field=cfg.get("items_field"),
        item_field=cfg.get("item_field"),
        count=count,
        concurrency=conc,
        error_policy=cfg.get("error_policy", "fail_fast"),
        on_empty=cfg.get("on_empty", "raise"),
        count_field=cfg.get("count_field"),
        inputs=cfg.get("inputs"),
        extra_outputs=cfg.get("extra_outputs"),
        errors_field=cfg.get("errors_field"),
        instance_middleware=instance_middleware,
        subgraph_identity=sub_name,
    )

    # Swap the registered FanOutNode for a tracing variant so the
    # conformance trace records the fan-out as one engine step. The
    # builder's compile-time validation already ran above; we only
    # replace the stored Node instance.
    original = cast("FanOutNode[State, State]", builder._nodes[node_name])
    builder._nodes[node_name] = _TracingFanOutNode(
        name=original.name,
        config=original.config,
        middleware=original.middleware,
        trace_list=trace,
    )


# Spec proposal 0075: callable branch ``call:`` directive.
def _build_call_fn(
    branch_name: str,
    call_cfg: Mapping[str, Any],
) -> Callable[[Any], Awaitable[Mapping[str, Any]]]:
    """Translate a callable branch's ``call:`` directive
    into an async function over the parent state.

    Reuses the node-behavior factories (``update`` / ``flaky`` /
    ``raises``) keyed by the same directive shape a plain node uses — a
    callable branch IS just a function. The callable's own trace
    recording is discarded into a throwaway sink: a parallel-branches
    node records as a single dispatch step, so callable-branch bodies
    must not appear in ``execution_order``.
    """
    sink: list[str] = []
    if "update" in call_cfg:
        return _make_update_fn(branch_name, call_cfg["update"], sink)
    if "flaky" in call_cfg:
        return _make_flaky_fn(branch_name, call_cfg["flaky"], sink)
    if "raises" in call_cfg:
        return _make_raising_fn(branch_name, call_cfg["raises"], sink)
    raise ValueError(f"callable branch {branch_name!r}: unsupported call directive {dict(call_cfg)!r}")


# Spec proposal 0075 §11.10: branch ``when:`` predicate.
def _build_when_predicate(when_cfg: Mapping[str, Any]) -> Callable[[Any], bool]:
    """Translate a branch ``when:`` directive into
    a parent-state predicate. Supports ``{field: <name>}`` — a truthy
    check on a parent-state field at dispatch time."""
    if "field" in when_cfg:
        field_name = cast("str", when_cfg["field"])

        def predicate(state: Any) -> bool:
            return bool(getattr(state, field_name))

        return predicate
    raise ValueError(f"unsupported when directive: {dict(when_cfg)!r}")


def _add_parallel_branches_node(
    builder: GraphBuilder[Any],
    node_name: str,
    cfg: Mapping[str, Any],
    subgraphs: Mapping[str, CompiledGraph[State]],
    trace: list[str],
    *,
    branch_middleware: Mapping[str, Sequence[Any]],
) -> None:
    """Translate a fixture's ``parallel_branches:`` block into a
    ``builder.add_parallel_branches_node`` call.

    A branch is either a ``subgraph`` (name resolved against the shared
    ``subgraphs`` registry, with optional ``inputs`` / ``outputs``) or an
    inline ``call``, and may carry a ``when`` predicate.
    ``branch_middleware`` maps branch-name to a pre-translated middleware
    list; the test driver populates it from each branch's ``middleware:``
    block.
    """
    branches_cfg = cast("dict[str, dict[str, Any]]", cfg["branches"])
    branches: dict[str, BranchSpec[Any]] = {}
    for branch_name, branch_cfg in branches_cfg.items():
        when_cfg = branch_cfg.get("when")
        when = _build_when_predicate(cast("Mapping[str, Any]", when_cfg)) if when_cfg is not None else None
        if "call" in branch_cfg:
            branches[branch_name] = BranchSpec(
                call=_build_call_fn(branch_name, cast("Mapping[str, Any]", branch_cfg["call"])),
                when=when,
                middleware=tuple(branch_middleware.get(branch_name, ())),
            )
            continue
        sub_compiled = subgraphs[branch_cfg["subgraph"]]
        branches[branch_name] = BranchSpec(
            subgraph=sub_compiled,
            inputs=dict(branch_cfg.get("inputs") or {}),
            outputs=dict(branch_cfg.get("outputs") or {}),
            when=when,
            middleware=tuple(branch_middleware.get(branch_name, ())),
        )

    builder.add_parallel_branches_node(
        node_name,
        branches=branches,
        error_policy=cfg.get("error_policy", "fail_fast"),
        errors_field=cfg.get("errors_field"),
    )

    # Swap the registered node for a tracing variant so the
    # conformance trace records the dispatcher as one engine step. The
    # builder's validation has already run; we only replace the stored
    # Node instance.
    original = cast("ParallelBranchesNode[State]", builder._nodes[node_name])
    builder._nodes[node_name] = _TracingParallelBranchesNode(
        name=original.name,
        branches=original.branches,
        error_policy=original.error_policy,
        errors_field=original.errors_field,
        middleware=original.middleware,
        trace_list=trace,
    )


def _resolve_callable_int_resolver(cfg: Mapping[str, Any]) -> Callable[[Any], int]:
    """Build a state-reader callable from a fixture's callable config.

    Supported shapes per the fan-out fixtures:

    - ``{callable: state_field_read, field: <name>}`` — return
      ``state.<name>`` (must be int).
    - ``{callable: queue_chunk, field: <name>, chunk_size: N}`` —
      return ``max(1, len(state.<name>) // N)``.
    """
    kind = cfg["callable"]
    field_name = cfg["field"]
    if kind == "state_field_read":

        def state_field_read(state: Any) -> int:
            return int(getattr(state, field_name))

        return state_field_read
    if kind == "queue_chunk":
        chunk_size = int(cfg["chunk_size"])

        def queue_chunk(state: Any) -> int:
            return max(1, len(getattr(state, field_name)) // chunk_size)

        return queue_chunk
    raise ValueError(f"unknown callable resolver: {kind!r}")
