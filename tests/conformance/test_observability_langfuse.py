# Spec mapping (observability §8): drives the Langfuse mapping
# fixtures (022 basic-trace, 023 generation-rendering, 024
# prompt-linkage) against the in-memory LangfuseObserver client.
# Sibling of test_observability.py (OTel mapping); shares no harness
# state with the OTel side — each fixture builds its own graph +
# observer instance.
#
# The harness also supports the graph-topology shapes used by
# 031/032/033 (subgraph / fan-out / detached-trace) via the
# cross-capability adapter.build_graph helper, but activation of
# those three fixtures is currently deferred — see the
# `_LANGFUSE_FIXTURES` frozenset comment for the gating questions.

"""Run spec observability Langfuse conformance fixtures."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
import yaml

from openarmature.graph import END, ExplicitMapping, GraphBuilder
from openarmature.llm import OpenAIProvider
from openarmature.llm.response import RuntimeConfig
from openarmature.observability.langfuse import (
    InMemoryLangfuseClient,
    LangfuseObservation,
    LangfuseObserver,
    LangfuseTrace,
)
from openarmature.prompts import (
    Prompt,
    PromptManager,
    SamplingConfig,
    TextPrompt,
)
from openarmature.prompts.context import with_active_prompt

from .adapter import build_graph, build_state_cls

CONFORMANCE_DIR = (
    Path(__file__).resolve().parents[2] / "openarmature-spec" / "spec" / "observability" / "conformance"
)


_LANGFUSE_FIXTURES = frozenset(
    {
        "022-langfuse-basic-trace",
        "023-langfuse-generation-rendering",
        "024-langfuse-prompt-linkage",
        # 027 — proposal 0034 (caller-supplied metadata propagation
        # into ``trace.metadata`` + every ``observation.metadata``
        # per §8.4.1 + §8.4.2).
        "027-langfuse-caller-supplied-metadata",
        # 031 / 032 / 033 — proposal 0035. Activated against spec
        # v0.27.1, which patched the two fixture-vs-impl ambiguities
        # raised in coord thread `clarify-subgraph-name-semantics`
        # (msg 04): fixture 031's `outer_out` step corrected 2 → 3
        # (graph-engine §6 shared-counter), and fixture 033's
        # detached-trace inner namespace corrected to the wrapper
        # node name (`["dispatch", "step"]`). The Option A
        # subgraph_identity wiring on main satisfies both.
        "031-langfuse-subgraph-span-hierarchy",
        "032-langfuse-fan-out-per-instance-spans",
        "033-langfuse-detached-trace-mode",
        # 034 — proposal 0040 outermost-serial open-span update.
        # Single-node graph; the ``augment_metadata`` directive on
        # the node body injects a ``set_invocation_metadata`` call
        # before the LLM call, exercising the §3.4 MUST that open
        # spans in the augmenting context's lineage update in place.
        "034-caller-metadata-open-span-update-serial",
        # 037 — proposal 0043 (trace.input/output sourcing). The four
        # decision-tree cases (default stub / disable_state_payload=False
        # / hooks non-null / hooks null-fallthrough) activate via the
        # caller-hook registry below; case 5 (resume re-fire) stays
        # deferred to a follow-up PR — it needs the langfuse harness to
        # grow checkpointer wiring + flaky-node test seam + two-phase
        # multi-trace assertion. Listed individually in
        # ``_DEFERRED_CASES`` rather than at the fixture level so the
        # four other cases run.
        "037-langfuse-trace-input-output",
        # 035/036 — proposal 0039 caller-invocation-id -> trace.id derivation
        # (UUID hex dashes-stripped / sha256-first-16 for a non-UUID). 059 —
        # proposal 0052 implementation-attribution rows on trace.metadata.
        # Wired here (the Langfuse conformance home) by the fixture-harness
        # catch-up; previously unit-only.
        "035-caller-invocation-id-uuid",
        "036-caller-invocation-id-non-uuid",
        "059-implementation-attribution-langfuse",
        # 029 (fan-out per-instance): the fixture omits collect_field/
        # target_field on the fan_out cfg and the inner subgraph omits a
        # state: block, both required by the cross-cap adapter. The harness
        # synthesizes the inner state (_infer_state_fields_from_nodes) and a
        # throwaway aggregation sink (_synthesize_fan_out_aggregation) -- 029
        # asserts per-instance span metadata + sibling isolation, never the
        # collected results -- and augment_metadata_from_field drives the
        # per-instance set_invocation_metadata.
        "029-caller-metadata-fan-out-per-instance",
        # 030 (parallel-branches per-branch): the LangfuseObserver now
        # synthesizes the per-branch dispatch-span observation (§4.3 + §8.4.2 +
        # proposal 0044) that inner branch nodes parent under, ported from the
        # OTel observer's parallel_branches_branch_spans machinery.
        "030-caller-metadata-parallel-branches-per-branch",
        # 039 (nested-lineage augmentation, proposal 0045): the LangfuseObserver
        # gained prefix-general fan-out-instance dispatch (so a fan-out under a
        # serial wrapper parents correctly) and skips shared-parent NODEs in the
        # augmentation walk (0045 §3.4 MUST-NOT). Case 3 (fan-out in a serial
        # subgraph) is wired via the dedicated hand-built _build_039_graph runner;
        # cases 1 + 2 are TEMPORARILY deferred via _DEFERRED_CASES pending the
        # shared nested-dispatch-keying fix (see that note).
        "039-nested-lineage-augmentation",
    }
)


# Per-case deferrals within an otherwise-activated fixture.  Each entry is
# ``(fixture_stem, case_name)``.  The case-loop in the runner ``continue``s
# past matching cases — NOT ``pytest.skip``, which would skip the whole
# fixture's test invocation and hide the surrounding cases that DO run.
_DEFERRED_CASES: frozenset[tuple[str, str]] = frozenset(
    {
        # 039 cases 1 + 2 are TEMPORARILY deferred pending one deeper observer
        # fix shared by both: dispatch keys
        # (fan_out_instance_observations / parallel_branches_branch_spans) are
        # namespace-local and do NOT encode the enclosing fan-out instance, so a
        # dispatch INSIDE an outer fan-out instance collides across instances --
        # case 1's inner instance dispatch and case 2's per-branch dispatch both
        # reparent the second outer instance's events under the first's dispatch.
        # The fix (thread the enclosing fan_out_index_chain / branch_name_chain
        # into the dispatch keys, across synthesis + resolution + the
        # augmentation walk, in both observers) is its own focused effort + spec
        # coordination. Case 3 (single fan-out level under a serial wrapper) does
        # not nest dispatches, so it is wired. See _build_039_graph.
        ("039-nested-lineage-augmentation", "inner_fan_out_augmenter_propagates_to_outer_dispatch_span"),
        ("039-nested-lineage-augmentation", "parallel_branch_augmenter_propagates_to_outer_fan_out_instance"),
    }
)


# Mocks the spec fixture 037 references for ``trace_input_from_state`` /
# ``trace_output_from_state`` caller hooks.  Each YAML hook name maps to
# a Python callable matching the spec fixture's documented mock
# convention (see fixture 037's case 3 / case 4 / case 5 inline comments).
def _returns_job_input_summary(_state: Any) -> dict[str, Any]:
    return {"summary": "job-input"}


def _returns_job_output_summary(_state: Any) -> dict[str, Any]:
    return {"summary": "job-output"}


def _returns_null(_state: Any) -> None:
    return None


def _returns_state_snapshot(state: Any) -> dict[str, Any]:
    # Fixture 037 case 5: the hook captures the state's full field set
    # at hook-fire time.  ``model_dump()`` returns the JSON-able
    # representation; the case asserts the trace's input/output exactly
    # match the values present at first-invoke entry / first-invoke
    # failure-exit / resumed-invoke entry / resumed-invoke exit.
    return cast("dict[str, Any]", state.model_dump())


_TRACE_IO_HOOK_REGISTRY: dict[str, Callable[[Any], Any]] = {
    "returns_job_input_summary": _returns_job_input_summary,
    "returns_job_output_summary": _returns_job_output_summary,
    "returns_null": _returns_null,
    "returns_state_snapshot": _returns_state_snapshot,
}


def _resolve_trace_io_hook(name: str) -> Callable[[Any], Any]:
    """Look up a YAML-named trace_io hook in the registry.  Raises a
    clear KeyError when the fixture references a name the harness
    hasn't mocked yet — surfaces missing-mock issues at test setup
    rather than as a downstream None/AttributeError.
    """
    try:
        return _TRACE_IO_HOOK_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(
            f"trace_io hook {name!r} not registered; known: {sorted(_TRACE_IO_HOOK_REGISTRY)}"
        ) from exc


def _normalize_fan_out_subgraph_keys(spec: dict[str, Any]) -> None:
    """In-place rename of fan-out config keys that fixture 029 uses
    but the cross-capability adapter doesn't:

    - ``inner_subgraph`` → ``subgraph`` (within each ``fan_out`` block)
    - top-level ``inner_subgraphs`` → ``subgraphs``

    The directive intent is identical; only the key naming differs
    across the spec fixture style and the cross-cap adapter's
    parser. Keep the original keys intact in the source spec; this
    function mutates a deepcopy in the harness wrapper.
    """
    if "inner_subgraphs" in spec and "subgraphs" not in spec:
        spec["subgraphs"] = spec.pop("inner_subgraphs")
    for node_spec in cast("dict[str, Any]", spec.get("nodes") or {}).values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        fan_out_cfg = cast("dict[str, Any] | None", node_dict.get("fan_out"))
        if fan_out_cfg is None:
            continue
        if "inner_subgraph" in fan_out_cfg and "subgraph" not in fan_out_cfg:
            fan_out_cfg["subgraph"] = fan_out_cfg.pop("inner_subgraph")


def _synthesize_fan_out_aggregation(spec: dict[str, Any]) -> None:
    """Synthesize a throwaway aggregation sink for a ``fan_out`` block that omits
    ``collect_field`` / ``target_field`` (fixture 029): collect the inner
    subgraph's ``stores_response_in`` value into a fresh outer list field, and
    declare the adapter's other required fan-out fields (``item_field``, the
    ``items_field`` source, the inner state).

    Call AFTER ``_normalize_fan_out_subgraph_keys`` so the ``subgraph(s)`` keys
    are already resolved.
    """
    # 029 asserts per-instance span metadata + sibling isolation, never the
    # collected results, so the sink only satisfies the adapter's collect/target
    # requirement. The inner subgraph (spec["subgraphs"]) is shared across a
    # fixture's cases, so its state seed below uses setdefault to stay
    # idempotent; the outer state / fan_out writes are per-case.
    subgraphs = cast("dict[str, Any]", spec.get("subgraphs") or {})
    state_block = cast("dict[str, Any]", spec.get("state") or {})
    state_fields = cast("dict[str, Any]", state_block.get("fields") or {})
    for node_name, node_spec in cast("dict[str, Any]", spec.get("nodes") or {}).items():
        if not isinstance(node_spec, dict):
            continue
        fan_out_cfg = cast("dict[str, Any] | None", cast("dict[str, Any]", node_spec).get("fan_out"))
        if fan_out_cfg is None or ("collect_field" in fan_out_cfg and "target_field" in fan_out_cfg):
            continue
        sub_name = cast("str | None", fan_out_cfg.get("subgraph"))
        sub_spec = cast("dict[str, Any]", subgraphs.get(sub_name or "", {}))
        inferred = _infer_state_fields_from_nodes(cast("dict[str, Any]", sub_spec.get("nodes") or {}))
        # Any inferred field works as the collected value; the sink is never
        # asserted, so the first one is fine.
        collect_field = next(iter(inferred), None)
        if collect_field is None:
            continue
        # The sink is node-scoped (one outer field per fan-out node). The item
        # slot is a fixed inner-state field name on purpose: distinct fan-outs
        # have distinct inner subgraphs, and a subgraph shared between two
        # fan-outs reuses the one slot, so scoping it per node would mismatch.
        sink = f"oa_fan_out_sink_{node_name}"
        item_field = "oa_fan_out_item"
        fan_out_cfg.setdefault("collect_field", collect_field)
        fan_out_cfg.setdefault("target_field", sink)
        # items_field mode also requires item_field (where the engine places
        # each item in the inner state). The augment middleware reads the item
        # back out of this slot at runtime.
        fan_out_cfg.setdefault("item_field", item_field)
        # Ensure the inner state declares item_field (+ the inferred response
        # fields) whether or not the subgraph shipped its own state block --
        # State is strict, so the engine's write to item_field needs it declared.
        sub_state = cast("dict[str, Any]", sub_spec.setdefault("state", {}))
        sub_fields = cast("dict[str, Any]", sub_state.setdefault("fields", {}))
        for fname, fdef in inferred.items():
            sub_fields.setdefault(fname, fdef)
        sub_fields.setdefault(item_field, {"type": "dict", "default": {}})
        state_fields[sink] = {"type": "list", "reducer": "append", "default": []}
        # The items_field source (e.g. products) must be declared on the outer
        # state; 029 ships it only via initial_state, so declare it here.
        items_field = cast("str | None", fan_out_cfg.get("items_field"))
        if items_field is not None:
            state_fields.setdefault(items_field, {"type": "list<dict>", "default": []})
    if state_fields:
        state_block["fields"] = state_fields
        spec["state"] = state_block


def _build_augment_middlewares(
    case: Mapping[str, Any],
) -> tuple[
    dict[str, list[Any]],  # fan_out_instance_middleware: node_name -> [Middleware]
    dict[str, dict[str, list[Any]]],  # parallel_branches_branch_middleware: node -> branch -> [Middleware]
]:
    """Detect proposal-0040 augment directives in the case spec and
    synthesize the middlewares that drive them via the adapter's
    ``fan_out_instance_middleware`` / ``parallel_branches_branch_middleware``
    hooks.

    - Fan-out ``augment_metadata_from_field: {key: field_path}`` →
      one instance middleware that reads ``current_fan_out_index()``,
      indexes into the parent's ``items_field`` list captured at
      fixture-build time, and calls ``set_invocation_metadata(**entries)``
      where entries are pulled from the per-instance item via field_path.
    - Parallel-branches ``branches.<name>.augment_metadata: {key: value}``
      → per-branch middleware that calls
      ``set_invocation_metadata(**entries)`` once at branch entry.
    """
    fan_out_mw: dict[str, list[Any]] = {}
    branch_mw: dict[str, dict[str, list[Any]]] = {}

    for node_name, node_spec_any in cast("dict[str, Any]", case.get("nodes") or {}).items():
        if not isinstance(node_spec_any, dict):
            continue
        node_spec = cast("dict[str, Any]", node_spec_any)
        fan_out_cfg = cast("dict[str, Any] | None", node_spec.get("fan_out"))
        if fan_out_cfg is not None:
            augment_field_map = cast("dict[str, str] | None", fan_out_cfg.get("augment_metadata_from_field"))
            if augment_field_map:
                item_field = cast("str | None", fan_out_cfg.get("item_field"))
                if item_field is None:
                    raise ValueError(
                        f"fan-out node {node_name!r}: augment_metadata_from_field requires item_field"
                    )
                fan_out_mw[node_name] = [_make_augment_instance_middleware(augment_field_map, item_field)]
        pb_cfg = cast("dict[str, Any] | None", node_spec.get("parallel_branches"))
        if pb_cfg is not None:
            branches_cfg = cast("dict[str, dict[str, Any]]", pb_cfg.get("branches") or {})
            per_branch: dict[str, list[Any]] = {}
            for branch_name, branch_cfg in branches_cfg.items():
                augment_entries = cast("dict[str, Any] | None", branch_cfg.get("augment_metadata"))
                if augment_entries:
                    per_branch[branch_name] = [_make_augment_branch_middleware(augment_entries)]
            if per_branch:
                branch_mw[node_name] = per_branch
    return fan_out_mw, branch_mw


def _make_augment_instance_middleware(field_map: dict[str, str], item_field: str) -> Any:
    """Per-instance fan-out middleware that reads the instance's own item from
    the ``item_field`` slot of its subgraph state and calls
    ``set_invocation_metadata`` with the mapped entries."""

    # Reads runtime state, not a build-time list indexed by
    # current_fan_out_index(): instance middleware wraps the inner subgraph from
    # OUTSIDE, but the fan_out_index ContextVar is set deeper (inside the inner
    # node execution), so it is None here. The engine has already placed each
    # item in item_field by the time the chain runs, and set_invocation_metadata
    # lands before the inner spans open, so the instance's dispatch + inner spans
    # all carry the augmentation.
    class _AugmentInstanceMW:
        async def __call__(self, state: Any, next_: Any, /) -> Any:
            from openarmature.observability.metadata import (  # noqa: PLC0415
                set_invocation_metadata,
            )

            item = getattr(state, item_field, None)
            if isinstance(item, Mapping):
                item_map = cast("Mapping[str, Any]", item)
                entries = {key: item_map[field_path] for key, field_path in field_map.items()}
                set_invocation_metadata(**entries)
            return await next_(state)

    return _AugmentInstanceMW()


def _make_augment_branch_middleware(entries: dict[str, Any]) -> Any:
    """Per-branch middleware that calls ``set_invocation_metadata``
    once at branch entry. Captures ``entries`` at fixture-build
    time so the call inside the middleware doesn't need to read
    the case spec at runtime."""

    class _AugmentBranchMW:
        async def __call__(self, state: Any, next_: Any, /) -> Any:
            from openarmature.observability.metadata import (  # noqa: PLC0415
                set_invocation_metadata,
            )

            set_invocation_metadata(**entries)
            return await next_(state)

    return _AugmentBranchMW()


def _fixture_paths() -> list[Path]:
    return sorted(p for p in CONFORMANCE_DIR.glob("[0-9][0-9][0-9]-*.yaml") if p.stem in _LANGFUSE_FIXTURES)


def _fixture_id(path: Path) -> str:
    return path.stem


def _load(path: Path) -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(path.read_text()))


# ---------------------------------------------------------------------------
# Mock LLM transport
# ---------------------------------------------------------------------------


def _build_mock_llm_handler(responses: list[dict[str, Any]]) -> httpx.MockTransport:
    queue = list(responses)

    def _handler(_request: httpx.Request) -> httpx.Response:
        if not queue:
            raise AssertionError("mock_llm queue exhausted")
        spec_resp = queue.pop(0)
        body = cast("dict[str, Any]", spec_resp.get("body") or {})
        return httpx.Response(
            int(spec_resp.get("status", 200)),
            content=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(_handler)


# ---------------------------------------------------------------------------
# Mock prompt backend (fixture 024)
# ---------------------------------------------------------------------------


class _MockPromptBackend:
    """Returns canned Prompts for fixture 024.

    Two flavors per the fixture YAML's ``prompt_backend.type``:

    - ``mock_with_langfuse_reference``: attaches the supplied
      ``langfuse_prompt_reference`` sentinel under
      ``Prompt.observability_entities['langfuse_prompt']``. Verifies
      the Generation-linked-to-Prompt-entity case.
    - ``filesystem``: no Langfuse reference attached. Verifies the
      metadata-only case.
    """

    def __init__(self, prompts: dict[str, dict[str, Any]], *, with_langfuse_reference: bool) -> None:
        self._prompts: dict[str, Prompt] = {}
        now = datetime.now(UTC)
        for prompt_name, spec in prompts.items():
            observability_entities: dict[str, Any] | None = None
            if with_langfuse_reference and "langfuse_prompt_reference" in spec:
                observability_entities = {"langfuse_prompt": spec["langfuse_prompt_reference"]}
            self._prompts[prompt_name] = TextPrompt(
                name=spec["name"],
                version=spec["version"],
                label=spec["label"],
                template=spec["template"],
                template_hash=spec["template_hash"],
                fetched_at=now,
                observability_entities=observability_entities,
            )

    async def fetch(
        self, name: str, label: str = "production", *, cache_ttl_seconds: int | None = None
    ) -> Prompt:
        return self._prompts[name]


# ---------------------------------------------------------------------------
# Fixture runner
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=_fixture_id)
async def test_langfuse_fixture(fixture_path: Path) -> None:
    spec = _load(fixture_path)
    fixture_stem = fixture_path.stem
    if "cases" in spec:
        # Fold fixture-level ``subgraphs`` / ``inner_subgraphs`` into
        # each case so the per-case runner sees them locally. Fixture
        # 030 declares its branch subgraphs at fixture-level (alongside
        # ``cases:``); without this fold the per-case build can't
        # resolve ``branches.fraud_check.subgraph: fraud_check``.
        fixture_subgraphs = cast("dict[str, Any] | None", spec.get("subgraphs"))
        fixture_inner_subgraphs = cast("dict[str, Any] | None", spec.get("inner_subgraphs"))
        for case in cast("list[dict[str, Any]]", spec["cases"]):
            case_name = cast("str", case.get("name") or "<unnamed>")
            if (fixture_stem, case_name) in _DEFERRED_CASES:
                # Per-case deferral. Skipping inside the loop rather
                # than emitting a separate pytest.skip lets us keep the
                # surrounding cases running under the same parametrized
                # test id.
                continue
            if fixture_subgraphs is not None and "subgraphs" not in case:
                case["subgraphs"] = fixture_subgraphs
            if fixture_inner_subgraphs is not None and "inner_subgraphs" not in case:
                case["inner_subgraphs"] = fixture_inner_subgraphs
            try:
                await _run_case(case, fixture_stem=fixture_stem)
            except AssertionError as e:
                raise AssertionError(f"case {case_name!r}: {e}") from e
    else:
        await _run_case(spec, fixture_stem=fixture_stem)


def _has_topology_constructs(case: Mapping[str, Any]) -> bool:
    """Return True when the fixture uses subgraph / fan_out / parallel_branches
    constructs. Such fixtures need the full ``adapter.build_graph`` machinery
    rather than the simpler per-node hand-rolled path used for the
    LLM/prompt-only fixtures."""
    if "subgraph" in case or "subgraphs" in case:
        return True
    nodes_spec = cast("dict[str, Any]", case.get("nodes") or {})
    for node_spec in nodes_spec.values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        if "subgraph" in node_dict or "fan_out" in node_dict or "parallel_branches" in node_dict:
            return True
    return False


def _patch_unsupported_directives(spec: Mapping[str, Any]) -> None:
    """Replace inner-node test-seam directives the cross-capability
    adapter doesn't translate (``update_pure_from_state``) with a
    benign ``update_pure: {}`` no-op. The topology fixtures assert
    observation structure (parenting, trace ids, subgraph_name,
    correlation_id), not computed state values, so the swap is safe.
    Mirrors the OTel harness's helper of the same name."""

    def patch_nodes(graph_block: Mapping[str, Any] | None) -> None:
        if not graph_block:
            return
        nodes = cast("dict[str, Any]", graph_block.get("nodes") or {})
        for node_spec_any in nodes.values():
            if not isinstance(node_spec_any, dict):
                continue
            node_spec = cast("dict[str, Any]", node_spec_any)
            if "update_pure_from_state" in node_spec:
                node_spec.pop("update_pure_from_state")
                node_spec.setdefault("update_pure", {})

    patch_nodes(spec)
    if "subgraph" in spec:
        patch_nodes(cast("Mapping[str, Any]", spec["subgraph"]))
    for sub in cast("dict[str, Any]", spec.get("subgraphs") or {}).values():
        patch_nodes(cast("Mapping[str, Any]", sub))


def _compile_subgraphs(
    spec: Mapping[str, Any],
    *,
    provider: OpenAIProvider | None = None,
    prompt_manager: PromptManager | None = None,
    render_variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build any subgraphs declared by the fixture and return a
    name→compiled-graph registry the adapter consumes. Mirrors the
    OTel-side helper in ``test_observability.py``.

    When ``provider`` is supplied, inner subgraph nodes carrying the
    ``calls_llm:`` / ``renders_prompt:`` directives (fixtures 029 /
    030) are built via the langfuse-specific
    :func:`_build_node_body` rather than the cross-cap adapter — the
    adapter doesn't model LLM directives. Outer subgraphs without
    LLM directives still resolve through ``build_graph`` so the
    existing 031/032/033 wiring is unchanged.
    """
    subgraph_specs: dict[str, Any] = {}
    if "subgraph" in spec:
        single = cast("Mapping[str, Any]", spec["subgraph"])
        name = single.get("name") or "subgraph"
        subgraph_specs[name] = single
    if "subgraphs" in spec:
        for k, v in cast("dict[str, Any]", spec["subgraphs"]).items():
            subgraph_specs[k] = v
    compiled_subgraphs: dict[str, Any] = {}
    for name, sub_spec in subgraph_specs.items():
        if provider is not None and _has_llm_nodes(sub_spec):
            compiled_subgraphs[name] = _build_inner_subgraph_with_llm(
                sub_spec,
                provider=provider,
                prompt_manager=prompt_manager,
                render_variables=render_variables or {},
            )
        else:
            sub_built = build_graph(sub_spec, trace=[])
            compiled_subgraphs[name] = sub_built.builder.compile()
    return compiled_subgraphs


def _has_llm_nodes(spec: Mapping[str, Any]) -> bool:
    """True iff any node in the subgraph spec declares an LLM
    directive (``calls_llm`` / ``renders_prompt``) — those need the
    langfuse-specific node body builder rather than the cross-cap
    adapter."""
    nodes_spec = cast("dict[str, Any]", spec.get("nodes") or {})
    for node_spec in nodes_spec.values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        if "calls_llm" in node_dict or "renders_prompt" in node_dict:
            return True
    return False


def _infer_state_fields_from_nodes(nodes_spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build a minimal state-fields block from nodes' partial-update
    targets so an inner subgraph without an explicit ``state:`` block
    (fixture 029) still compiles. Walks ``stores_response_in``
    directives on ``calls_llm`` blocks; defaults each inferred field
    to ``string``."""
    fields: dict[str, dict[str, Any]] = {}
    for node_spec_any in nodes_spec.values():
        if not isinstance(node_spec_any, dict):
            continue
        node_spec = cast("dict[str, Any]", node_spec_any)
        calls_llm = cast("dict[str, Any] | None", node_spec.get("calls_llm"))
        if calls_llm is None:
            continue
        stores_in = cast("str | None", calls_llm.get("stores_response_in"))
        if stores_in is not None and stores_in not in fields:
            fields[stores_in] = {"type": "string", "default": ""}
    return fields


def _build_inner_subgraph_with_llm(
    spec: Mapping[str, Any],
    *,
    provider: OpenAIProvider,
    prompt_manager: PromptManager | None,
    render_variables: dict[str, Any],
) -> Any:
    """Compile an inner subgraph spec into a CompiledGraph using the
    langfuse-specific node body builder so ``calls_llm`` / ``renders_prompt``
    directives resolve correctly. Used by fixtures 029 / 030 whose
    branch / per-instance subgraphs each make an LLM call."""
    # Some inner-subgraph specs (fixture 029) omit a ``state:`` block.
    # Synthesize one from ``stores_response_in`` directives so the
    # partial update each node returns has a corresponding field on
    # the state class. Default field type is ``string`` with empty
    # default, matching the canonical fixture convention.
    state_block = cast("dict[str, Any] | None", spec.get("state"))
    if state_block is not None:
        state_fields = cast("dict[str, dict[str, Any]]", state_block["fields"])
    else:
        state_fields = _infer_state_fields_from_nodes(cast("dict[str, Any]", spec.get("nodes") or {}))
    state_cls = build_state_cls("InnerSubgraphState", state_fields)
    nodes_spec = cast("dict[str, Any]", spec["nodes"])
    entry = cast("str", spec["entry"])
    edges = cast("list[dict[str, str]]", spec["edges"])
    builder = GraphBuilder(state_cls)
    for node_name, node_spec in nodes_spec.items():
        node_body = _build_node_body(
            node_name=node_name,
            node_spec=cast("dict[str, Any]", node_spec),
            provider=provider,
            prompt_manager=prompt_manager,
            render_variables=render_variables,
        )
        builder.add_node(node_name, node_body)
    for edge in edges:
        target_raw = edge["to"]
        target = END if target_raw == "END" else target_raw
        builder.add_edge(edge["from"], target)
    builder.set_entry(entry)
    return builder.compile()


# Fixture 039 (nested-lineage augmentation) declares nested fan-out graphs the
# generic cross-cap adapter can't construct (a fan-out inside a subgraph wrapper
# / another fan-out, and a per-item sub-field as the inner fan-out's items
# source). Each case is hand-built here against the engine's GraphBuilder --
# mirroring the dedicated 044 builder on the OTel side -- then driven through the
# shared observer + assertion path. The expected langfuse_trace in the YAML
# stays the oracle.
_FIXTURE_039 = "039-nested-lineage-augmentation"


def _build_039_graph(
    case: Mapping[str, Any],
    *,
    provider: OpenAIProvider | None,
    prompt_manager: PromptManager | None,
) -> tuple[Any, Any]:
    """Dispatch a 039 case to its hand-built graph; return (graph, factory)."""
    name = cast("str", case.get("name"))
    if name == "fan_out_in_serial_subgraph_augmenter_propagates_to_wrapper_span":
        return _build_039_case3(case, provider=provider, prompt_manager=prompt_manager)
    raise NotImplementedError(f"039 case not yet wired: {name!r}")


def _build_039_case3(
    case: Mapping[str, Any],
    *,
    provider: OpenAIProvider | None,
    prompt_manager: PromptManager | None,
) -> tuple[Any, Any]:
    # Case 3: a serial subgraph wrapper (`wrap`) descends into `wrapped_fan_out`,
    # whose `pick` fan-out runs per-product; each instance augments note=<id>.
    # The wrapper span must carry the augmentation (last-writer) per 0045's
    # lineage-aware rule, the fan-out NODE must not.
    # The fan-out places each outer product into per_product's item_field slot;
    # the augment middleware reads <id> from it. per_product's declared state
    # ({picked}) lacks the slot, so inject it (mirrors _synthesize_fan_out_
    # aggregation on the generic 029 path).
    assert provider is not None, "039 cases declare mock_llm, so the provider must be set"
    per_product_spec = copy.deepcopy(cast("dict[str, Any]", case["inner_subgraphs"]["per_product"]))
    per_product_spec.setdefault("state", {}).setdefault("fields", {}).setdefault(
        "oa_fan_out_item", {"type": "dict", "default": {}}
    )
    per_product = _build_inner_subgraph_with_llm(
        per_product_spec,
        provider=provider,
        prompt_manager=prompt_manager,
        render_variables={},
    )
    wrap_state_cls = build_state_cls(
        "Wrapped039C3",
        {
            "picks": {"type": "list", "reducer": "append", "default": []},
            "products": {"type": "list<dict>", "default": []},
            "oa_fan_out_item": {"type": "dict", "default": {}},
        },
    )
    wrap_builder: GraphBuilder[Any] = GraphBuilder(wrap_state_cls)
    wrap_builder.set_entry("pick")
    wrap_builder.add_fan_out_node(
        "pick",
        subgraph=per_product,
        items_field="products",
        item_field="oa_fan_out_item",
        collect_field="picked",
        target_field="picks",
        instance_middleware=[_make_augment_instance_middleware({"note": "id"}, "oa_fan_out_item")],
    )
    wrap_builder.add_edge("pick", END)
    wrapped_fan_out = wrap_builder.compile()

    outer_state_cls = build_state_cls(
        "Outer039C3",
        {"result": {"type": "list", "default": []}, "products": {"type": "list<dict>", "default": []}},
    )
    outer_builder: GraphBuilder[Any] = GraphBuilder(outer_state_cls)
    outer_builder.set_entry("wrap")
    outer_builder.add_subgraph_node(
        "wrap",
        wrapped_fan_out,
        ExplicitMapping(inputs={"products": "products"}, outputs={"result": "picks"}),
    )
    outer_builder.add_edge("wrap", END)
    graph = outer_builder.compile()
    initial = cast("dict[str, Any]", case.get("initial_state") or {})
    return graph, (lambda: outer_state_cls(**initial))


# Proposal 0045 §3.4: a key set via set_invocation_metadata inside a fan-out
# instance / parallel-branches branch lands ONLY on the dispatch ancestors on
# the augmenter's call-stack path -- NOT on the shared fan-out/pb NODE, sibling
# instances, or (inside a dispatch) the Trace. The tree asserter is subset-based
# (extra keys tolerated), which can't catch a MUST-NOT violation, so 039
# additionally enforces that augmented keys absent from an observation's expected
# metadata are absent in the actual. Scoped to 039 via this ContextVar so the
# established subset semantics for the other fixtures are unchanged.
_AUGMENT_KEYS_UNDER_TEST: ContextVar[frozenset[str]] = ContextVar(
    "augment_keys_under_test", default=frozenset()
)


def _collect_augment_keys(case: Mapping[str, Any]) -> frozenset[str]:
    """Collect the metadata keys augment directives set, anywhere in the case's
    topology (fan-out / parallel-branches augment blocks at any nesting)."""
    keys: set[str] = set()
    directives = ("augment_metadata_from_field", "augment_metadata_from_outer_item", "augment_metadata")

    def _harvest(block: Any) -> None:
        if not isinstance(block, dict):
            return
        for directive in directives:
            mapping = cast("dict[str, Any]", block).get(directive)
            if isinstance(mapping, dict):
                keys.update(cast("dict[str, Any]", mapping).keys())

    def _walk(spec: Mapping[str, Any]) -> None:
        for node in cast("dict[str, Any]", spec.get("nodes") or {}).values():
            if not isinstance(node, dict):
                continue
            node_dict = cast("dict[str, Any]", node)
            _harvest(node_dict.get("fan_out"))
            pb = cast("dict[str, Any] | None", node_dict.get("parallel_branches"))
            for branch in cast("dict[str, Any]", (pb or {}).get("branches") or {}).values():
                _harvest(branch)
        for collection in ("subgraphs", "inner_subgraphs"):
            for sub in cast("dict[str, Any]", spec.get(collection) or {}).values():
                if isinstance(sub, dict):
                    _walk(cast("Mapping[str, Any]", sub))

    _walk(case)
    return frozenset(keys)


def _assert_augment_keys_not_leaked(
    label: str, actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> None:
    # Proposal 0045 §3.4 MUST-NOT: an augmented key absent from the expected
    # metadata (a shared fan-out/pb NODE, a sibling, or the Trace inside a
    # dispatch) MUST also be absent in the actual. Complements the subset matcher.
    for key in _AUGMENT_KEYS_UNDER_TEST.get():
        if key not in expected:
            assert key not in actual, (
                f"{label}: MUST NOT carry augmented key {key!r} (proposal 0045 §3.4); got {actual.get(key)!r}"
            )


def _resolve_detached_wrapper_names(case: Mapping[str, Any]) -> frozenset[str]:
    """Translate fixture-level ``detached_subgraphs`` (a list of SUBGRAPH
    IDENTITY names) into the set of WRAPPER NODE names the observer keys
    on. The fixture identifies detached subgraphs by their declaration name
    in ``subgraphs:`` (e.g., ``long_running_workflow``), but the
    LangfuseObserver matches by the wrapper node name in the parent graph
    that references the subgraph (e.g., ``dispatch``).
    """
    detached_identities = set(cast("list[str]", case.get("detached_subgraphs") or []))
    if not detached_identities:
        return frozenset()
    nodes_spec = cast("dict[str, Any]", case.get("nodes") or {})
    wrappers: set[str] = set()
    for wrapper_name, node_spec in nodes_spec.items():
        if not isinstance(node_spec, dict):
            continue
        sub_id = cast("dict[str, Any]", node_spec).get("subgraph")
        if isinstance(sub_id, str) and sub_id in detached_identities:
            wrappers.add(wrapper_name)
    return frozenset(wrappers)


async def _run_case(case: Mapping[str, Any], *, fixture_stem: str | None = None) -> None:
    # 039 additionally enforces proposal 0045's MUST-NOT scoping (an augmented
    # key absent from an observation's expected metadata must be absent in the
    # actual); other fixtures keep the established subset semantics.
    _AUGMENT_KEYS_UNDER_TEST.set(_collect_augment_keys(case) if fixture_stem == _FIXTURE_039 else frozenset())
    # ---- Mock LLM transport (if the graph has an LLM call)
    mock_responses = cast("list[dict[str, Any]] | None", case.get("mock_llm"))
    transport = _build_mock_llm_handler(mock_responses) if mock_responses else None
    provider: OpenAIProvider | None = None
    if transport is not None:
        provider = OpenAIProvider(
            base_url="http://mock-llm.test",
            model=_resolve_llm_model(case),
            api_key="test",
            transport=transport,
        )

    # ---- Prompt backend (fixture 024)
    prompt_manager: PromptManager | None = None
    prompt_backend_spec = cast("dict[str, Any] | None", case.get("prompt_backend"))
    if prompt_backend_spec is not None:
        backend_type = prompt_backend_spec.get("type")
        prompts_block = cast("dict[str, dict[str, Any]]", prompt_backend_spec.get("prompts") or {})
        backend = _MockPromptBackend(
            prompts_block,
            with_langfuse_reference=(backend_type == "mock_with_langfuse_reference"),
        )
        prompt_manager = PromptManager(backend)

    # ---- Graph build
    # Two paths: topology fixtures (031/032/033) need the full
    # ``adapter.build_graph`` machinery for subgraph / fan_out shapes;
    # LLM/prompt fixtures (022/023/024) use the simpler hand-rolled
    # per-node build that knows about ``calls_llm`` / ``renders_prompt``.
    if fixture_stem == _FIXTURE_039:
        # 039's nested fan-out graphs are hand-built (the generic adapter can't
        # construct them); see _build_039_graph.
        graph, initial_state_factory = _build_039_graph(
            case, provider=provider, prompt_manager=prompt_manager
        )
    elif _has_topology_constructs(case):
        # The topology fixtures (031/032/033) use inner-node test-seam
        # directives the cross-capability adapter doesn't translate
        # (``update_pure_from_state`` computes a value the assertions
        # don't inspect — they check span / observation structure, not
        # state values). Swap those for a benign ``update_pure: {}``
        # no-op so the graph is runnable, mirroring the OTel harness's
        # ``_patch_unsupported_directives``.
        _patch_unsupported_directives(case)
        # Per proposal 0040 fixture 029: rename ``inner_subgraph(s)`` →
        # ``subgraph(s)`` so the cross-cap adapter resolves the
        # references. Pure key normalization; semantics unchanged.
        if isinstance(case, dict):
            _normalize_fan_out_subgraph_keys(case)
            _synthesize_fan_out_aggregation(case)
        # Per proposal 0040 fixtures 029 / 030: synthesize the
        # augmentation middlewares that drive the per-instance /
        # per-branch ``set_invocation_metadata`` calls. Both flow into
        # ``build_graph`` via the adapter's standard middleware hooks;
        # the augmentation event then fires through the engine and
        # the LangfuseObserver handles it via
        # ``_handle_metadata_augmentation``.
        fan_out_instance_mw, branch_mw = _build_augment_middlewares(case)
        subgraphs = _compile_subgraphs(
            case,
            provider=provider,
            prompt_manager=prompt_manager,
            render_variables=cast("dict[str, Any]", case.get("render_variables") or {}),
        )
        built = build_graph(
            case,
            subgraphs=subgraphs,
            trace=[],
            fan_out_instance_middleware=fan_out_instance_mw or None,
            parallel_branches_branch_middleware=branch_mw or None,
        )
        graph = built.builder.compile()
        initial_state_factory = lambda: built.initial_state(case.get("initial_state", {}))  # noqa: E731
    else:
        state_fields = cast("dict[str, dict[str, Any]]", case["state"]["fields"])
        state_cls = build_state_cls("LangfuseFixtureState", state_fields)
        nodes_spec = cast("dict[str, Any]", case["nodes"])
        entry = cast("str", case["entry"])
        edges = cast("list[dict[str, str]]", case["edges"])
        render_variables = cast("dict[str, Any]", case.get("render_variables") or {})

        builder = GraphBuilder(state_cls)
        for node_name, node_spec in nodes_spec.items():
            node_body = _build_node_body(
                node_name=node_name,
                node_spec=cast("dict[str, Any]", node_spec),
                provider=provider,
                prompt_manager=prompt_manager,
                render_variables=render_variables,
            )
            builder.add_node(node_name, node_body)
        for edge in edges:
            target_raw = edge["to"]
            target = END if target_raw == "END" else target_raw
            builder.add_edge(edge["from"], target)
        builder.set_entry(entry)
        # Optional checkpointer wiring — fixture 037 case 5 needs an
        # in-memory checkpointer so the first invoke's pre-failure save
        # carries over to the resumed invoke.  Only the literal value
        # ``"in_memory"`` is recognized; other backends would need
        # additional registration shimmed here.
        checkpointer_spec = cast("str | None", case.get("checkpointer"))
        if checkpointer_spec == "in_memory":
            from openarmature.checkpoint import InMemoryCheckpointer  # noqa: PLC0415

            builder.with_checkpointer(InMemoryCheckpointer())
        elif checkpointer_spec is not None:
            raise NotImplementedError(
                f"langfuse harness only supports checkpointer: in_memory; got {checkpointer_spec!r}"
            )
        graph = builder.compile()
        # ``initial_state`` overrides on the case populate caller-
        # supplied fields; remaining fields fall back to the State
        # class's declared defaults.  Proposal 0043's case 2 relies on
        # this — it ships ``initial_state: {msg: "start"}`` to assert
        # the raw-state ``trace.input`` carries the caller-supplied
        # value rather than the default.
        case_initial_state = cast("dict[str, Any]", case.get("initial_state") or {})
        initial_state_factory = lambda: graph.state_cls(**case_initial_state)  # noqa: E731

    # ---- Observer
    observer_cfg = cast("dict[str, Any]", case.get("langfuse_observer") or {})
    observer_kwargs: dict[str, Any] = {}
    if "disable_provider_payload" in observer_cfg:
        observer_kwargs["disable_provider_payload"] = bool(observer_cfg["disable_provider_payload"])
    if "disable_llm_spans" in observer_cfg:
        observer_kwargs["disable_llm_spans"] = bool(observer_cfg["disable_llm_spans"])
    if "payload_byte_cap" in observer_cfg:
        observer_kwargs["payload_byte_cap"] = int(observer_cfg["payload_byte_cap"])
    # Proposal 0043 (§8.4.1 trace.input/output sourcing).
    if "disable_state_payload" in observer_cfg:
        observer_kwargs["disable_state_payload"] = bool(observer_cfg["disable_state_payload"])
    if "trace_input_from_state" in observer_cfg:
        observer_kwargs["trace_input_from_state"] = _resolve_trace_io_hook(
            cast("str", observer_cfg["trace_input_from_state"])
        )
    if "trace_output_from_state" in observer_cfg:
        observer_kwargs["trace_output_from_state"] = _resolve_trace_io_hook(
            cast("str", observer_cfg["trace_output_from_state"])
        )
    detached_subgraphs = _resolve_detached_wrapper_names(case)
    if detached_subgraphs:
        observer_kwargs["detached_subgraphs"] = detached_subgraphs
    detached_fan_outs = frozenset(cast("list[str]", case.get("detached_fan_outs") or []))
    if detached_fan_outs:
        observer_kwargs["detached_fan_outs"] = detached_fan_outs
    client = InMemoryLangfuseClient()
    observer = LangfuseObserver(client=client, **observer_kwargs)
    graph.attach_observer(observer)

    # ---- Run
    correlation_id = cast("str | None", case.get("caller_correlation_id"))
    invoke_kwargs: dict[str, Any] = {}
    if correlation_id is not None:
        invoke_kwargs["correlation_id"] = correlation_id
    caller_metadata = cast("dict[str, Any] | None", case.get("caller_metadata"))
    if caller_metadata is not None:
        invoke_kwargs["metadata"] = caller_metadata
    # Fixtures 035/036: caller-supplied invocation_id drives the trace.id
    # derivation (UUID hex dashes-stripped / sha256-first-16 for a non-UUID).
    caller_invocation_id = cast("str | None", case.get("caller_invocation_id"))
    if caller_invocation_id is not None:
        invoke_kwargs["invocation_id"] = caller_invocation_id

    # Resume cases run a two-phase flow (first invoke catches expected
    # error → resume invoke completes), then assert against both traces
    # separately.  Branch out here so the linear ``await graph.invoke``
    # below stays focused on the common case.
    if "resume" in case:
        await _run_resume_case(
            case=case,
            graph=graph,
            initial_state_factory=initial_state_factory,
            client=client,
            invoke_kwargs=invoke_kwargs,
        )
        if provider is not None:
            await provider.aclose()
        return

    await graph.invoke(initial_state_factory(), **invoke_kwargs)
    await graph.drain()
    if provider is not None:
        await provider.aclose()

    # ---- Assert
    # Single-trace fixtures use ``langfuse_trace:``; detached / multi-trace
    # fixtures use ``langfuse_traces:`` (a list). Branch on which the
    # fixture supplies.
    expected = cast("dict[str, Any]", case["expected"])
    expected_invariants = cast("dict[str, Any]", expected.get("invariants") or {})
    if "langfuse_traces" in expected:
        _assert_multi_traces(client, expected, expected_invariants)
    else:
        expected_trace = cast("dict[str, Any]", expected["langfuse_trace"])
        assert len(client.traces) == 1, f"expected exactly one Trace, got {len(client.traces)}"
        trace = next(iter(client.traces.values()))
        _assert_trace(trace, expected_trace, expected_invariants=expected_invariants)


async def _run_resume_case(
    *,
    case: Mapping[str, Any],
    graph: Any,
    initial_state_factory: Callable[[], Any],
    client: InMemoryLangfuseClient,
    invoke_kwargs: dict[str, Any],
) -> None:
    """Two-phase test flow for fixture 037 case 5.

    Step 1 — first invoke catches the expected NodeException at the
    designated node; the captured Langfuse Trace's input/output match
    ``first_run_expected.langfuse_trace``.  We snapshot the first trace's
    headline fields immediately so the ``first_trace_unchanged`` invariant
    can verify the resumed invoke leaves them untouched.

    Step 2 — resume invoke runs the same graph with
    ``resume_invocation=first_invocation_id``, completes successfully, and
    the resumed Trace's input/output match ``resume.expected.langfuse_trace``.

    Step 3 — invariants compare the two traces (distinct trace ids,
    shared correlation_id, the snapshotted first trace's fields unchanged).
    """
    from openarmature.graph.errors import RuntimeGraphError  # noqa: PLC0415

    # ---- Step 1: first invoke catches expected error
    first_run_expected_error = cast("dict[str, Any]", case.get("first_run_expected_error") or {})
    expected_category = cast("str", first_run_expected_error.get("category", "node_exception"))
    expected_raised_from = cast("str | None", first_run_expected_error.get("raised_from"))

    # Catch the common ``RuntimeGraphError`` base so the harness handles
    # any spec §4 category (node_exception / reducer_error /
    # state_validation_error / edge_exception / routing_error).  The
    # "raised from" node attribute differs per category — check
    # ``node_name`` on NodeException, ``producing_node`` on
    # ReducerError, ``source_node`` on EdgeException / RoutingError —
    # via a small attribute walk so we don't hardcode per-category
    # accessor knowledge here.
    try:
        await graph.invoke(initial_state_factory(), **invoke_kwargs)
    except RuntimeGraphError as exc:
        assert exc.category == expected_category, (
            f"first run error category: expected {expected_category!r}, got {exc.category!r}"
        )
        if expected_raised_from is not None:
            actual_raised_from = (
                getattr(exc, "node_name", None)
                or getattr(exc, "producing_node", None)
                or getattr(exc, "source_node", None)
            )
            assert actual_raised_from == expected_raised_from, (
                f"first run error raised_from: expected {expected_raised_from!r}, got {actual_raised_from!r}"
            )
    else:
        raise AssertionError(
            f"first run expected to raise RuntimeGraphError with category={expected_category!r}; "
            f"completed without error"
        )
    await graph.drain()

    assert len(client.traces) == 1, (
        f"first run should produce exactly one Langfuse Trace; got {len(client.traces)}"
    )
    first_invocation_id, first_trace = next(iter(client.traces.items()))

    # Snapshot the first trace's headline fields before the resume runs
    # so the ``first_trace_unchanged`` invariant can compare against the
    # state captured here.  ``client.traces`` holds the live recorder
    # objects; ``copy.deepcopy`` protects against in-place writes.
    first_trace_snapshot = {
        "input": copy.deepcopy(first_trace.input),
        "output": copy.deepcopy(first_trace.output),
    }

    first_run_expected = cast("dict[str, Any]", case["first_run_expected"])
    first_expected_trace = cast("dict[str, Any]", first_run_expected["langfuse_trace"])
    _assert_trace(first_trace, first_expected_trace, expected_invariants={})

    # ---- Step 2: resume invoke
    resume_block = cast("dict[str, Any]", case["resume"])
    # Drop ``correlation_id`` from invoke_kwargs on resume — the engine
    # restores it from the saved record per §3.1.
    resume_invoke_kwargs = {k: v for k, v in invoke_kwargs.items() if k != "correlation_id"}
    await graph.invoke(
        initial_state_factory(),
        resume_invocation=first_invocation_id,
        **resume_invoke_kwargs,
    )
    await graph.drain()

    # Python dicts are insertion-ordered (PEP 468; guaranteed since
    # 3.7).  The first invoke added one trace; the resume added another.
    # Reading by position is more deterministic than scanning by
    # not-equal — if a future engine change adds synthetic traces, the
    # scan would silently pick the wrong key, but the position-based
    # read fails the length assertion below explicitly.
    trace_ids = list(client.traces.keys())
    assert len(trace_ids) == 2, (
        f"after resume there should be exactly two Langfuse Traces; got {len(trace_ids)}"
    )
    assert trace_ids[0] == first_invocation_id, (
        f"first trace id changed during resume: was {first_invocation_id!r}, now {trace_ids[0]!r}"
    )
    resumed_invocation_id = trace_ids[1]
    resumed_trace = client.traces[resumed_invocation_id]

    resume_expected = cast("dict[str, Any]", resume_block["expected"])
    resume_expected_trace = cast("dict[str, Any]", resume_expected["langfuse_trace"])
    _assert_trace(resumed_trace, resume_expected_trace, expected_invariants={})

    # ---- Step 3: invariants
    if resume_expected.get("first_trace_unchanged"):
        assert first_trace.input == first_trace_snapshot["input"], (
            f"first_trace_unchanged failed: input was {first_trace_snapshot['input']!r}, "
            f"now {first_trace.input!r}"
        )
        assert first_trace.output == first_trace_snapshot["output"], (
            f"first_trace_unchanged failed: output was {first_trace_snapshot['output']!r}, "
            f"now {first_trace.output!r}"
        )

    invariants = cast("dict[str, Any]", case.get("invariants") or {})
    if invariants.get("distinct_trace_ids"):
        assert first_invocation_id != resumed_invocation_id, (
            f"distinct_trace_ids failed: both traces have id {first_invocation_id!r}"
        )
    if invariants.get("correlation_id_consistent_across_traces"):
        first_corr = first_trace.metadata.get("correlation_id")
        resumed_corr = resumed_trace.metadata.get("correlation_id")
        assert first_corr == resumed_corr, (
            f"correlation_id_consistent_across_traces failed: first={first_corr!r}, resumed={resumed_corr!r}"
        )
    # ``hooks_refire_on_resumed_trace`` is implicit — verified by the
    # ``_assert_trace`` call on the resumed trace above, which checks the
    # hook-derived input/output match the resumed invocation's state.


def _resolve_llm_model(case: Mapping[str, Any]) -> str:
    # Single LLM call per fixture today; pick up the per-call model if
    # supplied (fixture 023 explicitly sets `model: "test-model"` on
    # the calls_llm block).
    nodes_spec = cast("dict[str, Any]", case["nodes"])
    for node_spec in nodes_spec.values():
        if not isinstance(node_spec, dict):
            continue
        node_dict = cast("dict[str, Any]", node_spec)
        calls_llm = node_dict.get("calls_llm")
        if isinstance(calls_llm, dict):
            return cast("str", cast("dict[str, Any]", calls_llm).get("model", "test-model"))
    return "test-model"


def _build_node_body(
    *,
    node_name: str,
    node_spec: dict[str, Any],
    provider: OpenAIProvider | None,
    prompt_manager: PromptManager | None,
    render_variables: dict[str, Any],
) -> Any:
    # Three node shapes in fixtures 022-024:
    #   - `update: {...}` — set fields on state directly (022).
    #   - `calls_llm: {...}` — invoke provider.complete (023).
    #   - `renders_prompt: <name>` + optional `calls_llm` — render the
    #     named prompt, then call the LLM under `with_active_prompt`
    #     so the Generation's prompt-linkage metadata + entity link
    #     populate per §8.4.4 (024).
    # Per proposal 0040 fixture 034 the ``augment_metadata`` directive
    # MAY wrap any of the above shapes: at body entry, the harness
    # calls ``set_invocation_metadata(**augment)``. Open spans
    # outermost-serial (the invocation span / the calling node span)
    # MUST then carry the augmented keys in place.
    augment_spec = cast("dict[str, Any] | None", node_spec.get("augment_metadata"))

    def _maybe_augment() -> None:
        if augment_spec is not None:
            from openarmature.observability.metadata import (  # noqa: PLC0415
                set_invocation_metadata,
            )

            set_invocation_metadata(**augment_spec)

    update_spec = cast("dict[str, Any] | None", node_spec.get("update"))
    if update_spec is not None:

        async def _node(_s: Any) -> dict[str, Any]:
            _maybe_augment()
            return dict(update_spec)

        return _node

    # ``update_pure: {...}`` is the spec's literal-value update directive
    # (paralleling ``update_pure`` in tests/conformance/adapter.py:727).
    # Treated identically to ``update`` here — the langfuse harness only
    # needs the literal-value form to drive proposal 0043's simple
    # decision-tree cases.
    update_pure_spec = cast("dict[str, Any] | None", node_spec.get("update_pure"))
    if update_pure_spec is not None:

        async def _node_pure(_s: Any) -> dict[str, Any]:
            _maybe_augment()
            return dict(update_pure_spec)

        return _node_pure

    # ``flaky: {fail_first_invocation_only: true, on_success: {...}}`` —
    # the compact resume-fixture flaky shape (paralleling the equivalent
    # form in tests/conformance/adapter.py:_make_flaky_fn).  The node
    # raises on its first call (a fresh ``RuntimeError`` the engine wraps
    # as ``NodeException``) and returns ``on_success`` on subsequent
    # calls.  Used by fixture 037 case 5: the first invoke aborts at
    # this node; the resumed invoke calls the same node body — the
    # closure-scoped ``has_failed`` survives the resume because the
    # graph (and the closure) lives for the harness's full case run,
    # so the second call returns success.
    flaky_spec = cast("dict[str, Any] | None", node_spec.get("flaky"))
    if flaky_spec is not None:
        if not flaky_spec.get("fail_first_invocation_only"):
            raise NotImplementedError(
                f"langfuse harness only supports the fail_first_invocation_only flaky shape; got {flaky_spec}"
            )
        on_success = dict(cast("dict[str, Any]", flaky_spec.get("on_success") or {}))
        has_failed = [False]

        async def _node_flaky(_s: Any) -> dict[str, Any]:
            _maybe_augment()
            if not has_failed[0]:
                has_failed[0] = True
                raise RuntimeError(f"flaky({node_name}) first-invocation failure")
            return dict(on_success)

        return _node_flaky

    calls_llm_spec = cast("dict[str, Any] | None", node_spec.get("calls_llm"))
    renders_prompt_name = cast("str | None", node_spec.get("renders_prompt"))

    async def _llm_node(_s: Any) -> dict[str, Any]:
        assert provider is not None, f"node {node_name!r} has calls_llm but no mock_llm responses"
        _maybe_augment()
        messages_spec = cast(
            "list[dict[str, Any]] | None",
            (calls_llm_spec or {}).get("messages"),
        )
        config_spec = cast("dict[str, Any] | None", (calls_llm_spec or {}).get("config"))
        stores_in = cast("str", (calls_llm_spec or {}).get("stores_response_in", "msg"))
        if renders_prompt_name is not None:
            assert prompt_manager is not None, "renders_prompt requires a prompt_backend block"
            prompt = await prompt_manager.fetch(renders_prompt_name)
            rendered = prompt_manager.render(prompt, render_variables)
            llm_messages = list(rendered.messages)
            with with_active_prompt(rendered):
                response = await provider.complete(
                    cast("Sequence[Any]", llm_messages),
                    config=_runtime_config_from_spec(config_spec),
                )
        else:
            llm_messages = _materialize_messages(messages_spec or [])
            response = await provider.complete(
                cast("Sequence[Any]", llm_messages),
                config=_runtime_config_from_spec(config_spec),
            )
        return {stores_in: response.message.content or ""}

    return _llm_node


def _materialize_messages(raw: list[dict[str, Any]]) -> list[Any]:
    from openarmature.llm.messages import AssistantMessage, SystemMessage, ToolMessage, UserMessage

    out: list[Any] = []
    for entry in raw:
        role = entry.get("role")
        content: Any
        if "content_repeat" in entry:
            repeat = cast("dict[str, Any]", entry["content_repeat"])
            content = cast("str", repeat["char"]) * int(repeat["bytes"])
        else:
            content = entry.get("content")
        if role == "system":
            out.append(SystemMessage(content=cast("str", content)))
        elif role == "user":
            out.append(UserMessage(content=cast("str", content)))
        elif role == "assistant":
            out.append(AssistantMessage(content=cast("str", content)))
        elif role == "tool":
            out.append(
                ToolMessage(content=cast("str", content), tool_call_id=cast("str", entry["tool_call_id"]))
            )
        else:
            raise AssertionError(f"unknown message role: {role!r}")
    return out


def _runtime_config_from_spec(config_spec: dict[str, Any] | None) -> RuntimeConfig | None:
    if not config_spec:
        return None
    declared = {
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "frequency_penalty",
        "presence_penalty",
        "stop_sequences",
    }
    kwargs = {k: v for k, v in config_spec.items() if k in declared}
    extras = cast("dict[str, Any]", config_spec.get("extras") or {})
    kwargs.update(extras)
    return RuntimeConfig(**kwargs)


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


# Per-trace invariants — invariants ``_assert_trace`` knows how to
# check on a single Trace. The multi-trace runner filters
# ``expected_invariants`` to this set when delegating per-Trace
# assertions; the rest (``distinct_trace_ids``,
# ``correlation_id_consistent_across_traces``, etc.) stay in
# ``_assert_multi_traces`` as cross-Trace checks.
_PER_TRACE_INVARIANTS = frozenset({"trace_id_equals_invocation_id", "correlation_id_consistency"})


def _assert_multi_traces(
    client: InMemoryLangfuseClient,
    expected: dict[str, Any],
    expected_invariants: dict[str, Any],
) -> None:
    """Detached-trace fixtures (033) span multiple Traces. The expected
    block's ``langfuse_traces:`` list names the parent Trace explicitly;
    the additional detached Traces are either fully enumerated (subgraph
    case) or counted via ``detached_trace_count`` (fan-out case where
    only the parent's shape is asserted explicitly and the per-instance
    Traces share an identical shape).
    """
    expected_traces = cast("list[dict[str, Any]]", expected.get("langfuse_traces") or [])
    detached_trace_count = cast("int | None", expected.get("detached_trace_count"))
    # If the fixture enumerates all Traces explicitly, the actual count
    # MUST match. Otherwise, ``detached_trace_count`` indicates how many
    # additional traces beyond the enumerated parent the fixture expects.
    expected_total = (
        len(expected_traces) + detached_trace_count
        if detached_trace_count is not None
        else len(expected_traces)
    )
    assert len(client.traces) == expected_total, (
        f"expected {expected_total} Traces, got {len(client.traces)}: "
        f"{[t.name for t in client.traces.values()]}"
    )

    # Match each enumerated expected Trace to an actual Trace by name
    # compatibility (literal match, or wildcard ``<...>`` placeholder)
    # then by root-observation count. Tracks consumed traces so two
    # expected entries can't bind to the same actual one.
    consumed: set[str] = set()
    for exp in expected_traces:
        exp_name = cast("str", exp.get("name") or "")
        is_wildcard = exp_name.startswith("<") and exp_name.endswith(">")
        expected_obs_count = len(cast("list[Any]", exp.get("observations") or []))
        candidates = [
            t for t in client.traces.values() if t.id not in consumed and (is_wildcard or t.name == exp_name)
        ]
        # Prefer candidates whose root-observation count matches the
        # expected structure; the fan-out case has multiple traces of
        # the same name where the parent is the only one with a
        # populated observation tree.
        matching = [t for t in candidates if len(t.children_of(None)) == expected_obs_count]
        assert matching or candidates, (
            f"no Trace matches expected name={exp_name!r} (consumed={sorted(consumed)})"
        )
        trace = matching[0] if matching else candidates[0]
        consumed.add(trace.id)
        per_trace_invariants = {k: v for k, v in expected_invariants.items() if k in _PER_TRACE_INVARIANTS}
        _assert_trace(trace, exp, expected_invariants=per_trace_invariants)

    # Invariants that span multiple Traces.
    if expected_invariants.get("distinct_trace_ids"):
        trace_ids = {t.id for t in client.traces.values()}
        assert len(trace_ids) == len(client.traces), f"trace ids not all distinct: {sorted(trace_ids)}"
    if expected_invariants.get("all_instance_trace_ids_distinct"):
        trace_ids = {t.id for t in client.traces.values()}
        assert len(trace_ids) == len(client.traces), (
            f"instance trace ids not all distinct: {sorted(trace_ids)}"
        )
    expected_child_count = cast(
        "int | None", expected_invariants.get("dispatch_detached_child_trace_id_count")
    )
    if expected_child_count is not None:
        # Find the parent-trace observation whose metadata carries
        # detached_child_trace_ids; assert its length.
        found = False
        for trace in client.traces.values():
            for obs in trace.observations:
                child_ids_raw = obs.metadata.get("detached_child_trace_ids")
                if isinstance(child_ids_raw, list):
                    child_ids = cast("list[Any]", child_ids_raw)
                    assert len(child_ids) == expected_child_count, (
                        f"observation {obs.name!r} detached_child_trace_ids length: "
                        f"expected {expected_child_count}, got {len(child_ids)}"
                    )
                    found = True
        assert found, "no observation carried metadata.detached_child_trace_ids"
    if expected_invariants.get("correlation_id_consistent_across_traces"):
        correlation_ids = {
            cast("str | None", t.metadata.get("correlation_id")) for t in client.traces.values()
        }
        correlation_ids.discard(None)
        if len(correlation_ids) > 1:
            sorted_ids = sorted(c for c in correlation_ids if c is not None)
            raise AssertionError(f"correlation_id not consistent across Traces: {sorted_ids}")
    if expected_invariants.get("no_instance_spans_in_parent_trace"):
        # The parent Trace is the one named in expected_traces (singular).
        if len(expected_traces) == 1:
            parent_name = cast("str", expected_traces[0]["name"])
            parents = [t for t in client.traces.values() if t.name == parent_name]
            # If multiple Traces share the parent name (fan-out case where
            # detached per-instance Traces inherit the fan-out node name),
            # the actual parent is the one with a non-empty observation
            # tree. Per-instance Traces have their own subtree under their
            # own dispatch observation; their leaf shapes are different
            # from the parent's flat fan-out node observation.
            for t in parents:
                # Parent observations must be limited to the fan-out
                # dispatch (no leaked per-instance inner-node names).
                root_obs = t.children_of(None)
                for obs in root_obs:
                    if obs.name != parent_name:
                        # If we got here, an inner-node observation
                        # leaked into the parent Trace.
                        raise AssertionError(
                            f"unexpected observation {obs.name!r} in parent Trace {t.id!r}; "
                            f"parent should only contain the fan-out dispatch observation"
                        )


def _assert_trace(
    trace: LangfuseTrace,
    expected: dict[str, Any],
    *,
    expected_invariants: dict[str, Any],
) -> None:
    expected_id = expected.get("id")
    if expected_id is not None and not _is_placeholder(expected_id):
        # Fixtures 035/036: a LITERAL trace.id is the DERIVED Langfuse id; the
        # in-memory recorder keys by the raw invocation_id, so bridge via the
        # impl's langfuse_trace_id (the derivation the real SDK adapter uses).
        from openarmature.observability.langfuse import langfuse_trace_id  # noqa: PLC0415

        derived = langfuse_trace_id(trace.id)
        assert derived == expected_id, (
            f"trace.id: derived {derived!r} (from raw {trace.id!r}) != {expected_id!r}"
        )
    else:
        _assert_string_or_placeholder("trace.id", trace.id, expected_id)
    if "name" in expected:
        _assert_string_or_placeholder("trace.name", trace.name, expected.get("name"))
    expected_metadata = dict(cast("dict[str, Any]", expected.get("metadata") or {}))
    # Fixture 036 asserts the raw invocation_id as metadata.invocation_id. The
    # real SDK derives trace.id and preserves the raw in metadata; the in-memory
    # recorder keeps the raw AS trace.id, so recover it from there.
    if "invocation_id" in expected_metadata and "invocation_id" not in trace.metadata:
        expected_invocation_id = expected_metadata.pop("invocation_id")
        assert trace.id == expected_invocation_id, (
            f"trace.metadata.invocation_id: raw trace.id {trace.id!r} != {expected_invocation_id!r}"
        )
    _assert_metadata_subset("trace.metadata", trace.metadata, expected_metadata)
    _assert_augment_keys_not_leaked("trace.metadata", trace.metadata, expected_metadata)
    # Proposal 0043 (§8.4.1 trace.input/output sourcing).  Fixtures that
    # opt in supply these as YAML maps; older fixtures leave them absent.
    if "input" in expected:
        expected_input = expected["input"]
        assert trace.input == expected_input, (
            f"trace.input mismatch: expected {expected_input!r}, got {trace.input!r}"
        )
    if "output" in expected:
        expected_output = expected["output"]
        assert trace.output == expected_output, (
            f"trace.output mismatch: expected {expected_output!r}, got {trace.output!r}"
        )
    # ``observations:`` is asserted only when the fixture supplies it.
    # Older fixtures that omit the block implicitly say "I'm asserting
    # trace-level fields only; don't care about the observation tree"
    # (proposal 0043's fixture 037 is the first to use this shape — it
    # focuses purely on trace.input/output).
    if "observations" in expected:
        expected_observations = cast("list[dict[str, Any]]", expected["observations"])
        root_observations = trace.children_of(None)
        _assert_observation_tree(trace, root_observations, expected_observations)

    # Invariants: cross-cutting checks that hold across the full Trace.
    if expected_invariants.get("trace_id_equals_invocation_id"):
        # No direct accessor for the invocation_id from outside the
        # observer; the §8.4.1 contract is that trace.id == invocation_id,
        # so the invariant degenerates to "trace.id matches the UUIDv4
        # pattern" — already asserted above via `<uuid>` placeholder.
        pass
    if expected_invariants.get("correlation_id_consistency"):
        trace_correlation = cast("str | None", trace.metadata.get("correlation_id"))
        if trace_correlation is not None:
            for obs in trace.observations:
                obs_correlation = obs.metadata.get("correlation_id")
                assert obs_correlation == trace_correlation, (
                    f"correlation_id mismatch: trace={trace_correlation!r}, "
                    f"observation {obs.name!r}={obs_correlation!r}"
                )


def _assert_observation_tree(
    trace: LangfuseTrace,
    actual_children: list[LangfuseObservation],
    expected_children: list[dict[str, Any]],
) -> None:
    assert len(actual_children) == len(expected_children), (
        f"observation children count mismatch: expected {len(expected_children)}, got {len(actual_children)}"
    )
    for actual, expected in zip(actual_children, expected_children, strict=True):
        _assert_observation(trace, actual, expected)


def _assert_observation(
    trace: LangfuseTrace,
    actual: LangfuseObservation,
    expected: dict[str, Any],
) -> None:
    if "type" in expected:
        assert actual.type == expected["type"], (
            f"observation {actual.name!r} type: expected {expected['type']!r}, got {actual.type!r}"
        )
    if "name" in expected:
        _assert_string_or_placeholder(f"observation[{actual.name}].name", actual.name, expected["name"])
    if "level" in expected:
        assert actual.level == expected["level"], (
            f"observation {actual.name!r} level: expected {expected['level']!r}, got {actual.level!r}"
        )
    if "model" in expected:
        assert actual.model == expected["model"], (
            f"observation {actual.name!r} model: expected {expected['model']!r}, got {actual.model!r}"
        )
    if "modelParameters" in expected:
        expected_params = cast("dict[str, Any]", expected["modelParameters"])
        for key, value in expected_params.items():
            assert actual.model_parameters.get(key) == value, (
                f"observation {actual.name!r} modelParameters.{key}: "
                f"expected {value!r}, got {actual.model_parameters.get(key)!r}"
            )
    if "usage" in expected:
        expected_usage = cast("dict[str, Any]", expected["usage"])
        assert actual.usage is not None, f"observation {actual.name!r} usage absent"
        for key, value in expected_usage.items():
            actual_value = getattr(actual.usage, key, None)
            assert actual_value == value, (
                f"observation {actual.name!r} usage.{key}: expected {value!r}, got {actual_value!r}"
            )
    if "input_parses_as_messages" in expected:
        expected_messages = cast("list[dict[str, Any]]", expected["input_parses_as_messages"])
        # input is the native message-list shape OR a JSON string of it;
        # either way, parse-to-shape MUST succeed and match.
        parsed = _parse_messages(actual.input)
        # The actual messages are produced by the LLM provider's
        # _serialize_messages_for_payload which carries the full §3 shape
        # (including ``content`` as either str or block list). Loose
        # subset compare on role+content for fixture parity.
        for expected_msg, parsed_msg in zip(expected_messages, parsed, strict=False):
            assert parsed_msg.get("role") == expected_msg["role"], (
                f"observation {actual.name!r} input message role mismatch: "
                f"expected {expected_msg['role']!r}, got {parsed_msg.get('role')!r}"
            )
            assert parsed_msg.get("content") == expected_msg["content"], (
                f"observation {actual.name!r} input message content mismatch: "
                f"expected {expected_msg['content']!r}, got {parsed_msg.get('content')!r}"
            )
    if expected.get("input_is_raw_string_with_marker") is True:
        assert isinstance(actual.input, str), (
            f"observation {actual.name!r} input expected raw string, got {type(actual.input).__name__}"
        )
        assert "[truncated," in actual.input, (
            f"observation {actual.name!r} input missing truncation marker: {actual.input!r}"
        )
    if "output" in expected:
        assert actual.output == expected["output"], (
            f"observation {actual.name!r} output: expected {expected['output']!r}, got {actual.output!r}"
        )
    if "prompt_entity_link" in expected:
        assert actual.prompt_entity_link == expected["prompt_entity_link"], (
            f"observation {actual.name!r} prompt_entity_link: "
            f"expected {expected['prompt_entity_link']!r}, got {actual.prompt_entity_link!r}"
        )
    if expected.get("prompt_entity_link_absent") is True:
        assert actual.prompt_entity_link is None, (
            f"observation {actual.name!r} prompt_entity_link expected absent, "
            f"got {actual.prompt_entity_link!r}"
        )
    expected_metadata = cast("dict[str, Any]", expected.get("metadata") or {})
    _assert_metadata_subset(f"observation[{actual.name}].metadata", actual.metadata, expected_metadata)
    _assert_augment_keys_not_leaked(
        f"observation[{actual.name}].metadata", actual.metadata, expected_metadata
    )

    expected_children = cast("list[dict[str, Any]]", expected.get("children") or [])
    actual_children = trace.children_of(actual.id)
    _assert_observation_tree(trace, actual_children, expected_children)


def _parse_messages(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return cast("list[dict[str, Any]]", value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"input attribute did not parse as JSON: {value!r}") from exc
        if isinstance(decoded, list):
            return cast("list[dict[str, Any]]", decoded)
    raise AssertionError(f"input attribute did not parse as a message list: {value!r}")


def _is_placeholder(value: Any) -> bool:
    """``<anything>`` literals in fixtures are wildcards: they assert
    shape (non-empty string) without binding a specific value. Cross-
    occurrence consistency lives in the ``invariants:`` block."""
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


_METADATA_MATCHER_SUBKEYS = frozenset({"harness_parameterized", "non_empty_string"})


def _assert_metadata_matcher_subkeys(label: str, actual: Any, spec: dict[str, Any]) -> None:
    """Fixture 059 attribution matcher sub-keys: ``non_empty_string`` (the value
    is a non-empty string) and ``harness_parameterized`` (the value equals the
    named harness-injected parameter, e.g. the implementation name)."""
    if spec.get("non_empty_string") is True:
        assert isinstance(actual, str) and actual != "", f"{label}: expected non-empty string, got {actual!r}"
    if "harness_parameterized" in spec:
        import openarmature  # noqa: PLC0415

        params = {"implementation_name": openarmature.__implementation_name__}
        param_name = cast("str", spec["harness_parameterized"])
        assert actual == params.get(param_name), (
            f"{label}: expected harness param {param_name!r}={params.get(param_name)!r}, got {actual!r}"
        )


def _assert_metadata_subset(
    label: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    # Subset-compare: every key in `expected` MUST be present in
    # `actual` with the corresponding value (or matching placeholder);
    # additional keys in `actual` are tolerated.
    for key, expected_value in expected.items():
        assert key in actual, f"{label}: missing key {key!r}; got keys {sorted(actual)}"
        actual_value = actual[key]
        if _is_placeholder(expected_value):
            assert isinstance(actual_value, str) and len(actual_value) > 0, (
                f"{label}.{key}: expected placeholder {expected_value!r} match, got {actual_value!r}"
            )
            continue
        if (
            isinstance(expected_value, dict)
            and expected_value
            and set(cast("dict[str, Any]", expected_value)).issubset(_METADATA_MATCHER_SUBKEYS)
        ):
            # Fixture 059 attribution: assertion sub-keys, not a nested mapping.
            _assert_metadata_matcher_subkeys(
                f"{label}.{key}", actual_value, cast("dict[str, Any]", expected_value)
            )
            continue
        if isinstance(expected_value, dict) and isinstance(actual_value, dict):
            _assert_metadata_subset(
                f"{label}.{key}",
                cast("Mapping[str, Any]", actual_value),
                cast("Mapping[str, Any]", expected_value),
            )
            continue
        if isinstance(expected_value, list) and isinstance(actual_value, list):
            # List values may contain placeholders element-by-element
            # (e.g., ``detached_child_trace_ids: ["<trace_id_detached>"]``).
            # Length must match; each element matches by placeholder
            # rules or strict equality.
            expected_list = cast("list[Any]", expected_value)
            actual_list = cast("list[Any]", actual_value)
            assert len(actual_list) == len(expected_list), (
                f"{label}.{key} length: expected {len(expected_list)}, got {len(actual_list)}"
            )
            for i, (exp_el, act_el) in enumerate(zip(expected_list, actual_list, strict=True)):
                if _is_placeholder(exp_el):
                    assert isinstance(act_el, str) and len(act_el) > 0, (
                        f"{label}.{key}[{i}]: expected placeholder {exp_el!r}, got {act_el!r}"
                    )
                else:
                    assert act_el == exp_el, f"{label}.{key}[{i}]: expected {exp_el!r}, got {act_el!r}"
            continue
        assert actual_value == expected_value, (
            f"{label}.{key}: expected {expected_value!r}, got {actual_value!r}"
        )


def _assert_string_or_placeholder(label: str, actual: str | None, expected: Any) -> None:
    if expected is None:
        return
    # Any ``<placeholder>`` form is treated as a wildcard that requires
    # a non-empty string. Cross-occurrence consistency (e.g., the same
    # ``<corr_id_1>`` appearing on multiple Traces must resolve to the
    # same value) is enforced by the fixture's ``invariants:`` block
    # (``correlation_id_consistent_across_traces``, etc.), not by
    # placeholder-binding here. Known shape-only placeholders:
    # ``<uuid>``, ``<any-string>``, ``<corr_id_N>``,
    # ``<trace_id_parent>``, ``<trace_id_child>``,
    # ``<trace_id_instance_N>``.
    if isinstance(expected, str) and expected.startswith("<") and expected.endswith(">"):
        assert isinstance(actual, str) and len(actual) > 0, (
            f"{label}: expected non-empty string, got {actual!r}"
        )
        return
    assert actual == expected, f"{label}: expected {expected!r}, got {actual!r}"


# Unused helper kept for the SamplingConfig import to avoid a future
# F401 when 024's prompt-sampling path lands.
_ = SamplingConfig
