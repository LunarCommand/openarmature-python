"""Typed fixture root models.

Per the Phase 0 plan: every YAML fixture under
``openarmature-spec/spec/<capability>/conformance/`` lands as one of three
typed shapes. The shape is chosen by a callable discriminator that inspects
the raw dict's top-level keys (no tag field is present in the YAML).

The three shapes:

- :class:`LlmProviderFixture` — ``mock_provider`` is at the top level. Tests
  the stateless ``complete()`` / ``ready()`` operations of the
  ``llm-provider`` capability against canned wire responses. May contain
  ``cases:`` for table-style sub-cases that share the mock provider.

- :class:`CasesFixture` — top-level ``cases:`` list (and no
  ``mock_provider``). Each case carries its own graph definition and
  expected block. Optional shared ``subgraph`` / ``subgraph_with_idx``
  blocks at the top level apply across cases.

- :class:`GraphFixture` — direct graph at the top level (state + entry +
  nodes + edges + initial_state + expected). Optional ``run_count`` for
  determinism fixtures, plus a long tail of optional harness directives
  (``observers``, ``middleware``, ``caller_correlation_id``,
  ``detached_subgraphs``, etc.).

Sub-shapes (state field schemas, node directives, edge specs, middleware
specs, observer specs, expected blocks) live in :mod:`directives` and
:mod:`expectations`. The split is for readability; what's authoritative is
the union of all three shapes here parsing every fixture in the spec
submodule with ``extra="forbid"`` rejecting unknown keys at every level.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Tag

from .directives import (
    EdgeSpec,
    LlmCallSpec,
    MiddlewareConfig,
    MockProviderConfig,
    MockResponse,
    NodeSpec,
    ObserverSpec,
    StateSchema,
)
from .expectations import ExpectedBlock, LlmProviderExpected


class _ForbidExtras(BaseModel):
    """Common base — strict by default. Catches both fixture authors and us
    drifting from the spec; new directives surface as parse errors at the
    point they're introduced rather than getting silently dropped."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Shared sub-shapes
# ---------------------------------------------------------------------------


class SubgraphDefinition(BaseModel):
    """A subgraph at the fixture's top level (singular ``subgraph:`` form
    or one entry of the plural ``subgraphs:`` map). Carries its own state
    schema, nodes, and edges — structurally a mini-graph. Permissive
    extras to absorb subgraph-local middleware blocks (pipeline-utilities/
    020) and any future extension."""

    model_config = ConfigDict(extra="allow")

    name: str | None = None  # singular `subgraph:` form
    state: StateSchema
    entry: str
    nodes: dict[str, NodeSpec]
    edges: list[EdgeSpec]
    middleware: MiddlewareConfig | None = None


class CaseSpec(BaseModel):
    """One sub-case in a ``CasesFixture`` (or in the ``cases:`` block of an
    LlmProviderFixture).

    The shape of a case is fluid — checkpointing fixtures (027–031) bring
    in ``checkpointer``/``first_run_expected_error``/``saved_record_assertions``/
    ``resume`` blocks; llm-provider cases bring in ``call`` /
    ``expected_wire_request``; graph-engine ``007-compile-errors`` cases
    have ``graph:`` wrapping the graph + ``expected_compile_error``;
    observability cases inherit any harness directive a top-level
    ``GraphFixture`` could carry. Permissive extras so the parse keeps
    pace with case-shape evolution without quarterly model edits.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    description: str | None = None
    # graph-engine 007 compile-errors: a case wraps the malformed graph
    # under a `graph:` key alongside `expected_compile_error`.
    graph: dict[str, Any] | None = None
    expected_compile_error: str | None = None
    # The graph-shaped fields when a case carries the graph inline (rather
    # than under ``graph:``).
    state: StateSchema | None = None
    entry: str | None = None
    nodes: dict[str, NodeSpec] | None = None
    edges: list[EdgeSpec] | None = None
    initial_state: dict[str, Any] | None = None
    subgraph: SubgraphDefinition | None = None
    subgraphs: dict[str, SubgraphDefinition] | None = None
    middleware: MiddlewareConfig | None = None
    observers: list[ObserverSpec] | None = None
    expected: ExpectedBlock | None = None
    expected_error: dict[str, Any] | None = None
    # llm-provider sub-cases.
    call: LlmCallSpec | None = None
    expected_wire_request: dict[str, Any] | None = None
    # Checkpointing fixtures (024–031).
    checkpointer: str | None = None
    first_run_expected_error: dict[str, Any] | None = None
    saved_record_assertions: dict[str, Any] | None = None
    latest_record_assertions: dict[str, Any] | None = None
    resume: dict[str, Any] | None = None
    invariants: dict[str, Any] | None = None
    # Either an int (run count) or a list of run configs — fixtures vary.
    populate_checkpointer_via_runs: Any = None
    invoke_with: dict[str, Any] | None = None
    caller_correlation_id: str | None = None
    # observability — mock LLM responses + per-case run config.
    mock_llm: list[MockResponse] | None = None
    invocations: int | None = None


# ---------------------------------------------------------------------------
# LlmProviderFixture
# ---------------------------------------------------------------------------


class LlmProviderFixture(_ForbidExtras):
    """A fixture under ``spec/llm-provider/conformance/``.

    Either ``calls`` is at the top level (single-case) or wrapped in
    ``cases`` (table-style). ``mock_provider`` is always present and
    discriminates this shape from the graph-shaped fixtures.
    """

    mock_provider: MockProviderConfig
    calls: list[LlmCallSpec] | None = None
    cases: list[CaseSpec] | None = None


# ---------------------------------------------------------------------------
# CasesFixture
# ---------------------------------------------------------------------------


class CasesFixture(_ForbidExtras):
    """A fixture whose top level is ``cases:`` rather than a single graph.

    Used by ``007-compile-errors``, the checkpointing fixtures (024–031),
    and the determinism / multi-run observability fixtures. Optional shared
    ``subgraph`` / ``subgraph_with_idx`` at the top level apply across all
    cases. Any other top-level key not listed here is rejected.
    """

    cases: list[CaseSpec]
    # Shared graph-shape blocks that apply across every case. Empirically
    # only `subgraph` and `subgraph_with_idx` appear at the top level of
    # cases-fixtures; the plural `subgraphs` form has not been seen at
    # the cases-fixture top level.
    subgraph: SubgraphDefinition | None = None
    subgraph_with_idx: SubgraphDefinition | None = None


# ---------------------------------------------------------------------------
# GraphFixture
# ---------------------------------------------------------------------------


class GraphFixture(_ForbidExtras):
    """A fixture whose top level IS a single graph.

    Covers the bulk of graph-engine, pipeline-utilities, and observability
    fixtures. Most fields are optional because different fixtures exercise
    different facets of the graph contract.
    """

    # Graph definition (graph-engine + most others).
    state: StateSchema
    entry: str | None = None
    nodes: dict[str, NodeSpec] | None = None
    edges: list[EdgeSpec] | None = None
    initial_state: dict[str, Any] | None = None
    expected: ExpectedBlock | None = None

    # Legacy: top-level expected_error in graph-engine fixtures 008/009.
    expected_error: dict[str, Any] | None = None

    # Subgraph definitions — singular form for graph-engine; plural map for
    # the multi-subgraph cases in observability/008, observability/010, and
    # pipeline-utilities/029.
    subgraph: SubgraphDefinition | None = None
    subgraphs: dict[str, SubgraphDefinition] | None = None
    # Used by pipeline-utilities/020 (fan-out instances expose their idx).
    subgraph_with_idx: SubgraphDefinition | None = None

    # graph-engine §6 observers (since proposal 0003).
    observers: list[ObserverSpec] | None = None

    # pipeline-utilities §6 middleware (proposal 0004) and §10 checkpointer
    # registration (proposal 0008).
    middleware: MiddlewareConfig | None = None
    checkpointer: str | None = None
    clock_stub: dict[str, Any] | None = None

    # Determinism fixtures — graph-engine/010 and pipeline-utilities/011.
    run_count: int | None = None

    # observability / pipeline-utilities cross-cutting harness directives.
    # These are inputs to the test harness, NOT the engine.
    caller_correlation_id: str | None = None
    detached_subgraphs: list[str] | None = None
    detached_fan_outs: list[str] | None = None
    disable_llm_spans: bool | None = None
    mock_llm: list[MockResponse] | None = None
    caller_global_otel_active: bool | None = None
    invocations: int | None = None


# ---------------------------------------------------------------------------
# Discriminator + root union
# ---------------------------------------------------------------------------


def _discriminate_fixture(value: Any) -> Literal["llm_provider", "cases", "graph"]:
    """Pick the fixture shape from a raw YAML dict.

    Order matters: ``mock_provider`` wins over ``cases`` because some
    llm-provider fixtures (e.g. 003-message-validation) have BOTH —
    ``mock_provider`` is the load-bearing discriminator, ``cases`` is just
    the table style for sub-cases.
    """
    if isinstance(value, dict):
        if "mock_provider" in value:
            return "llm_provider"
        if "cases" in value:
            return "cases"
    return "graph"


Fixture = Annotated[
    Annotated[LlmProviderFixture, Tag("llm_provider")]
    | Annotated[CasesFixture, Tag("cases")]
    | Annotated[GraphFixture, Tag("graph")],
    Discriminator(_discriminate_fixture),
]


__all__ = [
    "CaseSpec",
    "CasesFixture",
    "Fixture",
    "GraphFixture",
    "LlmProviderExpected",
    "LlmProviderFixture",
    "SubgraphDefinition",
]
