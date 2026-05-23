"""Subgraphs as nodes.

A compiled graph is used as a node inside another graph. The
subgraph runs against its own state schema; projection between parent
and subgraph is delegated to a ``ProjectionStrategy`` (default:
``FieldNameMatching``; ``ExplicitMapping`` is also available).

When a subgraph runs as part of a parent invocation, its inner-node
events bubble up to outer observers (in addition to the subgraph's
own attached observers), the step counter spans the
subgraph boundary, and the namespace extends. SubgraphNode.run accepts an
optional `_InvocationContext` so the engine can thread that context through;
called without it (e.g., direct test invocation), SubgraphNode falls back to
a fresh subgraph-only invocation.

Parameterized on both the parent's state type (`ParentT`) and the subgraph's
state type (`ChildT`). The outer graph only ever sees `run(state: ParentT)`
; the `ChildT` lives on the `compiled` and `projection` fields and is
invisible at the outer graph's node dispatch site.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from .middleware import Middleware
from .projection import FieldNameMatching, ProjectionStrategy
from .state import State

if TYPE_CHECKING:
    from .compiled import CompiledGraph
    from .observer import _InvocationContext


@dataclass(frozen=True)
class SubgraphNode[ParentT: State, ChildT: State]:
    """A node backed by a compiled subgraph.

    The parent's per-node middleware on a SubgraphNode wraps the
    subgraph dispatch as a single atomic call; parent middleware
    does NOT cross into the subgraph's internal nodes (those are
    wrapped by the subgraph's own middleware independently).
    """

    name: str
    compiled: "CompiledGraph[ChildT]"
    projection: ProjectionStrategy[ParentT, ChildT] = field(
        default_factory=FieldNameMatching[ParentT, ChildT]
    )
    middleware: tuple[Middleware, ...] = field(default_factory=tuple[Middleware, ...])

    async def run(
        self,
        state: ParentT,
        context: "_InvocationContext | None" = None,
    ) -> Mapping[str, Any]:
        """Execute the subgraph and project its result back into the parent.

        When `context` is None (e.g., direct invocation in tests, or a parent
        call that doesn't thread a context), the subgraph runs via its own
        public `invoke()`; a fresh root invocation with no parent observer
        chain.

        When `context` is provided (the engine's normal path during
        a parent run), the subgraph descends into a child context
        that shares the parent's queue + step counter and extends the
        namespace and parent-state stack. Observer events from inner
        nodes bubble up to outer observers.
        """
        # Resume-with-saved-inner-state (spec pipeline-utilities §10.4):
        # if the loaded record's latest save fired from inside this
        # subgraph (or a deeper nested one we'll re-enter), the engine
        # threads the saved inner state through ``pending_resume_states``
        # keyed by descent depth. Consume the matching depth here
        # before falling back to the normal projection — this is what
        # makes "skip step_one, run step_two with its post-merge inner
        # state" work without re-running step_one.
        saved: Any = None
        if context is not None:
            # Descent depth of THIS subgraph's inner = current
            # depth + 1. Outer is depth 0; first subgraph is depth 1.
            target_depth = len(context.parent_states_prefix) + 1
            saved = context.pending_resume_states.pop(target_depth, None)
        if saved is not None:
            # Coerce dict → typed instance if the backend stored JSON;
            # in-memory backends preserve the live instance. A
            # ValidationError on the JSON path means the persisted
            # inner state is incompatible with the current subgraph's
            # state class — surface as CheckpointRecordInvalid per
            # §10.10 rather than a raw pydantic ValidationError.
            sub_initial: ChildT
            if isinstance(saved, dict):
                try:
                    sub_initial = self.compiled.state_cls.model_validate(saved)
                except ValidationError as exc:
                    # Local imports to avoid an import cycle between
                    # graph and checkpoint at module load time.
                    from openarmature.checkpoint.errors import CheckpointRecordInvalid

                    assert context is not None  # saved is non-None only when context is set
                    raise CheckpointRecordInvalid(
                        context.resume_invocation or context.invocation_id,
                        f"saved inner state for subgraph {self.name!r} does not "
                        f"validate against {self.compiled.state_cls.__name__}: {exc}",
                    ) from exc
            else:
                sub_initial = cast("ChildT", saved)
        else:
            sub_initial = self.projection.project_in(state, self.compiled.state_cls)
        if context is None:
            sub_final = await self.compiled.invoke(sub_initial)
        else:
            child_context = context.descend_into_subgraph(
                subgraph_node_name=self.name,
                parent_state=state,
                sub_attached=tuple(self.compiled._attached_observers),
            )
            sub_final = await self.compiled._invoke(sub_initial, child_context)
        return self.projection.project_out(sub_final, state, self.compiled.state_cls)
