"""Graph builder: mutable construction → compile to immutable `CompiledGraph`.

Compilation MUST fail if the graph has no declared entry, unreachable
nodes, dangling edges, a node with more than one outgoing edge, or a
field with more than one declared reducer.

`GraphBuilder[StateT]` is parameterized on the graph's state type. Node
functions, conditional-edge functions, and the returned `CompiledGraph[StateT]`
all carry `StateT` forward so consumers get typed `invoke()` return values and
a type-checked `state` parameter on every callback, without `cast(...)` calls.
"""

from collections.abc import Awaitable, Callable, Iterable, Mapping
from types import GenericAlias, UnionType
from typing import Any, Literal, Self, cast, get_args, get_origin

from openarmature.checkpoint.errors import CheckpointStateMigrationChainAmbiguous
from openarmature.checkpoint.migration import MigrationRegistry, StateMigration
from openarmature.checkpoint.protocol import Checkpointer

from .compiled import CompiledGraph
from .edges import ConditionalEdge, EndSentinel, StaticEdge
from .errors import (
    ConflictingReducers,
    DanglingEdge,
    FanOutCountModeAmbiguous,
    FanOutFieldNotList,
    MappingReferencesUndeclaredField,
    MultipleOutgoingEdges,
    NoDeclaredEntry,
    ParallelBranchesNoBranches,
    UnreachableNode,
)
from .fan_out import ConcurrencyResolver, CountResolver, FanOutConfig, FanOutNode
from .middleware import Middleware
from .nodes import FunctionNode, Node
from .parallel_branches import BranchSpec, ParallelBranchesNode
from .projection import FieldNameMatching, ProjectionStrategy
from .reducers import Reducer
from .state import State, field_reducers, resolve_reducer
from .subgraph import SubgraphNode


class GraphBuilder[StateT: State]:
    """Mutable builder for a graph; call `compile()` to produce a `CompiledGraph`."""

    def __init__(self, state_cls: type[StateT]) -> None:
        self.state_cls: type[StateT] = state_cls
        self._nodes: dict[str, Node[StateT]] = {}
        self._edges: list[StaticEdge | ConditionalEdge[StateT]] = []
        self._entry: str | None = None
        # Per-graph middleware in registration order (outer-to-inner).
        # Composed OUTSIDE per-node middleware at runtime per spec §3.
        self._middleware: list[Middleware] = []
        # Optional Checkpointer attached at compile time; ``None`` is
        # the spec §10.1.1 default-off behavior.
        self._checkpointer: Checkpointer | None = None
        # State-migration registry per pipeline-utilities §10.12
        # (proposal 0014). Populated by ``with_state_migration(s)``;
        # passed through to the compiled graph.
        self._migration_registry: MigrationRegistry = MigrationRegistry()

    def add_node(
        self,
        name: str,
        fn: Callable[[StateT], Awaitable[Mapping[str, Any]]],
        *,
        middleware: Iterable[Middleware] | None = None,
    ) -> Self:
        """Register a function node.

        ``fn`` is an async callable taking the current state and
        returning a partial update (a mapping from declared state-field
        names to new values). The merge into parent state happens
        through the field reducers; ``fn`` itself does not see or touch
        the merge.

        ``middleware`` is an optional per-node middleware tuple,
        outer-to-inner. Per-graph middleware (see
        :meth:`add_middleware`) wraps these.

        Raises ``ValueError`` if ``name`` is already declared on this
        builder.
        """
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        self._nodes[name] = FunctionNode[StateT](
            name=name,
            fn=fn,
            middleware=tuple(middleware) if middleware is not None else (),
        )
        return self

    def add_subgraph_node[ChildT: State](
        self,
        name: str,
        compiled: CompiledGraph[ChildT],
        projection: ProjectionStrategy[StateT, ChildT] | None = None,
        *,
        middleware: Iterable[Middleware] | None = None,
    ) -> Self:
        """Register a subgraph as a node.

        ``compiled`` is a child :class:`CompiledGraph` (parameterized
        on its own state type). On entry the parent's state is
        translated to the child's via ``projection.project_in``; on
        exit the child's final state is folded back via
        ``projection.project_out``. The default projection is
        :class:`FieldNameMatching`; pass an :class:`ExplicitMapping`
        for declarative ``inputs``/``outputs`` instead.

        ``middleware`` wraps the whole subgraph dispatch as one atomic
        call from the parent's perspective. Per-graph middleware on
        the parent does NOT propagate into the child; child middleware
        is configured on the child's own builder.

        Raises ``ValueError`` if ``name`` is already declared.
        """
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        proj: ProjectionStrategy[StateT, ChildT] = (
            projection if projection is not None else FieldNameMatching[StateT, ChildT]()
        )
        self._nodes[name] = SubgraphNode[StateT, ChildT](
            name=name,
            compiled=compiled,
            projection=proj,
            middleware=tuple(middleware) if middleware is not None else (),
        )
        return self

    def add_fan_out_node[ChildT: State](
        self,
        name: str,
        *,
        subgraph: CompiledGraph[ChildT],
        collect_field: str,
        target_field: str,
        items_field: str | None = None,
        item_field: str | None = None,
        count: int | CountResolver | None = None,
        concurrency: int | ConcurrencyResolver | None = 10,
        error_policy: str = "fail_fast",
        on_empty: str = "raise",
        count_field: str | None = None,
        inputs: Mapping[str, str] | None = None,
        extra_outputs: Mapping[str, str] | None = None,
        instance_middleware: Iterable[Middleware] | None = None,
        errors_field: str | None = None,
        middleware: Iterable[Middleware] | None = None,
    ) -> Self:
        """Register a fan-out node.

        Validates configuration at registration time:

        - Exactly one of ``items_field`` or ``count`` MUST be
          specified (``fan_out_count_mode_ambiguous`` otherwise).
        - ``items_field`` MUST refer to a list-typed field on the
          parent state schema (``fan_out_field_not_list`` otherwise).
        - ``items_field`` mode requires ``item_field``; ``count``
          mode forbids ``item_field``.
        - ``on_empty`` and ``error_policy`` MUST be one of the
          permitted string literals (``"raise"`` / ``"noop"`` and
          ``"fail_fast"`` / ``"collect"`` respectively).
        - ``inputs`` / ``extra_outputs`` / ``count_field`` field
          references go through the existing
          ``mapping_references_undeclared_field`` rule.
        """
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")

        # Mode validation: exactly one of items_field / count.
        if (items_field is None) == (count is None):
            raise FanOutCountModeAmbiguous(
                node_name=name,
                message=(
                    "must specify exactly one of items_field or count "
                    f"(got items_field={items_field!r}, count={count!r})"
                ),
            )
        if items_field is not None and item_field is None:
            raise FanOutCountModeAmbiguous(node_name=name, message="items_field mode requires item_field")
        if count is not None and item_field is not None:
            raise FanOutCountModeAmbiguous(node_name=name, message="count mode forbids item_field")

        # items_field must be a list-typed parent field.
        if items_field is not None:
            parent_fields = self.state_cls.model_fields
            if items_field not in parent_fields:
                raise MappingReferencesUndeclaredField(
                    direction="fan_out.items_field", side="parent", field_name=items_field
                )
            ann = parent_fields[items_field].annotation
            if not _is_list_typed(ann):
                raise FanOutFieldNotList(node_name=name, field_name=items_field)

        # error_policy + on_empty literal validation.
        if error_policy not in {"fail_fast", "collect"}:
            raise ValueError(
                f"fan-out node {name!r}: error_policy must be 'fail_fast' or 'collect', got {error_policy!r}"
            )
        if on_empty not in {"raise", "noop"}:
            raise ValueError(f"fan-out node {name!r}: on_empty must be 'raise' or 'noop', got {on_empty!r}")

        # *_field references must match declared fields.
        parent_fields = self.state_cls.model_fields
        sub_fields = subgraph.state_cls.model_fields
        if target_field not in parent_fields:
            raise MappingReferencesUndeclaredField(
                direction="fan_out.target_field", side="parent", field_name=target_field
            )
        if collect_field not in sub_fields:
            raise MappingReferencesUndeclaredField(
                direction="fan_out.collect_field", side="subgraph", field_name=collect_field
            )
        # NOTE: item_field is intentionally NOT validated against declared
        # subgraph fields. Per fixture 023, the spec allows item_field to
        # name a field the subgraph doesn't declare (treated as a
        # placeholder when the subgraph doesn't read the item). The
        # runtime projection in fan_out._build_instance_states skips the
        # assignment if the field isn't declared, so non-declared
        # item_field values are effectively no-ops.
        if count_field is not None and count_field not in parent_fields:
            raise MappingReferencesUndeclaredField(
                direction="fan_out.count_field", side="parent", field_name=count_field
            )
        if errors_field is not None and errors_field not in parent_fields:
            raise MappingReferencesUndeclaredField(
                direction="fan_out.errors_field", side="parent", field_name=errors_field
            )
        for sub_f, parent_f in (inputs or {}).items():
            if sub_f not in sub_fields:
                raise MappingReferencesUndeclaredField(
                    direction="fan_out.inputs", side="subgraph", field_name=sub_f
                )
            if parent_f not in parent_fields:
                raise MappingReferencesUndeclaredField(
                    direction="fan_out.inputs", side="parent", field_name=parent_f
                )
        for parent_f, sub_f in (extra_outputs or {}).items():
            if parent_f not in parent_fields:
                raise MappingReferencesUndeclaredField(
                    direction="fan_out.extra_outputs", side="parent", field_name=parent_f
                )
            if sub_f not in sub_fields:
                raise MappingReferencesUndeclaredField(
                    direction="fan_out.extra_outputs", side="subgraph", field_name=sub_f
                )

        cfg = FanOutConfig(
            subgraph=subgraph,
            collect_field=collect_field,
            target_field=target_field,
            items_field=items_field,
            item_field=item_field,
            count=count,
            concurrency=concurrency,
            error_policy=cast(Any, error_policy),
            on_empty=cast(Any, on_empty),
            count_field=count_field,
            inputs=dict(inputs or {}),
            extra_outputs=dict(extra_outputs or {}),
            instance_middleware=tuple(instance_middleware or ()),
            errors_field=errors_field,
        )
        # FanOutNode satisfies the Node[StateT] structural protocol (run
        # returns a partial update; name and middleware are present),
        # but pyright loses the StateT correspondence through the second
        # type parameter — cast restores it for the dict assignment.
        fan_out: Node[StateT] = cast(
            "Node[StateT]",
            FanOutNode[StateT, ChildT](
                name=name,
                config=cfg,
                middleware=tuple(middleware) if middleware is not None else (),
            ),
        )
        self._nodes[name] = fan_out
        return self

    def add_parallel_branches_node(
        self,
        name: str,
        *,
        branches: Mapping[str, BranchSpec[Any]],
        error_policy: Literal["fail_fast", "collect"] = "fail_fast",
        errors_field: str | None = None,
        middleware: Iterable[Middleware] | None = None,
    ) -> Self:
        """Register a parallel-branches node per pipeline-utilities §11.

        ``branches`` is a mapping from non-empty branch name to a
        :class:`BranchSpec`. Insertion order is preserved and is
        the dispatch + merge order per §11.8.

        Validates at registration:

        - ``branches`` non-empty (raises ``ParallelBranchesNoBranches``).
        - Each branch name is a non-empty string (raises ``ValueError``).
        - Each branch's ``inputs`` / ``outputs`` refer only to declared
          fields on the (parent, branch-subgraph) state schemas
          (raises ``MappingReferencesUndeclaredField``).
        - ``errors_field`` (when set) is a declared parent-state field.
        """
        if name in self._nodes:
            raise ValueError(f"node {name!r} already declared")
        if not branches:
            raise ParallelBranchesNoBranches(node_name=name)

        parent_fields = self.state_cls.model_fields
        if errors_field is not None and errors_field not in parent_fields:
            raise MappingReferencesUndeclaredField(
                direction="parallel_branches.errors_field",
                side="parent",
                field_name=errors_field,
            )

        for branch_name, spec in branches.items():
            if not branch_name:
                raise ValueError(f"parallel-branches node {name!r}: branch_name MUST be non-empty")
            sub_fields = spec.subgraph.state_cls.model_fields
            for sub_field, parent_field in spec.inputs.items():
                if sub_field not in sub_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=f"parallel_branches.{branch_name}.inputs",
                        side="subgraph",
                        field_name=sub_field,
                    )
                if parent_field not in parent_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=f"parallel_branches.{branch_name}.inputs",
                        side="parent",
                        field_name=parent_field,
                    )
            for parent_field, sub_field in spec.outputs.items():
                if parent_field not in parent_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=f"parallel_branches.{branch_name}.outputs",
                        side="parent",
                        field_name=parent_field,
                    )
                if sub_field not in sub_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=f"parallel_branches.{branch_name}.outputs",
                        side="subgraph",
                        field_name=sub_field,
                    )

        pb: Node[StateT] = cast(
            "Node[StateT]",
            ParallelBranchesNode[StateT](
                name=name,
                branches=dict(branches),
                error_policy=error_policy,
                errors_field=errors_field,
                middleware=tuple(middleware) if middleware is not None else (),
            ),
        )
        self._nodes[name] = pb
        return self

    def with_checkpointer(self, checkpointer: Checkpointer) -> Self:
        """Register a Checkpointer for the compiled graph.

        At most one Checkpointer per graph; calling
        ``with_checkpointer`` again replaces the previously-stored
        one. Pass the result of :meth:`compile` to
        :meth:`CompiledGraph.invoke` as usual; the engine fires saves
        at every ``completed`` event for outermost-graph and
        subgraph-internal nodes.
        """
        self._checkpointer = checkpointer
        return self

    def with_state_migration(
        self,
        from_version: str,
        to_version: str,
        migrate: Callable[[Any], Any],
    ) -> Self:
        """Register one state migration per pipeline-utilities §10.12.

        On resume, when the saved record's ``schema_version`` does not
        match the current state class's ``schema_version``, the engine
        consults the registry for a chain that bridges the two and
        applies it to the record's state (and to each entry in
        ``parent_states``) before deserialization.

        Migrations MUST be pure: deterministic, no I/O, no implicit
        state. The framework does not police purity (per §10.12.2),
        but violating it risks non-deterministic resume.

        Raises ``CheckpointStateMigrationChainAmbiguous`` at
        registration if the ``(from_version, to_version)`` pair is
        already registered (per spec §10.10 / §10.12.1; proposal
        0018 / spec v0.16.0). Also raises ``ValueError`` if
        ``to_version`` is the empty-string sentinel (the un-declared
        marker per §10.2 is not a valid chain target).
        """
        self._migration_registry.register(
            StateMigration(
                from_version=from_version,
                to_version=to_version,
                migrate=migrate,
            )
        )
        return self

    def with_state_migrations(self, *migrations: StateMigration) -> Self:
        """Register multiple migrations in one call. Convenience over
        ``with_state_migration``; each entry is registered through the
        same path and obeys the same ambiguity rule.

        Pre-validates the full input list against the existing
        registry + against earlier entries in the same call before
        mutating, so a duplicate in the third entry cannot leave
        the first two half-registered. If any duplicate ``(from,
        to)`` pair is detected the call raises
        ``CheckpointStateMigrationChainAmbiguous`` without mutating
        the registry; otherwise all entries register atomically.
        """
        # Pre-validation pass: collect every (from, to) we're about
        # to add, check both against the existing registry and
        # against earlier entries in the input. Raise before
        # mutating if anything collides.
        seen_in_call: set[tuple[str, str]] = set()
        for m in migrations:
            key = (m.from_version, m.to_version)
            if key in self._migration_registry._migrations:  # noqa: SLF001
                raise CheckpointStateMigrationChainAmbiguous(
                    f"duplicate state migration {m.from_version!r}→{m.to_version!r} already registered",
                    from_version=m.from_version,
                    to_version=m.to_version,
                )
            if key in seen_in_call:
                raise CheckpointStateMigrationChainAmbiguous(
                    f"duplicate state migration {m.from_version!r}→"
                    f"{m.to_version!r} repeated in with_state_migrations call",
                    from_version=m.from_version,
                    to_version=m.to_version,
                )
            seen_in_call.add(key)
        # Validation passed — commit them all.
        for migration in migrations:
            self._migration_registry.register(migration)
        return self

    def add_middleware(self, middleware: Middleware) -> Self:
        """Register a per-graph middleware applied to every node in this graph.

        Per-graph middleware composes OUTSIDE per-node middleware.
        Calling order is preserved (outer-to-inner); earlier
        ``add_middleware`` calls produce outer layers in the runtime
        chain.
        """
        self._middleware.append(middleware)
        return self

    def add_edge(self, source: str, target: str | EndSentinel) -> Self:
        """Register a static edge from ``source`` to ``target``.

        ``target`` is either the name of another declared node or the
        :data:`END` sentinel. A node may have at most one outgoing
        static edge; declaring two raises
        :class:`MultipleOutgoingEdges` at ``compile()``. For
        state-dependent branching use :meth:`add_conditional_edge`.
        """
        self._edges.append(StaticEdge(source=source, target=target))
        return self

    def add_conditional_edge(
        self,
        source: str,
        fn: Callable[[StateT], str | EndSentinel],
    ) -> Self:
        """Register a conditional edge from ``source``.

        ``fn`` is a synchronous callable that receives the merged
        post-node state and returns either the name of the next node
        or the :data:`END` sentinel. Returning any other value raises
        :class:`RoutingError` at runtime; an exception inside ``fn``
        becomes an :class:`EdgeException`.

        Like static edges, a node has at most one outgoing edge;
        declaring both a static and a conditional edge from the same
        source raises :class:`MultipleOutgoingEdges` at ``compile()``.
        """
        self._edges.append(ConditionalEdge[StateT](source=source, fn=fn))
        return self

    def set_entry(self, name: str) -> Self:
        """Declare which registered node the graph starts at.

        Calling ``set_entry`` again replaces the previously set entry.
        ``compile()`` raises :class:`NoDeclaredEntry` if no entry was
        set, and :class:`DanglingEdge` if the entry name was never
        declared with ``add_*_node``.
        """
        self._entry = name
        return self

    def compile(self) -> CompiledGraph[StateT]:
        """Validate the builder and return an immutable
        :class:`CompiledGraph`.

        Runs structural checks in order:

        - ``conflicting_reducers`` (state-field reducer conflicts)
        - declarative projection ``validate`` hooks (e.g.
          :class:`ExplicitMapping`)
        - ``no_declared_entry`` / ``dangling_edge`` (entry pointer)
        - ``dangling_edge`` (edge endpoints reference declared nodes)
        - ``multiple_outgoing_edges`` (one edge per source)
        - ``unreachable_node`` (every node reachable from entry)

        The first failing check raises its specific
        :class:`CompileError` subclass; passing means the returned
        graph is ready for :meth:`CompiledGraph.invoke`. A previously
        attached :class:`Checkpointer` is forwarded onto the compiled
        graph.
        """
        # 1. ConflictingReducers — state schema check.
        per_field = field_reducers(self.state_cls)
        for fname, declared in per_field.items():
            if len(declared) > 1:
                raise ConflictingReducers(fname)
        resolved: dict[str, Reducer] = {
            fname: resolve_reducer(declared) for fname, declared in per_field.items()
        }

        # 2. MappingReferencesUndeclaredField — declarative projection
        #    strategies (e.g. `ExplicitMapping`) expose an optional
        #    `validate(parent_cls, child_cls)` hook that we invoke here so
        #    misconfigured mappings fail compile rather than at runtime.
        #    The hook is duck-typed: strategies with nothing declarative to
        #    check (the default `FieldNameMatching`, hand-written imperative
        #    projections) simply omit `validate` and the engine skips it.
        #    ChildT is erased once SubgraphNode is stored as Node[StateT];
        #    the cast restores enough type info to access `compiled.state_cls`
        #    without pyright flagging an unknown member type.
        for node in self._nodes.values():
            if isinstance(node, SubgraphNode):
                sub = cast(SubgraphNode[StateT, State], node)
                validate = getattr(sub.projection, "validate", None)
                if validate is not None:
                    validate(self.state_cls, sub.compiled.state_cls)

        # 3. NoDeclaredEntry.
        if self._entry is None:
            raise NoDeclaredEntry()

        # 4. Entry must point to a declared node (treat as DanglingEdge).
        if self._entry not in self._nodes:
            raise DanglingEdge(source="<entry>", target=self._entry)

        # 5. DanglingEdge — both endpoints of every edge must be declared.
        for edge in self._edges:
            if edge.source not in self._nodes:
                raise DanglingEdge(source=edge.source, target=edge.source)
            if isinstance(edge, StaticEdge) and isinstance(edge.target, str):
                if edge.target not in self._nodes:
                    raise DanglingEdge(source=edge.source, target=edge.target)

        # 6. MultipleOutgoingEdges + index by source for the reachability pass.
        edges_by_source: dict[str, StaticEdge | ConditionalEdge[StateT]] = {}
        for edge in self._edges:
            if edge.source in edges_by_source:
                raise MultipleOutgoingEdges(edge.source)
            edges_by_source[edge.source] = edge

        # 7. UnreachableNode — BFS from entry. Conditional edges over-approximate
        #    by reaching every declared node (we cannot statically know the fn's
        #    range), which keeps the check sound (no false positives).
        reachable = self._reachable_nodes(edges_by_source)
        for node_name in self._nodes:
            if node_name not in reachable:
                raise UnreachableNode(node_name)

        compiled = CompiledGraph[StateT](
            state_cls=self.state_cls,
            entry=self._entry,
            nodes=dict(self._nodes),
            edges=edges_by_source,
            reducers=resolved,
            middleware=tuple(self._middleware),
            migration_registry=self._migration_registry,
        )
        if self._checkpointer is not None:
            compiled.attach_checkpointer(self._checkpointer)
        return compiled

    def _reachable_nodes(
        self,
        edges_by_source: Mapping[str, StaticEdge | ConditionalEdge[StateT]],
    ) -> set[str]:
        assert self._entry is not None
        reachable: set[str] = {self._entry}
        frontier = [self._entry]
        all_names = set(self._nodes.keys())
        while frontier:
            current = frontier.pop()
            edge = edges_by_source.get(current)
            if edge is None:
                continue
            if isinstance(edge, StaticEdge):
                if isinstance(edge.target, str) and edge.target not in reachable:
                    reachable.add(edge.target)
                    frontier.append(edge.target)
            else:
                for name in all_names - reachable:
                    reachable.add(name)
                    frontier.append(name)
        return reachable


def _is_list_typed(annotation: Any) -> bool:
    """True if ``annotation`` resolves to a ``list[...]`` shape.

    Used by ``add_fan_out_node`` to validate that ``items_field`` refers
    to a list-typed parent field. Handles both bare ``list[X]`` and
    ``Annotated[list[X], reducer]`` forms (the latter is how state
    fields commonly attach an `append` reducer).
    """
    if annotation is list:
        return True
    origin = get_origin(annotation)
    if origin is list:
        return True
    # Annotated[T, ...] — peel the metadata, recurse on the type.
    if isinstance(annotation, GenericAlias):
        return False
    args = get_args(annotation)
    if args and origin is None:
        # Likely Annotated; first arg is the underlying type.
        return _is_list_typed(args[0])
    if isinstance(annotation, UnionType):
        return any(_is_list_typed(a) for a in args)
    return False
