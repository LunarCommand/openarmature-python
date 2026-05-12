"""Subgraph projection strategies.

The default is **no projection in** (a subgraph runs from its own
schema's field defaults) and **field-name matching for projection
out** (subgraph fields whose names match parent fields are merged
back into the parent via the parent's reducers).

A subgraph-as-node MAY also declare ``inputs`` (parent → subgraph,
additive over the default of no-projection-in) and/or ``outputs``
(subgraph → parent, replacement
for field-name matching). Implemented here as `ExplicitMapping`.

Strategies parameterize on parent and child state types so consumer-authored
projections get typed `project_in` / `project_out` signatures without
`cast(...)` gymnastics.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from .errors import MappingReferencesUndeclaredField
from .state import State


def _field_name_match_projection[ChildT: State](
    subgraph_final_state: ChildT,
    parent_state: State,
    subgraph_state_cls: type[ChildT],
) -> Mapping[str, Any]:
    """Default projection-out: subgraph fields whose names match
    parent fields are merged back via the parent's reducers;
    non-matching subgraph fields are discarded.

    Shared by ``FieldNameMatching.project_out`` (which always uses it)
    and ``ExplicitMapping.project_out`` (which falls back to it when
    ``outputs`` was not declared).
    """
    parent_fields = set(type(parent_state).model_fields.keys())
    sub_fields = set(subgraph_state_cls.model_fields.keys())
    shared = parent_fields & sub_fields
    return {name: getattr(subgraph_final_state, name) for name in shared}


class ProjectionStrategy[ParentT: State, ChildT: State](Protocol):
    """Strategy for moving state across the parent ↔ subgraph boundary.

    Two required methods plus one optional hook:

    - `project_in` and `project_out` are required: the engine calls them on
      every subgraph step.
    - `validate(parent_cls, subgraph_state_cls) -> None` is an *optional*
      compile-time validation hook. If a strategy defines it, the parent
      graph's `compile()` calls it once per `SubgraphNode`; the strategy
      may raise a `CompileError` subclass when its declarations don't
      match the supplied schemas. Declarative strategies like
      `ExplicitMapping` use this to catch field-name typos before any
      node runs. Imperative custom projections typically have nothing
      declarative to check and can simply omit the method — the engine
      uses duck typing (`getattr`) to find it.
    """

    def project_in(self, parent_state: ParentT, subgraph_state_cls: type[ChildT]) -> ChildT:
        """Build the subgraph's initial state at the moment it begins."""
        raise NotImplementedError

    def project_out(
        self,
        subgraph_final_state: ChildT,
        parent_state: ParentT,
        subgraph_state_cls: type[ChildT],
    ) -> Mapping[str, Any]:
        """Project the subgraph's final state back to the parent as a partial update."""
        raise NotImplementedError


class FieldNameMatching[ParentT: State, ChildT: State]:
    """Default subgraph projection strategy.

    Parameterized for protocol conformance under generics. ``ParentT``
    is not consumed (the default projection ignores parent state on
    the way in), but carrying the type variable keeps the default
    assignable to ``ProjectionStrategy[ParentT, ChildT]`` without type
    gymnastics at the SubgraphNode default-factory site.
    """

    def project_in(self, parent_state: ParentT, subgraph_state_cls: type[ChildT]) -> ChildT:
        return subgraph_state_cls()

    def project_out(
        self,
        subgraph_final_state: ChildT,
        parent_state: ParentT,
        subgraph_state_cls: type[ChildT],
    ) -> Mapping[str, Any]:
        return _field_name_match_projection(subgraph_final_state, parent_state, subgraph_state_cls)


class ExplicitMapping[ParentT: State, ChildT: State]:
    """Explicit input/output mapping between parent and subgraph
    state.

    ``inputs``: subgraph_field → parent_field. At entry, the named
    parent field's current value is copied into the named subgraph
    field. Subgraph fields not listed receive their schema-declared
    defaults — there is NO field-name fallback (additive over the
    default no-projection-in).

    ``outputs``: parent_field → subgraph_field. At exit, the named
    subgraph field's value is merged into the named parent field via
    the parent's reducer. Subgraph fields not listed are discarded —
    ``outputs`` REPLACES field-name matching for projection-out.

    The two directions are independent: pass either, both, or
    neither. The ``outputs`` field distinguishes "absent" (default
    applies) from "present but empty"; ``outputs=None`` means absent
    (fall back to field-name matching), ``outputs={}`` means present
    and empty (project nothing). For ``inputs`` the two defaults
    coincide (no-projection-in either way), so the distinction is
    only meaningful for ``outputs``.
    """

    def __init__(
        self,
        *,
        inputs: Mapping[str, str] | None = None,
        outputs: Mapping[str, str] | None = None,
    ) -> None:
        self.inputs: dict[str, str] = dict(inputs) if inputs is not None else {}
        # Preserve absence on outputs so project_out can fall back to
        # field-name matching when None.
        self.outputs: dict[str, str] | None = dict(outputs) if outputs is not None else None

    def project_in(self, parent_state: ParentT, subgraph_state_cls: type[ChildT]) -> ChildT:
        kwargs: dict[str, Any] = {
            sub_field: getattr(parent_state, parent_field) for sub_field, parent_field in self.inputs.items()
        }
        return subgraph_state_cls(**kwargs)

    def project_out(
        self,
        subgraph_final_state: ChildT,
        parent_state: ParentT,
        subgraph_state_cls: type[ChildT],
    ) -> Mapping[str, Any]:
        if self.outputs is None:
            # Outputs absent → spec default of field-name matching applies.
            return _field_name_match_projection(subgraph_final_state, parent_state, subgraph_state_cls)
        return {
            parent_field: getattr(subgraph_final_state, sub_field)
            for parent_field, sub_field in self.outputs.items()
        }

    def validate(self, parent_cls: type[ParentT], subgraph_state_cls: type[ChildT]) -> None:
        parent_fields = set(parent_cls.model_fields.keys())
        sub_fields = set(subgraph_state_cls.model_fields.keys())

        for sub_field, parent_field in self.inputs.items():
            if sub_field not in sub_fields:
                raise MappingReferencesUndeclaredField(
                    direction="inputs", side="subgraph", field_name=sub_field
                )
            if parent_field not in parent_fields:
                raise MappingReferencesUndeclaredField(
                    direction="inputs", side="parent", field_name=parent_field
                )

        if self.outputs is not None:
            for parent_field, sub_field in self.outputs.items():
                if parent_field not in parent_fields:
                    raise MappingReferencesUndeclaredField(
                        direction="outputs", side="parent", field_name=parent_field
                    )
                if sub_field not in sub_fields:
                    raise MappingReferencesUndeclaredField(
                        direction="outputs", side="subgraph", field_name=sub_field
                    )
