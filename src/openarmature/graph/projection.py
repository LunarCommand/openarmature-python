"""Subgraph projection strategies.

Per spec v0.1.1 §2 Subgraph: the default is **no projection in** (a subgraph
runs from its own schema's field defaults) and **field-name matching for
projection out** (subgraph fields whose names match parent fields are merged
back into the parent via the parent's reducers).

`ProjectionStrategy` is exposed as a seam so proposal 0002 (explicit
input/output mapping) can slot in without changes to the engine's compile or
execute paths.
"""

from collections.abc import Mapping
from typing import Any, Protocol

from .state import State


class ProjectionStrategy(Protocol):
    """Strategy for moving state across the parent ↔ subgraph boundary."""

    def project_in(self, parent_state: State, subgraph_state_cls: type[State]) -> State: ...

    def project_out(
        self,
        subgraph_final_state: State,
        parent_state: State,
        subgraph_state_cls: type[State],
    ) -> Mapping[str, Any]: ...


class FieldNameMatching:
    """Default projection per spec v0.1.1 §2 Subgraph."""

    def project_in(self, parent_state: State, subgraph_state_cls: type[State]) -> State:
        return subgraph_state_cls()

    def project_out(
        self,
        subgraph_final_state: State,
        parent_state: State,
        subgraph_state_cls: type[State],
    ) -> Mapping[str, Any]:
        parent_fields = set(type(parent_state).model_fields.keys())
        sub_fields = set(subgraph_state_cls.model_fields.keys())
        shared = parent_fields & sub_fields
        return {name: getattr(subgraph_final_state, name) for name in shared}
