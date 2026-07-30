"""Subgraph projection strategies.

The default is **no projection in** (a subgraph runs from its own
schema's field defaults) and **field-name matching for projection
out** (subgraph fields whose names match parent fields are merged
back into the parent via the parent's reducers).

A subgraph-as-node MAY also declare ``inputs`` (parent → subgraph,
additive over the default of no-projection-in) and/or ``outputs``
(subgraph → parent, replacement
for field-name matching). Implemented here as `ExplicitMapping`.

A third form, `DeclaredSameName`, names the crossing fields as two
sets for the common case where the parent and subgraph names already
agree: checked like the maps, terse like the default, and a complete
declaration with no field-name-matching fallback.

Strategies parameterize on parent and child state types so consumer-authored
projections get typed `project_in` / `project_out` signatures without
`cast(...)` gymnastics.
"""

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

from .errors import MappingReferencesUndeclaredField
from .state import State


def map_round_tripped_parent_fields(
    inputs: Mapping[str, str] | None,
    outputs: Mapping[str, str] | None,
) -> set[str]:
    """The parent fields a map-form projection carries in and then merges
    back out through the SAME subgraph field.

    ``inputs`` is ``{subgraph_field: parent_field}`` and ``outputs``
    ``{parent_field: subgraph_field}`` -- the shape shared by
    :class:`ExplicitMapping`, the fan-out ``inputs`` / ``extra_outputs``
    config, and a parallel-branches subgraph branch's ``inputs`` /
    ``outputs``, so all three round-trip surfaces resolve here.
    """
    # Proposal 0094: a parent field that is an ``inputs`` source and an
    # ``outputs`` target via DIFFERENT subgraph fields is NOT a round-trip
    # -- the value merged back is a distinct, subgraph-computed value.
    # That is the no-false-positive case the fixtures pin.
    if not inputs or not outputs:
        return set()
    return {
        parent_field for parent_field, sub_field in outputs.items() if inputs.get(sub_field) == parent_field
    }


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
      `ExplicitMapping` and `DeclaredSameName` use this to catch
      field-name typos before any node runs. Imperative custom projections typically have nothing
      declarative to check and can simply omit the method; the engine
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
        """No-projection-in: the subgraph starts from its schema's
        field defaults, ignoring the parent state entirely."""
        return subgraph_state_cls()

    def project_out(
        self,
        subgraph_final_state: ChildT,
        parent_state: ParentT,
        subgraph_state_cls: type[ChildT],
    ) -> Mapping[str, Any]:
        """Project shared field names back to the parent. Subgraph
        fields whose names match a parent field are folded through the
        parent's reducer; non-matching subgraph fields are discarded."""
        return _field_name_match_projection(subgraph_final_state, parent_state, subgraph_state_cls)


class ExplicitMapping[ParentT: State, ChildT: State]:
    """Explicit input/output mapping between parent and subgraph
    state.

    ``inputs``: subgraph_field to parent_field. At entry, the named
    parent field's current value is copied into the named subgraph
    field. Subgraph fields not listed receive their schema-declared
    defaults; there is NO field-name fallback (additive over the
    default no-projection-in).

    ``outputs``: parent_field to subgraph_field. At exit, the named
    subgraph field's value is merged into the named parent field via
    the parent's reducer. Subgraph fields not listed are discarded;
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
        """Construct the subgraph's initial state from ``inputs``.
        Each declared ``subgraph_field → parent_field`` pair copies
        the parent's current value into the subgraph kwarg; subgraph
        fields not listed get their schema defaults."""
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
        """Project back per ``outputs``: each declared
        ``parent_field → subgraph_field`` pair folds the subgraph
        value into the parent's reducer. When ``outputs`` was not
        provided, falls back to field-name matching."""
        if self.outputs is None:
            return _field_name_match_projection(subgraph_final_state, parent_state, subgraph_state_cls)
        return {
            parent_field: getattr(subgraph_final_state, sub_field)
            for parent_field, sub_field in self.outputs.items()
        }

    def validate(self, parent_cls: type[ParentT], subgraph_state_cls: type[ChildT]) -> None:
        """Compile-time check that every field name in ``inputs`` and
        ``outputs`` exists on the relevant state schema. Called once
        per subgraph node from the parent's ``compile()``. Raises
        :class:`MappingReferencesUndeclaredField` on the first typo."""
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

    def round_tripped_parent_fields(self) -> set[str]:
        """The parent fields this projection carries in and then merges
        back out through the SAME subgraph field."""
        # Proposal 0094. When ``outputs`` is absent the effective
        # projection-out is field-name matching, NOT "nothing": every
        # shared name is carried back, so a same-name ``inputs`` entry
        # round-trips just as surely as an explicit ``outputs`` pair.
        #
        # No schema check is needed here: ``validate()`` runs earlier in
        # compile() and proves both halves of a same-name entry exist on
        # their respective schemas, so a same-name entry is necessarily a
        # shared field and field-name matching necessarily carries it back.
        if self.outputs is None:
            return {
                parent_field for sub_field, parent_field in self.inputs.items() if sub_field == parent_field
            }
        return map_round_tripped_parent_fields(self.inputs, self.outputs)


class DeclaredSameName[ParentT: State, ChildT: State]:
    """Declared same-name projection boundary between parent and
    subgraph state.

    Names the fields that cross the boundary as two sets, for the common
    case where the parent and subgraph field names already agree:

    ``in_fields``: at entry, each named field's current parent value is
    copied into the subgraph field OF THE SAME NAME. Subgraph fields not
    named receive their schema-declared defaults.

    ``out_fields``: at exit, each named subgraph field is merged into the
    parent field OF THE SAME NAME, via the parent's reducer. Subgraph
    fields not named are discarded.

    Both names are compile-validated against both schemas, so a rename or
    typo on either side becomes a compile error rather than a silently
    dropped field -- the checked middle between the unchecked
    field-name-matching default (:class:`FieldNameMatching`) and the
    fully-explicit rename maps (:class:`ExplicitMapping`).

    This is a COMPLETE boundary declaration with no fallback: an empty
    ``out_fields`` projects nothing out (it does NOT fall back to
    field-name matching), symmetrically with an empty ``in_fields``. A
    subgraph node wanting implicit field-name matching uses the default
    strategy instead. Renaming across the boundary uses
    :class:`ExplicitMapping`; this form is same-name only.
    """

    # Proposal 0094 (graph-engine §2). Kept a separate strategy rather than
    # extra fields on ExplicitMapping: fusing the two forms would mean
    # introducing the invalid both-forms combination in order to detect it,
    # and would fuse two forms whose fallback rules differ (the maps'
    # outputs=None name-matching fallback vs this form's no-fallback). The
    # spec's ``conflicting_projection_forms`` category presumes a
    # config-schema-shaped surface where both can be declared at once; the
    # builder still enforces it duck-typed, so a custom strategy exposing
    # both forms fails rather than the rule being absent.
    def __init__(
        self,
        *,
        in_fields: Iterable[str] = (),
        out_fields: Iterable[str] = (),
    ) -> None:
        # A bare str satisfies Iterable[str], so `in_fields="notes"` would
        # silently become {'e','n','o','s','t'} and surface later as a
        # compile error about a field the author never typed.
        for name, value in (("in_fields", in_fields), ("out_fields", out_fields)):
            if isinstance(value, str):
                raise TypeError(
                    f"{name} must be a collection of field names, not a bare str; "
                    f"pass {{{value!r}}} for a single field"
                )
        self.in_fields: frozenset[str] = frozenset(in_fields)
        self.out_fields: frozenset[str] = frozenset(out_fields)

    def project_in(self, parent_state: ParentT, subgraph_state_cls: type[ChildT]) -> ChildT:
        """Copy each declared field's parent value into the subgraph
        field of the same name; undeclared subgraph fields get their
        schema defaults."""
        kwargs: dict[str, Any] = {name: getattr(parent_state, name) for name in self.in_fields}
        return subgraph_state_cls(**kwargs)

    def project_out(
        self,
        subgraph_final_state: ChildT,
        parent_state: ParentT,
        subgraph_state_cls: type[ChildT],
    ) -> Mapping[str, Any]:
        """Merge each declared subgraph field into the parent field of
        the same name. No field-name-matching fallback: an empty
        ``out_fields`` projects nothing."""
        return {name: getattr(subgraph_final_state, name) for name in self.out_fields}

    def validate(self, parent_cls: type[ParentT], subgraph_state_cls: type[ChildT]) -> None:
        """Compile-time check that every declared name exists on BOTH
        schemas (the names coincide, so each must). Raises
        :class:`MappingReferencesUndeclaredField` on the first drift."""
        parent_fields = set(parent_cls.model_fields.keys())
        sub_fields = set(subgraph_state_cls.model_fields.keys())

        for direction, names in (("in_fields", self.in_fields), ("out_fields", self.out_fields)):
            for name in sorted(names):
                if name not in parent_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=direction, side="parent", field_name=name
                    )
                if name not in sub_fields:
                    raise MappingReferencesUndeclaredField(
                        direction=direction, side="subgraph", field_name=name
                    )

    def round_tripped_parent_fields(self) -> set[str]:
        """The parent fields carried in and merged back out -- for the
        same-name form, simply the fields named in both sets."""
        return set(self.in_fields & self.out_fields)
