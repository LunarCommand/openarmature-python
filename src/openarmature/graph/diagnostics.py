# Spec: realizes the graph-engine §2 compile-time *warning* surface added
# by proposal 0094 -- non-fatal diagnostics, distinct from the MUST-fail
# CompileError categories in errors.py. Warnings are raised through the
# stdlib ``warnings`` machinery so a caller (or the conformance adapter's
# §5.8 ``expected_compile_warning`` directive) captures them with
# ``warnings.catch_warnings(record=True)`` around ``compile()``.

"""Compile-time warnings.

A warning never fails compilation and never changes runtime behavior: it
flags a construction that compiles and runs exactly as written, but is
very likely not what the author meant.

Each warning carries a ``category`` string mirroring the ``category``
convention on :class:`~openarmature.graph.errors.CompileError`, so a
caller can branch on the stable identifier rather than the message text.
"""


class CompileWarning(UserWarning):
    """Base class for non-fatal diagnostics raised during ``compile()``.

    Subclasses a plain ``UserWarning`` so the stdlib filter vocabulary
    (``warnings.simplefilter``, ``-W`` flags, pytest's ``filterwarnings``)
    applies unchanged, and so a caller who wants these fatal can opt in
    with ``warnings.simplefilter("error", CompileWarning)``.
    """

    category: str


class ProjectionReducerRoundTrip(CompileWarning):
    """A projection carries a parent field in and then merges that same
    field back out, into a reducer that is not round-trip-idempotent.

    The projected-out value merges through the parent's reducer exactly
    as any node's return does, so the round-tripped value is applied
    again. Either route the field through an idempotent reducer, or do
    not round-trip it (keep the subgraph read-only with respect to that
    field and have the parent read it directly).

    What the re-application costs varies by reducer AND by surface, which
    is why neither this class nor its message names a single outcome. A
    subgraph projection merges one value, so a growing reducer grows. A
    fan-out channel merges a LIST of per-instance values, which
    ``concat_flatten`` absorbs, ``merge_all`` folds, and ``append``
    rejects outright at state validation.
    """

    # Proposal 0094. A structural heuristic: an implementation cannot
    # statically prove the subgraph left the value unchanged, so this MAY
    # fire on a round-trip that legitimately replaces the value. It fires
    # on the structural condition alone so implementations agree on when
    # it is raised. MUST for a canonical non-round-trip-idempotent
    # reducer; SHOULD for a custom reducer classified non-idempotent.
    category = "projection_reducer_round_trip"

    def __init__(self, *, field_name: str, reducer_name: str, node_name: str | None = None) -> None:
        where = f" on node {node_name!r}" if node_name is not None else ""
        # State the structural fact and leave the consequence open: it
        # differs by reducer and by surface (a fan-out merges a list of
        # per-instance values, so the value returns once PER INSTANCE and
        # append rejects it outright), so any single hard-coded outcome
        # would be wrong for most of the cases this fires on.
        super().__init__(
            f"projection{where} carries field {field_name!r} in and merges it "
            f"back out through the {reducer_name!r} reducer, which is not "
            f"round-trip-idempotent: the value is merged again. Route the "
            f"field through an idempotent reducer, or do not project it back "
            f"out."
        )
        self.field_name = field_name
        self.reducer_name = reducer_name
        self.node_name = node_name


__all__ = ["CompileWarning", "ProjectionReducerRoundTrip"]
