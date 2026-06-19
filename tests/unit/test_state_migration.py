"""Focused unit tests for the state-migration surface.

The conformance suite (``tests/conformance/test_state_migration.py``)
covers the spec's behavioral surface end-to-end against fixtures
039-046. These unit tests fill gaps the fixtures don't exercise
directly: BFS edge cases on the registry, multi-shortest-path
ambiguity detection, GraphBuilder ergonomics, and the error
attribute carriage shape.
"""

from __future__ import annotations

import pytest

from openarmature.checkpoint import (
    CheckpointStateMigrationChainAmbiguous,
    CheckpointStateMigrationFailed,
    CheckpointStateMigrationMissing,
    MigrationRegistry,
    StateMigration,
)
from openarmature.graph import END, GraphBuilder, State

# ---------------------------------------------------------------------------
# MigrationRegistry — basic registration + iteration + describe
# ---------------------------------------------------------------------------


def _id(x: int) -> int:
    return x


def test_registry_empty_describes_with_sentinel() -> None:
    registry = MigrationRegistry()
    assert len(registry) == 0
    assert registry.describe() == "<no migrations registered>"


def test_registry_lists_registered_in_order() -> None:
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    registry.register(StateMigration(from_version="v2", to_version="v3", migrate=_id))
    assert len(registry) == 2
    assert "v1 → v2" in registry.describe()
    assert "v2 → v3" in registry.describe()


def test_registry_rejects_empty_to_version() -> None:
    """Empty to_version routes the chain TO the "not declared"
    sentinel — incoherent. Registration
    MUST reject it. Empty from_version stays valid (documented
    bridging path for pre-declaration records)."""
    registry = MigrationRegistry()
    with pytest.raises(ValueError, match="to_version MUST be non-empty"):
        registry.register(StateMigration(from_version="v1", to_version="", migrate=_id))


def test_registry_accepts_empty_from_version_bridging_case() -> None:
    """Empty from_version is the documented Q4 bridging path: pre-
    declaration records carrying the empty sentinel migrate forward
    to a newly-declared schema. The registration MUST succeed; BFS
    treats the empty sentinel as a valid source node."""
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="", to_version="v1", migrate=_id))
    chain = registry.resolve_chain("", "v1")
    assert chain is not None
    assert len(chain) == 1


def test_registry_rejects_duplicate_edge() -> None:
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    # Per spec §10.10 / §10.12.1 (proposal 0018, v0.16.0), duplicate-
    # pair detection raises the canonical category directly at
    # registration time. ``CheckpointStateMigrationChainAmbiguous``
    # inherits from ``CheckpointError`` (not ``ValueError``), so the
    # canonical-category assertion is the right shape.
    with pytest.raises(CheckpointStateMigrationChainAmbiguous, match="duplicate state migration"):
        registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))


# ---------------------------------------------------------------------------
# Chain resolution — empty, identity, single hop, multi hop
# ---------------------------------------------------------------------------


def test_resolve_chain_same_version_returns_empty_chain() -> None:
    registry = MigrationRegistry()
    chain = registry.resolve_chain("v1", "v1")
    assert chain == []


def test_resolve_chain_empty_registry_returns_none() -> None:
    registry = MigrationRegistry()
    assert registry.resolve_chain("v1", "v2") is None


def test_resolve_chain_unrelated_registry_returns_none() -> None:
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v3", to_version="v4", migrate=_id))
    assert registry.resolve_chain("v1", "v2") is None


def test_resolve_chain_single_hop() -> None:
    registry = MigrationRegistry()
    a_to_b = StateMigration(from_version="a", to_version="b", migrate=_id)
    registry.register(a_to_b)
    chain = registry.resolve_chain("a", "b")
    assert chain == [a_to_b]


def test_resolve_chain_multi_hop_in_order() -> None:
    registry = MigrationRegistry()
    a_to_b = StateMigration(from_version="a", to_version="b", migrate=_id)
    b_to_c = StateMigration(from_version="b", to_version="c", migrate=_id)
    c_to_d = StateMigration(from_version="c", to_version="d", migrate=_id)
    # Register out of natural order to verify BFS doesn't depend on
    # registration order.
    registry.register(c_to_d)
    registry.register(a_to_b)
    registry.register(b_to_c)
    chain = registry.resolve_chain("a", "d")
    assert chain == [a_to_b, b_to_c, c_to_d]


def test_resolve_chain_picks_shortest_when_unique() -> None:
    """A short path exists alongside a longer one; BFS picks the short."""
    registry = MigrationRegistry()
    # Diamond with an extra step on one side.
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    registry.register(StateMigration(from_version="v2", to_version="v3", migrate=_id))
    # Long detour: v1 -> v1a -> v1b -> v3.
    registry.register(StateMigration(from_version="v1", to_version="v1a", migrate=_id))
    registry.register(StateMigration(from_version="v1a", to_version="v1b", migrate=_id))
    registry.register(StateMigration(from_version="v1b", to_version="v3", migrate=_id))
    chain = registry.resolve_chain("v1", "v3")
    assert chain is not None
    assert [(m.from_version, m.to_version) for m in chain] == [
        ("v1", "v2"),
        ("v2", "v3"),
    ]


def test_resolve_chain_ambiguous_shortest_paths_raises() -> None:
    """Diamond with two distinct same-length paths is ambiguous.
    ``resolve_chain`` raises the canonical
    ``CheckpointStateMigrationChainAmbiguous``
    directly — no boundary wrap needed at the resume site; the
    registry's exception contract is one type regardless of when
    ambiguity surfaces (register vs resolve)."""
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    registry.register(StateMigration(from_version="v1", to_version="v3", migrate=_id))
    registry.register(StateMigration(from_version="v2", to_version="v4", migrate=_id))
    registry.register(StateMigration(from_version="v3", to_version="v4", migrate=_id))
    with pytest.raises(CheckpointStateMigrationChainAmbiguous, match="ambiguous migration chain") as exc_info:
        registry.resolve_chain("v1", "v4")
    assert exc_info.value.from_version == "v1"
    assert exc_info.value.to_version == "v4"


def test_chain_ambiguous_category_string() -> None:
    """The canonical category string."""
    exc = CheckpointStateMigrationChainAmbiguous("boom")
    assert exc.category == "checkpoint_state_migration_chain_ambiguous"


def test_chain_ambiguous_carries_identity_when_set() -> None:
    exc = CheckpointStateMigrationChainAmbiguous(
        "duplicate v1→v2 registered",
        from_version="v1",
        to_version="v2",
    )
    assert exc.from_version == "v1"
    assert exc.to_version == "v2"


def test_chain_ambiguous_carries_none_when_unset() -> None:
    exc = CheckpointStateMigrationChainAmbiguous("boom")
    assert exc.from_version is None
    assert exc.to_version is None


# ---------------------------------------------------------------------------
# GraphBuilder ergonomics — singular + plural registration
# ---------------------------------------------------------------------------


class _Sv1(State):
    schema_version = "v1"
    x: int = 0


def _build_minimal_graph() -> GraphBuilder[_Sv1]:
    async def _noop(_s: _Sv1) -> dict[str, int]:
        return {}

    return GraphBuilder(_Sv1).add_node("noop", _noop).add_edge("noop", END).set_entry("noop")


def test_builder_with_state_migration_singular() -> None:
    builder = _build_minimal_graph()
    builder.with_state_migration("v0", "v1", _id)
    compiled = builder.compile()
    assert len(compiled.migration_registry) == 1


def test_builder_with_state_migrations_plural() -> None:
    builder = _build_minimal_graph()
    builder.with_state_migrations(
        StateMigration(from_version="v0", to_version="v1", migrate=_id),
        StateMigration(from_version="v1", to_version="v2", migrate=_id),
    )
    compiled = builder.compile()
    assert len(compiled.migration_registry) == 2


def test_builder_duplicate_registration_raises() -> None:
    builder = _build_minimal_graph()
    builder.with_state_migration("v0", "v1", _id)
    with pytest.raises(CheckpointStateMigrationChainAmbiguous):
        builder.with_state_migration("v0", "v1", _id)


def test_builder_with_state_migrations_atomic_on_duplicate() -> None:
    """The plural ``with_state_migrations`` pre-validates the full
    input list before mutating. A duplicate in the middle of the
    list MUST NOT leave the earlier entries half-registered."""
    builder = _build_minimal_graph()
    # Pre-seed one entry so the third in the plural call collides.
    builder.with_state_migration("v0", "v1", _id)

    with pytest.raises(CheckpointStateMigrationChainAmbiguous):
        builder.with_state_migrations(
            StateMigration(from_version="v1", to_version="v2", migrate=_id),
            StateMigration(from_version="v2", to_version="v3", migrate=_id),
            # Collides with the pre-seeded v0→v1.
            StateMigration(from_version="v0", to_version="v1", migrate=_id),
        )
    # Registry still has only the pre-seed; the v1→v2 and v2→v3
    # from the failed call MUST NOT have registered.
    compiled = builder.compile()
    keys = [(m.from_version, m.to_version) for m in compiled.migration_registry]
    assert keys == [("v0", "v1")]


def test_builder_with_state_migrations_atomic_on_internal_duplicate() -> None:
    """A duplicate pair WITHIN the with_state_migrations input list
    (not against an already-registered entry) also raises before
    mutating."""
    builder = _build_minimal_graph()
    with pytest.raises(CheckpointStateMigrationChainAmbiguous):
        builder.with_state_migrations(
            StateMigration(from_version="v1", to_version="v2", migrate=_id),
            # Same key as the entry above — internal duplicate.
            StateMigration(from_version="v1", to_version="v2", migrate=_id),
        )
    # Nothing registered.
    compiled = builder.compile()
    assert list(compiled.migration_registry) == []


# ---------------------------------------------------------------------------
# Error attribute carriage
# ---------------------------------------------------------------------------


def test_migration_missing_carries_identity() -> None:
    exc = CheckpointStateMigrationMissing(
        "no chain",
        from_version="v1",
        to_version="v2",
        registered_migrations_count=3,
        registry_description="v3 → v4\nv5 → v6\nv7 → v8",
    )
    assert exc.from_version == "v1"
    assert exc.to_version == "v2"
    assert exc.registered_migrations_count == 3
    assert "v3 → v4" in exc.registry_description


def test_migration_failed_carries_identity() -> None:
    exc = CheckpointStateMigrationFailed(
        "boom",
        from_version="v1",
        to_version="v2",
    )
    assert exc.from_version == "v1"
    assert exc.to_version == "v2"


def test_migration_failed_preserves_cause_when_raised_from() -> None:
    """The error's ``__cause__`` carries the original migration
    exception when raised from a try/except. This mirrors the engine's
    chain-application wrap."""
    underlying = KeyError("missing field")
    try:
        try:
            raise underlying
        except KeyError as exc:
            raise CheckpointStateMigrationFailed(
                "boom",
                from_version="v1",
                to_version="v2",
            ) from exc
    except CheckpointStateMigrationFailed as final:
        assert final.__cause__ is underlying
