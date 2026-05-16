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


def test_registry_rejects_duplicate_edge() -> None:
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    with pytest.raises(ValueError, match="duplicate state migration"):
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
    """Diamond with two distinct same-length paths is ambiguous per spec §10.12.2."""
    registry = MigrationRegistry()
    registry.register(StateMigration(from_version="v1", to_version="v2", migrate=_id))
    registry.register(StateMigration(from_version="v1", to_version="v3", migrate=_id))
    registry.register(StateMigration(from_version="v2", to_version="v4", migrate=_id))
    registry.register(StateMigration(from_version="v3", to_version="v4", migrate=_id))
    with pytest.raises(ValueError, match="ambiguous migration chain"):
        registry.resolve_chain("v1", "v4")


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
    with pytest.raises(ValueError, match="duplicate state migration"):
        builder.with_state_migration("v0", "v1", _id)


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
