"""State migration types and registry.

Realizes pipeline-utilities §10.12 (proposal 0014). A
``StateMigration`` describes one edge in the migration graph;
``MigrationRegistry`` holds the ordered set and resolves chains
via BFS. Ambiguity (duplicate ``(from, to)`` pairs OR multiple
distinct shortest paths between the same source/sink) is a
configuration-style error per §10.12.1 / §10.12.2.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from .errors import CheckpointStateMigrationChainAmbiguous


@dataclass(frozen=True)
class StateMigration:
    """One edge in the migration graph.

    ``migrate`` receives the most-deserialized form the backend can
    expose that is still independent of the current state class
    (a plain ``dict`` for JSON-backed backends). It MUST return a
    value of the same kind, suitable for the next migration in the
    chain (or for final deserialization into the current state class).

    Migrations MUST be pure: deterministic, no I/O, no implicit
    state. The framework does not police purity per spec §10.12.2
    ("the contract is documented, not policed"); violating it
    risks non-deterministic resume.
    """

    from_version: str
    to_version: str
    migrate: Callable[[Any], Any]


class MigrationRegistry:
    """Ordered set of registered migrations + BFS chain resolution.

    Registration-time invariants:

    - Two migrations with the same ``from_version`` AND
      ``to_version`` raise ``CheckpointStateMigrationChainAmbiguous``
      directly per spec §10.10 (proposal 0018) so the canonical
      category surfaces at the registration boundary without any
      wrapping by the builder.
    - Two migrations with the same ``from_version`` and different
      ``to_version`` are permitted (branched migration graph;
      chain resolution picks a path or raises ambiguity if multiple
      shortest paths exist).

    Resolution-time semantics (per §10.12.2):

    - BFS from ``record.schema_version`` to
      ``current.schema_version``. BFS naturally finds the shortest
      path.
    - Empty registry on mismatch → no path → caller raises
      ``CheckpointStateMigrationMissing``.
    - Non-empty registry with no connecting path → same.
    - Found a unique shortest path → return ordered list.
    - Found multiple distinct shortest paths (same edge count,
      different edge sequences) → raise ``ValueError`` internally;
      ``CompiledGraph._migrate_record`` wraps the ``ValueError`` as
      ``CheckpointStateMigrationChainAmbiguous`` at the resume
      boundary. The internal ``ValueError`` keeps the registry
      module dependency-light (no canonical-error import cycle).
    """

    def __init__(self) -> None:
        self._migrations: dict[tuple[str, str], StateMigration] = {}
        self._edges: dict[str, list[StateMigration]] = {}

    def register(self, migration: StateMigration) -> None:
        key = (migration.from_version, migration.to_version)
        if key in self._migrations:
            # Per spec §10.10 / §10.12.1 (proposal 0018, spec v0.16.0):
            # duplicate-pair detection raises the canonical category
            # directly at registration time. The category surfaces
            # before any resume attempt — neither the builder nor the
            # caller needs to wrap.
            raise CheckpointStateMigrationChainAmbiguous(
                f"duplicate state migration {migration.from_version!r}→"
                f"{migration.to_version!r} registered; chain would be ambiguous",
                from_version=migration.from_version,
                to_version=migration.to_version,
            )
        self._migrations[key] = migration
        self._edges.setdefault(migration.from_version, []).append(migration)

    def __iter__(self) -> Iterator[StateMigration]:
        return iter(self._migrations.values())

    def __len__(self) -> int:
        return len(self._migrations)

    def resolve_chain(
        self,
        from_version: str,
        to_version: str,
    ) -> list[StateMigration] | None:
        """Return an ordered chain of migrations bridging the two
        versions, or ``None`` if no chain exists.

        Raises ``ValueError`` if multiple distinct shortest paths
        exist (ambiguous chain per §10.12.2).
        """
        if from_version == to_version:
            return []

        # BFS that records every shortest-length path. If multiple
        # paths share the minimum length, the chain is ambiguous.
        # Standard BFS finds the shortest distance; the path-recording
        # variant lets us detect ambiguity without a second pass.
        # ``frontier`` items are (version, path_so_far).
        frontier: deque[tuple[str, list[StateMigration]]] = deque()
        frontier.append((from_version, []))
        shortest_paths: list[list[StateMigration]] = []
        shortest_length: int | None = None
        # ``distances`` tracks the BFS layer at which each node was
        # first seen. Frontier entries past the shortest_length layer
        # are pruned.
        distances: dict[str, int] = {from_version: 0}

        while frontier:
            version, path = frontier.popleft()
            depth = len(path)
            # Stop expanding once we've moved past the shortest target.
            if shortest_length is not None and depth >= shortest_length:
                continue
            for edge in self._edges.get(version, []):
                next_version = edge.to_version
                next_path = path + [edge]
                if next_version == to_version:
                    if shortest_length is None:
                        shortest_length = len(next_path)
                    if len(next_path) == shortest_length:
                        shortest_paths.append(next_path)
                    continue
                # Cycle-avoidance: a node revisited at the same or
                # deeper BFS layer can't contribute to a strict-
                # shortest path. Allow re-entry only when the new
                # arrival is at the same layer as the first arrival
                # (distinct shortest paths through the same node).
                # NOTE: the strict-less-than comparison is load-
                # bearing for multi-shortest-path detection — a
                # diamond v1→v2→v4 + v1→v3→v4 lets BFS reach v4 via
                # both v2 and v3 at layer 2, and both paths land in
                # ``shortest_paths``. Tightening this to ``<=`` would
                # break the ambiguity check.
                prior_depth = distances.get(next_version)
                if prior_depth is not None and prior_depth < depth + 1:
                    continue
                distances[next_version] = depth + 1
                frontier.append((next_version, next_path))

        if not shortest_paths:
            return None
        if len(shortest_paths) > 1:
            descriptions = [" → ".join([from_version, *(e.to_version for e in p)]) for p in shortest_paths]
            raise ValueError(
                f"ambiguous migration chain from {from_version!r} to "
                f"{to_version!r}: multiple distinct shortest paths exist "
                f"({descriptions}); register fewer migrations or pick a "
                f"single canonical route"
            )
        return shortest_paths[0]

    def describe(self) -> str:
        """Human-readable description of the registered set, used
        in the ``CheckpointStateMigrationMissing`` error payload.
        Empty registry returns ``"<no migrations registered>"``.

        Output is registration-order (Python's dict preserves
        insertion order). Diff-friendly test assertions should
        not depend on the order across distinct registration
        sequences; if cross-language conformance ever needs a
        canonical order, a future change can sort by
        ``(from_version, to_version)``.
        """
        if not self._migrations:
            return "<no migrations registered>"
        return "\n".join(f"{m.from_version} → {m.to_version}" for m in self._migrations.values())


__all__ = ["MigrationRegistry", "StateMigration"]
