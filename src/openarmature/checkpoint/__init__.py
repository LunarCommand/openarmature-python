# Spec mapping: this package implements the checkpointing capability
# from pipeline-utilities (spec proposal 0008).
# - Public surface satisfies §10.1 (Checkpointer Protocol),
#   §10.2 (record types), §10.10 (error categories).
# - Save fires at completed events for outermost-graph + subgraph-
#   internal + fan-out nodes per §10.3.
# - Resume via ``invoke(resume_invocation=...)`` restores per §10.4.

"""openarmature.checkpoint — checkpointing capability.

Public surface: the typed :class:`Checkpointer` Protocol,
:class:`CheckpointRecord` / :class:`NodePosition` /
:class:`CheckpointSummary` shapes, the checkpoint error categories,
and two reference backends (in-memory and SQLite).

Users register a backend at graph build time via
``GraphBuilder.with_checkpointer(...)``; the engine then fires saves
at every ``completed`` event for outermost-graph nodes and
subgraph-internal nodes, and ``invoke(resume_invocation=X)`` loads +
restores from a prior record.
"""

from .backends import InMemoryCheckpointer, SerializationMode, SQLiteCheckpointer
from .errors import (
    CheckpointError,
    CheckpointNotFound,
    CheckpointRecordInvalid,
    CheckpointSaveFailed,
    CheckpointStateMigrationChainAmbiguous,
    CheckpointStateMigrationFailed,
    CheckpointStateMigrationMissing,
)
from .migration import MigrationRegistry, StateMigration
from .protocol import (
    Checkpointer,
    CheckpointFilter,
    CheckpointRecord,
    CheckpointSummary,
    NodePosition,
)

__all__ = [
    "CheckpointError",
    "CheckpointFilter",
    "CheckpointNotFound",
    "CheckpointRecord",
    "CheckpointRecordInvalid",
    "CheckpointSaveFailed",
    "CheckpointStateMigrationChainAmbiguous",
    "CheckpointStateMigrationFailed",
    "CheckpointStateMigrationMissing",
    "CheckpointSummary",
    "Checkpointer",
    "InMemoryCheckpointer",
    "MigrationRegistry",
    "NodePosition",
    "SQLiteCheckpointer",
    "SerializationMode",
    "StateMigration",
]
