# Spec: pipeline-utilities §10.11 enumerates the reference backends
# (in-memory + SQLite shipped here) and names sibling-package adapters
# for Temporal / DBOS / Restate / Redis as informative future work,
# not part of this package.

"""Concrete Checkpointer backends.

Each backend satisfies the
:class:`openarmature.checkpoint.Checkpointer` Protocol. The current
catalog ships :class:`InMemoryCheckpointer` (no durability; tests +
short-lived runs) and :class:`SQLiteCheckpointer` (single-host
durable). Sibling-package adapters for Temporal, DBOS, Restate, and
Redis are out of scope here.

Users typically import from the package root::

    from openarmature.checkpoint import InMemoryCheckpointer, SQLiteCheckpointer
"""

from .memory import InMemoryCheckpointer
from .sqlite import SerializationMode, SQLiteCheckpointer

__all__ = [
    "InMemoryCheckpointer",
    "SQLiteCheckpointer",
    "SerializationMode",
]
