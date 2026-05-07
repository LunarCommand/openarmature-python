"""Concrete Checkpointer backends.

Each backend satisfies the
:class:`openarmature.checkpoint.Checkpointer` Protocol. The current
catalog ships :class:`InMemoryCheckpointer` (no durability; tests +
short-lived runs) and :class:`SQLiteCheckpointer` (single-host
durable). Sibling-package adapters for Temporal, DBOS, Restate, and
Redis are informative per spec §10.11 — not specified here.

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
