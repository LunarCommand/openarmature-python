# Spec mapping: this module realizes the three canonical
# checkpoint-error categories from pipeline-utilities §10.10. None
# inherits from :class:`graph.errors.RuntimeGraphError` because these
# errors are raised outside a node's execution scope — they don't fit
# the graph-engine §4 runtime-error contract that mandates a
# ``recoverable_state`` attribute.

"""Errors raised by the checkpointing layer.

Three canonical categories. None inherits from
:class:`openarmature.graph.errors.RuntimeGraphError` because checkpoint
errors are raised outside a node's execution scope (during resume
load, during a save call, or during record-shape validation) — they
don't fit the runtime-error contract that mandates a
``recoverable_state`` attribute.
"""

from __future__ import annotations


class CheckpointError(Exception):
    """Base for all checkpoint errors. Each subclass carries a
    ``category`` class attribute matching its canonical category
    string."""

    category: str


class CheckpointNotFound(CheckpointError):
    """Raised when ``invoke(resume_invocation=X)`` is called and
    ``Checkpointer.load(X)`` returns ``None``. Non-transient — the
    record genuinely does not exist; retrying without changing the
    invocation_id will never succeed."""

    category = "checkpoint_not_found"

    def __init__(self, invocation_id: str) -> None:
        super().__init__(f"no checkpoint record found for invocation_id={invocation_id!r}")
        self.invocation_id = invocation_id


class CheckpointSaveFailed(CheckpointError):
    """Raised when ``Checkpointer.save`` itself raises during a
    ``completed`` event handler. Engine behavior on save failure is
    implementation-defined; this implementation raises to the caller
    of ``invoke()`` immediately and does NOT retry the save itself
    (documented on :meth:`CompiledGraph.invoke`)."""

    category = "checkpoint_save_failed"

    def __init__(self, invocation_id: str, cause: BaseException) -> None:
        super().__init__(f"Checkpointer.save({invocation_id!r}) raised {type(cause).__name__}: {cause}")
        self.invocation_id = invocation_id
        self.__cause__ = cause


class CheckpointRecordInvalid(CheckpointError):
    """Raised when ``Checkpointer.load(X)`` returns a record whose
    schema is incompatible with the current graph (state shape
    mismatch, missing required fields, or
    ``schema_version`` mismatch). Non-transient — the persisted
    record was written by an incompatible version of the engine."""

    category = "checkpoint_record_invalid"

    def __init__(self, invocation_id: str, message: str) -> None:
        super().__init__(f"checkpoint record for invocation_id={invocation_id!r} is invalid: {message}")
        self.invocation_id = invocation_id


__all__ = [
    "CheckpointError",
    "CheckpointNotFound",
    "CheckpointRecordInvalid",
    "CheckpointSaveFailed",
]
