"""Cross-backend correlation primitives — no OTel deps.

Verifies the cross-backend correlation contract independently of any
backend mapping. Lives in the unit test root rather than under any
backend-specific directory because correlation_id is core: it MUST be
readable from any user code (node bodies, middleware, observers) even
when no observability backend is configured.
"""

from __future__ import annotations

import asyncio

import pytest

from openarmature.graph import END, GraphBuilder, NodeException, State
from openarmature.observability import current_correlation_id


class _S(State):
    captured: str = ""
    flag: bool = False


async def _read_correlation(state: _S) -> dict[str, str]:
    cid = current_correlation_id()
    return {"captured": cid or ""}


# ---------------------------------------------------------------------------
# §3.1 lifecycle: caller-supplied + auto-generated
# ---------------------------------------------------------------------------


async def test_caller_supplied_correlation_id_visible_inside_node() -> None:
    """User code in a node body can read the supplied correlation_id
    via :func:`current_correlation_id` — the spec's mandated
    cross-backend join key surface."""
    g = GraphBuilder(_S).add_node("read", _read_correlation).add_edge("read", END).set_entry("read").compile()
    final = await g.invoke(_S(), correlation_id="my-business-request-42")
    assert final.captured == "my-business-request-42"


async def test_auto_generated_correlation_id_is_uuidv4() -> None:
    """When the caller does not supply a correlation_id the framework
    MUST auto-generate a canonical 36-character UUIDv4."""
    g = GraphBuilder(_S).add_node("read", _read_correlation).add_edge("read", END).set_entry("read").compile()
    final = await g.invoke(_S())  # no caller correlation_id
    cid = final.captured
    # Canonical UUIDv4 form: xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx (36 chars).
    assert len(cid) == 36
    assert cid[14] == "4"  # version-4 nibble
    parts = cid.split("-")
    assert len(parts) == 5
    assert [len(p) for p in parts] == [8, 4, 4, 4, 12]


async def test_correlation_id_resets_between_invocations() -> None:
    """The context resets after the invocation completes so subsequent
    invocations get fresh correlation IDs."""
    # Outside any invocation, correlation_id is None.
    assert current_correlation_id() is None
    g = GraphBuilder(_S).add_node("read", _read_correlation).add_edge("read", END).set_entry("read").compile()
    await g.invoke(_S(), correlation_id="invocation-one")
    # After invoke returns, the ContextVar is reset.
    assert current_correlation_id() is None
    final2 = await g.invoke(_S(), correlation_id="invocation-two")
    assert final2.captured == "invocation-two"
    # And after the second invoke too.
    assert current_correlation_id() is None


async def test_correlation_id_isolated_across_concurrent_invocations() -> None:
    """ContextVar isolation: two concurrent invocations see their own
    correlation_id values — no cross-contamination via the shared
    ContextVar."""
    g = GraphBuilder(_S).add_node("read", _read_correlation).add_edge("read", END).set_entry("read").compile()
    final_a, final_b = await asyncio.gather(
        g.invoke(_S(), correlation_id="A"),
        g.invoke(_S(), correlation_id="B"),
    )
    assert final_a.captured == "A"
    assert final_b.captured == "B"


# ---------------------------------------------------------------------------
# §3.2 distinction from invocation_id
# ---------------------------------------------------------------------------


async def test_correlation_id_and_invocation_id_are_structurally_distinct() -> None:
    """``correlation_id`` and ``invocation_id`` serve different
    purposes and MUST NOT be
    conflated. Drive a real invocation with a checkpointer and read
    both ids from the saved record (deterministic) — plus an in-body
    cross-check via the public ContextVar readers — to verify the
    framework treats them as independent fields."""
    from openarmature.checkpoint import InMemoryCheckpointer
    from openarmature.observability import current_invocation_id

    captured: dict[str, str | None] = {}

    async def read_both(_s: _S) -> dict[str, str]:
        captured["correlation_id"] = current_correlation_id()
        captured["invocation_id"] = current_invocation_id()
        return {"captured": captured.get("invocation_id") or ""}

    cp = InMemoryCheckpointer()
    g = (
        GraphBuilder(_S)
        .add_node("read", read_both)
        .add_edge("read", END)
        .set_entry("read")
        .with_checkpointer(cp)
        .compile()
    )
    await g.invoke(_S(), correlation_id="user-supplied-cid")

    # In-body cross-check: the two ContextVars MUST return distinct
    # strings. ``correlation_id`` is the caller-supplied value;
    # ``invocation_id`` is framework-minted.
    body_corr = captured["correlation_id"]
    body_inv = captured["invocation_id"]
    assert body_corr == "user-supplied-cid"
    assert body_inv is not None and body_inv != body_corr

    # Saved-record cross-check: the framework persists both fields
    # independently and they MUST differ.
    summaries = list(await cp.list())
    assert len(summaries) == 1
    record = await cp.load(summaries[0].invocation_id)
    assert record is not None
    assert record.correlation_id == "user-supplied-cid"
    assert record.invocation_id != record.correlation_id
    assert record.invocation_id == body_inv


# ---------------------------------------------------------------------------
# Outside-invocation safety
# ---------------------------------------------------------------------------


def test_current_correlation_id_returns_none_outside_invocation() -> None:
    """Reading ``current_correlation_id()`` outside any invocation
    MUST return None (not raise, not return empty string). User code
    that may run inside or outside a graph context can rely on this."""
    assert current_correlation_id() is None


# ---------------------------------------------------------------------------
# Phase 5 / §10.4 step 3 + 4 — resume preserves correlation_id, mints new
# invocation_id. Already covered in test_checkpoint.py at the record
# level; here we additionally verify the user-visible ContextVar half.
# ---------------------------------------------------------------------------


async def test_resume_preserves_correlation_id_visible_to_user_code() -> None:
    """Resume MUST preserve the original correlation_id verbatim. The
    Phase 5 checkpoint test verifies
    this at the saved-record level; here we additionally verify it
    propagates to the ContextVar that user code reads from inside
    node bodies during the resumed invocation."""
    from openarmature.checkpoint import InMemoryCheckpointer

    class _ResumeState(State):
        flag: bool = False
        observed_cid: str = ""

    fail_once = [True]

    async def maybe_fail(_s: _ResumeState) -> dict[str, str | bool]:
        # Read the correlation_id from inside the node body.
        cid = current_correlation_id() or ""
        if fail_once[0]:
            fail_once[0] = False
            raise RuntimeError("first-run abort")
        return {"flag": True, "observed_cid": cid}

    cp = InMemoryCheckpointer()
    # The flaky-node abort needs a save to fire BEFORE the failure so
    # the resume path has something to load from. Single-node graph
    # would never save (the abort happens before its own merge); add
    # a ``pre`` node whose successful save arms the resume.
    pre = (
        GraphBuilder(_ResumeState)
        .add_node("pre", _pre)
        .add_node("a", maybe_fail)
        .add_edge("pre", "a")
        .add_edge("a", END)
        .set_entry("pre")
        .with_checkpointer(cp)
        .compile()
    )
    fail_once[0] = True
    with pytest.raises(NodeException):
        await pre.invoke(_ResumeState(), correlation_id="my-correlation-cid")
    # Find the saved invocation_id from the only record in the
    # checkpointer.
    summaries = list(await cp.list())
    assert len(summaries) == 1
    saved_invocation_id = summaries[0].invocation_id

    # Resume — the flaky node now succeeds and reads the
    # correlation_id from the ContextVar inside its body.
    final = await pre.invoke(_ResumeState(), resume_invocation=saved_invocation_id)
    assert final.observed_cid == "my-correlation-cid"


async def _pre(_s: State) -> dict[str, bool]:
    return {"flag": True}
