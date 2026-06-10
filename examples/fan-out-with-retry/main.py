"""openarmature demo: summarize a batch of lunar-mission headlines in
parallel, with per-headline retries and timing.

**Use case:** Given a list of lunar-mission news headlines, produce a
one-sentence summary and a topic tag for each one. The headlines are
independent, so fan them out and let them run concurrently. Each
per-headline run hits the LLM, which can transiently fail (rate-limit,
timeout, transient 5xx); wrap each instance in retry middleware so a
flaky call doesn't tank the whole batch. A timing middleware records how
long each instance took.

This is the canonical fan-out shape: N similar tasks, N runtime-determined
from state, the work independent enough to run concurrently. The
per-instance subgraph (summarize → classify) is a complete pipeline in
its own right; it would also work standalone against a single headline.

**What's interesting in the implementation:**

- ``GraphBuilder.add_fan_out_node`` with ``items_field`` mode: one
  instance per element of ``state.headlines``, ``item_field`` carries the
  per-instance input into the subgraph.
- ``extra_outputs`` collects a second per-instance field (``topic``) in
  parallel with the primary ``collect_field`` (``summary``). The two
  parent lists are index-aligned.
- ``instance_middleware=(RetryMiddleware(...), TimingMiddleware(...))``
  wraps EACH instance's whole subgraph invocation. Retries are
  per-instance: a failure on headline 3 doesn't restart headlines 0-2.
- ``concurrency=3`` caps how many instances run in flight at once. Use
  this to be polite to the upstream API.
- ``error_policy`` defaults to ``"fail_fast"``; the first instance
  failure (after retries exhaust) raises and cancels siblings. Set
  the ``COLLECT_MODE`` env var to switch to ``"collect"``: each
  instance runs independently and per-instance failures land in
  ``state.instance_errors`` instead of aborting the batch. The
  ``errors_field="instance_errors"`` knob names where the records go.
  Under COLLECT_MODE, the demo prepends a sentinel headline
  (``[FORCE_FAIL] ...``) that ``summarize`` raises
  ``ProviderUnavailable`` on; retry exhausts, the error lands in
  ``instance_errors``, and the rest of the batch completes. Without
  the sentinel, ``COLLECT_MODE`` would have nothing to capture.
- A ``TimingRecord`` is captured per instance via an ``on_complete``
  callback. ``TimingRecord`` carries the per-call duration but not the
  ``fan_out_index``; that index lives on observer NodeEvents instead.
  The demo prints captured durations in completion order plus a
  wall-clock vs sum-of-durations comparison that shows concurrency
  actually parallelized the work.
- A ``fan_out_config_observer`` reads ``NodeEvent.fan_out_config`` on
  the fan-out node's dispatch event. Inner-instance events carry
  ``fan_out_index`` but not ``fan_out_config``; the config lives on
  the fan-out node's own started / completed pair and gives observers
  a record of the resolved item_count, concurrency, and error_policy
  at dispatch time.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).

Run with:

    uv sync --group examples
    cd examples/fan-out-with-retry
    LLM_API_KEY=sk-... uv run python main.py
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import Field

from openarmature.graph import (
    END,
    CompiledGraph,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    State,
    append,
)
from openarmature.graph.middleware import (
    RetryConfig,
    RetryMiddleware,
    TimingMiddleware,
    TimingRecord,
    deterministic_backoff,
)
from openarmature.llm import OpenAIProvider, ProviderUnavailable, SystemMessage, UserMessage

_provider_instance: OpenAIProvider | None = None


def _get_provider() -> OpenAIProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = OpenAIProvider(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com"),
            model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.environ.get("LLM_API_KEY") or None,
        )
    return _provider_instance


async def _chat(system: str, user: str) -> str:
    response = await _get_provider().complete(
        [SystemMessage(content=system), UserMessage(content=user)],
    )
    return (response.message.content or "").strip()


# ---------------------------------------------------------------------------
# A small batch of headlines. In a real app this would come from an RSS
# feed, a database query, or wherever your batch lives.
# ---------------------------------------------------------------------------

HEADLINES: list[str] = [
    "Artemis II splashes down in Pacific after ten-day lunar flyby",
    "NASA pauses Lunar Gateway program in favor of crewed surface base",
    "Intuitive Machines prepares IM-3 lander for Reiner Gamma touchdown",
    "Lunar Reconnaissance Orbiter spots fresh impact crater on far side",
    "Researchers confirm abundant water ice in permanently shadowed south-pole craters",
]


# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class BatchState(State):
    """Outer graph: list of headlines goes in, parallel lists of summaries
    and topic tags come out. ``instance_errors`` only populates under
    ``error_policy="collect"``; each failed instance contributes one
    record naming its ``fan_out_index`` and the exception category."""

    headlines: list[str] = Field(default_factory=list)
    summaries: Annotated[list[str], append] = Field(default_factory=list)
    topics: Annotated[list[str], append] = Field(default_factory=list)
    instance_errors: Annotated[list[dict[str, Any]], append] = Field(default_factory=list[dict[str, Any]])
    trace: Annotated[list[str], append] = Field(default_factory=list)


class HeadlineState(State):
    """Per-instance subgraph state; one headline, its summary, its topic."""

    headline: str = ""
    summary: str = ""
    topic: str = ""
    trace: Annotated[list[str], append] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-instance subgraph: summarize → classify
# ---------------------------------------------------------------------------


async def summarize(s: HeadlineState) -> Mapping[str, Any]:
    # Sentinel for the COLLECT_MODE demo. Raising a transient error
    # (ProviderUnavailable carries the ``provider_unavailable``
    # category, which retry's default classifier recognizes as
    # retryable) lets the retry middleware exhaust its 3 attempts;
    # the final failure then surfaces according to the fan-out's
    # error_policy. Under fail_fast (default), the batch aborts.
    # Under collect, the failure lands in instance_errors and the
    # batch produces partial results.
    if "[FORCE_FAIL]" in s.headline:
        raise ProviderUnavailable("synthetic failure: provider unavailable (COLLECT_MODE demo)")
    content = await _chat(
        system=(
            "Rewrite the headline as one short sentence (~15 words) that would work as a lead. No preamble."
        ),
        user=s.headline,
    )
    return {"summary": content, "trace": ["summarize"]}


async def classify(s: HeadlineState) -> Mapping[str, Any]:
    content = await _chat(
        system=(
            "Tag the topic of the lunar-mission headline below with ONE word "
            "from this set: crew, lander, orbiter, science, hardware, policy, other. "
            "Reply with just the word, lowercase, no punctuation."
        ),
        user=s.headline,
    )
    tag = content.strip().lower().strip(".")
    return {"topic": tag, "trace": ["classify"]}


def build_headline_subgraph() -> CompiledGraph[HeadlineState]:
    return (
        GraphBuilder(HeadlineState)
        .add_node("summarize", summarize)
        .add_node("classify", classify)
        .add_edge("summarize", "classify")
        .add_edge("classify", END)
        .set_entry("summarize")
        .compile()
    )


# ---------------------------------------------------------------------------
# Instance middleware: retry + timing
# ---------------------------------------------------------------------------
# Both middlewares wrap each instance's whole subgraph invocation. Retry's
# loop is per-instance: if headline 3's first attempt raises a transient
# error, the retry middleware re-invokes the subgraph for headline 3 only.
# Headlines 0-2 (already complete) and 4 (still running) are unaffected.
#
# Timing's on_complete callback fires once per successful (or final-failure)
# instance. ``TimingRecord`` carries duration + outcome but not
# ``fan_out_index``; the index lives on observer NodeEvents, not in the
# middleware's record. The demo prints the captured timings in completion
# order to show "this is what middleware-level timing gives you out of the
# box." For per-instance correlation against the input list, use an
# observer instead (see the observer-hooks example).


# Captured timings, populated by the on_complete callback below.
_timings: list[TimingRecord] = []


async def _record_timing(record: TimingRecord) -> None:
    _timings.append(record)


# ---------------------------------------------------------------------------
# Outer graph
# ---------------------------------------------------------------------------


async def announce(s: BatchState) -> Mapping[str, Any]:
    del s
    return {"trace": ["announce"]}


async def present(s: BatchState) -> Mapping[str, Any]:
    """Marker node so the trace shows the outer presented results.

    The summaries and topics are already on parent state from the fan-out's
    projection; this node just appends to the trace.
    """
    del s
    return {"trace": ["present"]}


def build_graph(error_policy: str = "fail_fast") -> CompiledGraph[BatchState]:
    """Build the fan-out demo graph.

    ``error_policy`` switches between ``"fail_fast"`` (default; first
    exhausted-retry failure raises and cancels the rest) and
    ``"collect"`` (each instance runs independently; failures land in
    ``state.instance_errors`` and the batch produces partial results).
    The smoke test calls this with no argument, exercising the default
    path; main() lets the COLLECT_MODE env var flip to collect.
    """
    headline_subgraph = build_headline_subgraph()

    retry = RetryMiddleware(
        RetryConfig(
            max_attempts=3,
            # Short fixed delay so the demo isn't slow. A production app would
            # use exponential_jitter_backoff (the default).
            backoff=deterministic_backoff(0.2),
        )
    )
    timing = TimingMiddleware(
        node_name="headline_run",
        on_complete=_record_timing,
        clock=time.monotonic,
    )

    return (
        GraphBuilder(BatchState)
        .add_node("announce", announce)
        .add_fan_out_node(
            "headline_runs",
            subgraph=headline_subgraph,
            items_field="headlines",
            item_field="headline",
            collect_field="summary",
            target_field="summaries",
            extra_outputs={"topics": "topic"},
            concurrency=3,
            instance_middleware=(retry, timing),
            error_policy=error_policy,
            errors_field="instance_errors",
        )
        .add_node("present", present)
        .add_edge("announce", "headline_runs")
        .add_edge("headline_runs", "present")
        .add_edge("present", END)
        .set_entry("announce")
        .compile()
    )


async def fan_out_config_observer(event: ObserverEvent) -> None:
    """Print the fan-out node's resolved config when its dispatch event
    fires.

    NodeEvent carries ``fan_out_config`` ONLY on the fan-out node's own
    started / completed pair (the dispatch wrapper); inner-instance
    events carry ``fan_out_index`` but not ``fan_out_config``. Reading
    the config gives observability layers a record of how the dispatch
    actually resolved at runtime; useful when ``count`` or
    ``concurrency`` are callable resolvers whose value isn't visible
    in code.
    """
    if not isinstance(event, NodeEvent):
        return
    if event.fan_out_config is None:
        return
    if event.phase != "started":
        return
    cfg = event.fan_out_config
    print(
        f"  [observer] fan-out node {event.node_name!r} dispatching: "
        f"item_count={cfg.item_count} concurrency={cfg.concurrency} "
        f"error_policy={cfg.error_policy!r}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    # Reset module-level capture so a REPL or repeated-main() driver
    # doesn't accumulate timings across invocations.
    _timings.clear()

    # Set COLLECT_MODE=1 to switch the fan-out error policy from the
    # default fail_fast to collect. Under collect, each instance runs
    # independently and per-instance failures (after retries exhaust)
    # land in state.instance_errors instead of aborting the batch.
    error_policy = "collect" if os.environ.get("COLLECT_MODE") else "fail_fast"
    graph = build_graph(error_policy=error_policy)
    graph.attach_observer(fan_out_config_observer)

    # Under COLLECT_MODE, prepend a deliberately-failing headline so
    # the collect path is exercised end-to-end: retry middleware
    # exhausts on the sentinel, the failure lands in
    # state.instance_errors, and the rest of the batch completes.
    # Default (fail_fast) keeps the headline list clean so the demo's
    # happy path runs to completion.
    if error_policy == "collect":
        headlines = [
            "[FORCE_FAIL] Synthetic failing headline for the COLLECT_MODE demo",
            *HEADLINES,
        ]
    else:
        headlines = list(HEADLINES)
    initial = BatchState(headlines=headlines)

    print("=" * 72)
    print(f"Summarizing {len(headlines)} headlines in parallel (concurrency=3)")
    print(f"error_policy={error_policy!r}")
    print("=" * 72)
    print()

    wall_start = time.monotonic()
    try:
        final = await graph.invoke(initial)
        wall_ms = (time.monotonic() - wall_start) * 1000.0
        # Under collect, failed instances are absent from summaries /
        # topics (their projections don't fire on failure). Pull the
        # failed fan_out_indices out of instance_errors so the print
        # loop can align successes to original positions and mark the
        # gaps for the reader.
        failed_indices = {int(e["fan_out_index"]) for e in final.instance_errors}
        success_iter = iter(zip(final.summaries, final.topics, strict=True))
        print("Results (in input order):")
        print()
        for i, headline in enumerate(final.headlines):
            print(f"  [{i}] {headline}")
            if i in failed_indices:
                print("       (failed after retries; see instance_errors below)")
            else:
                s, t = next(success_iter)
                print(f"       summary: {s}")
                print(f"       topic:   {t}")
            print()
        if final.instance_errors:
            print(f"Captured {len(final.instance_errors)} per-instance error(s):")
            for err in final.instance_errors:
                print(f"  {err}")
            print()
        print("Per-instance timings (in completion order):")
        for nth, record in enumerate(_timings):
            print(f"  #{nth}  {record.duration_ms:7.1f} ms  outcome={record.outcome}")
        sum_ms = sum(record.duration_ms for record in _timings)
        print()
        print(f"  wall-clock total:        {wall_ms:7.1f} ms")
        print(f"  sum of per-instance:     {sum_ms:7.1f} ms")
        print(f"  → concurrency speedup:   {sum_ms / wall_ms:5.2f}x")
    finally:
        await graph.drain()
        if _provider_instance is not None:
            await _provider_instance.aclose()


if __name__ == "__main__":
    asyncio.run(main())
