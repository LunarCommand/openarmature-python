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
  In ``degrade`` mode a ``FailureIsolationMiddleware`` is prepended as
  the outermost layer (retry stays inner, so it still sees raw
  transients first).
- ``concurrency=3`` caps how many instances run in flight at once. Use
  this to be polite to the upstream API.
- The ``MODE`` env var selects the per-instance failure posture.
  ``"fail_fast"`` (default) raises on the first instance whose retries
  exhaust and cancels its siblings. ``"collect"`` lets each instance
  run independently and lands per-instance failures in
  ``state.instance_errors`` (named by ``errors_field``) instead of
  aborting. ``"degrade"`` wraps each instance in
  ``FailureIsolationMiddleware`` (outermost) so an exhausted instance
  is caught and returns a placeholder partial, leaving the batch intact
  with a degraded entry in place. ``collect`` and ``degrade`` both
  prepend a sentinel headline (``[FORCE_FAIL] ...``) that ``summarize``
  raises ``ProviderUnavailable`` on, so there is a failure to handle;
  ``fail_fast`` keeps the list clean for the happy path.
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
- In ``degrade`` mode a ``failure_isolation_observer`` captures each
  ``FailureIsolatedEvent``; the demo prints its ``event_name``, the
  resolved ``caught_exception.category`` (the originating cause, e.g.
  ``provider_unavailable``, not the masking ``node_exception``), and the
  exhausting ``attempt_index``.

**Configuration** (env vars; OpenAI defaults shown):

- ``LLM_BASE_URL`` defaults to ``https://api.openai.com``. **Host root only.**
- ``LLM_MODEL`` defaults to ``gpt-4o-mini``.
- ``LLM_API_KEY`` required (empty for local servers that don't authenticate).
- ``MODE`` defaults to ``fail_fast``. One of ``fail_fast`` / ``collect`` /
  ``degrade`` (see the failure-posture bullet above).

Run with:

    uv sync --group examples
    cd examples/fan-out-with-retry
    LLM_API_KEY=sk-... uv run python main.py

    # exercise the degrade failure-path: prepends a synthetic failing
    # headline and prints the Failure-isolation events block
    MODE=degrade LLM_API_KEY=sk-... uv run python main.py
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
    FailureIsolatedEvent,
    GraphBuilder,
    NodeEvent,
    ObserverEvent,
    State,
    append,
)
from openarmature.graph.middleware import (
    FailureIsolationMiddleware,
    Middleware,
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
    # Sentinel for the collect / degrade failure-path demos (those modes
    # prepend a [FORCE_FAIL] headline). Raising a transient error
    # (ProviderUnavailable carries the ``provider_unavailable`` category,
    # which retry's default classifier recognizes as retryable) lets the
    # retry middleware exhaust its 3 attempts; the final failure then
    # surfaces according to MODE: under collect it lands in
    # instance_errors and the batch produces partial results; under
    # degrade FailureIsolationMiddleware catches it and substitutes a
    # placeholder so the batch finishes intact.
    if "[FORCE_FAIL]" in s.headline:
        raise ProviderUnavailable("synthetic failure: provider unavailable (failure-path demo)")
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


# Captured failure-isolation events, populated by the observer below.
# Only fires in ``degrade`` mode, where FailureIsolationMiddleware catches
# an exhausted instance and emits one FailureIsolatedEvent per degraded
# instance.
_isolated: list[FailureIsolatedEvent] = []


async def failure_isolation_observer(event: ObserverEvent) -> None:
    """Capture each FailureIsolatedEvent so the demo can surface the
    resolved failure cause.

    When FailureIsolation wraps Retry at a fan-out instance, the engine
    has already wrapped the originating error as a node_exception carrier
    by the time isolation catches it. ``caught_exception.category``
    resolves through that carrier to the originating cause, so a degraded
    instance reports ``provider_unavailable`` (what actually failed)
    rather than the masking ``node_exception``.
    """
    if isinstance(event, FailureIsolatedEvent):
        _isolated.append(event)


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


def build_graph(mode: str = "fail_fast") -> CompiledGraph[BatchState]:
    """Build the fan-out demo graph.

    ``mode`` selects the per-instance failure posture:

    - ``"fail_fast"`` (default): the first instance whose retries
      exhaust raises and cancels the rest.
    - ``"collect"``: each instance runs independently; failures land in
      ``state.instance_errors`` and the batch produces partial results.
    - ``"degrade"``: each instance is additionally wrapped (outermost)
      in ``FailureIsolationMiddleware``; an instance whose retries
      exhaust is caught and returns a placeholder partial, so the batch
      completes with a degraded entry in place rather than aborting or
      dropping it.

    The smoke test calls this with no argument, exercising the default
    path; main() lets the MODE env var pick the posture.
    """
    if mode not in ("fail_fast", "collect", "degrade"):
        raise ValueError(f"mode must be one of fail_fast / collect / degrade; got {mode!r}")
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

    instance_middleware: tuple[Middleware, ...] = (retry, timing)
    error_policy = "fail_fast"
    if mode == "collect":
        error_policy = "collect"
    elif mode == "degrade":
        # Outermost instance middleware: catches the exception retry
        # re-raises once its attempts exhaust and returns a degraded
        # partial in place of the instance result, so the batch finishes
        # instead of aborting (fail_fast) or dropping the instance
        # (collect). Retry stays inner so it still sees raw transients
        # first. The degraded mapping is keyed in the subgraph's
        # field-name space (proposal 0066): the collect_field (``summary``)
        # plus each extra_outputs subgraph field (``topic``, which the
        # fan-out projects to the parent ``topics`` list). Supplying
        # ``topic`` keeps the slot non-null so the ``list[str]`` parent
        # field validates (an omitted source would be a null slot, §9.3).
        degrade = FailureIsolationMiddleware(
            degraded_update={"summary": "(unavailable)", "topic": "other"},
            event_name="headline_degraded",
        )
        instance_middleware = (degrade, retry, timing)

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
            instance_middleware=instance_middleware,
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
    # doesn't accumulate timings / isolation events across invocations.
    _timings.clear()
    _isolated.clear()

    # MODE selects the per-instance failure posture: fail_fast (default,
    # abort on the first exhausted-retry failure), collect (record
    # failures in state.instance_errors and finish the batch), or
    # degrade (FailureIsolationMiddleware catches an exhausted instance
    # and substitutes a placeholder so the batch finishes intact).
    mode = os.environ.get("MODE", "fail_fast")
    graph = build_graph(mode=mode)
    graph.attach_observer(fan_out_config_observer)
    graph.attach_observer(failure_isolation_observer)

    # collect and degrade both need a failure to demonstrate, so prepend
    # a deliberately-failing headline that summarize() always raises on.
    # collect lands it in state.instance_errors; degrade catches it and
    # substitutes a placeholder. fail_fast keeps the list clean so the
    # happy path runs to completion.
    if mode in ("collect", "degrade"):
        headlines = [
            "[FORCE_FAIL] Synthetic failing headline for the failure-path demo",
            *HEADLINES,
        ]
    else:
        headlines = list(HEADLINES)
    initial = BatchState(headlines=headlines)

    print("=" * 72)
    print(f"Summarizing {len(headlines)} headlines in parallel (concurrency=3)")
    print(f"mode={mode!r}")
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
        if _isolated:
            print(f"Failure-isolation events ({len(_isolated)}):")
            for ev in _isolated:
                print(
                    f"  event={ev.event_name!r}  cause={ev.caught_exception.category}  "
                    f"attempt_index={ev.attempt_index}"
                )
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
