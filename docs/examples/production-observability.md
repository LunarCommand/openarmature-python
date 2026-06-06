# Production observability with dual observers, timing middleware, and per-invocation cost rollup

!!! info "Source"
    [https://github.com/LunarCommand/openarmature-python/blob/main/examples/production-observability/main.py](https://github.com/LunarCommand/openarmature-python/blob/main/examples/production-observability/main.py){target="_blank" rel="noopener"}

A single-turn lunar-mission Q&A endpoint instrumented the way you'd
ship it: BOTH OTel and Langfuse observers attached to the same
graph, caller hooks deriving domain-shaped `trace.input` /
`trace.output` from State, the built-in `TimingMiddleware`
recording per-node duration, multi-tenant caller-supplied
metadata propagating to both observers in one `invoke()` call, AND
a third queryable-accumulator observer that a terminal `persist`
node reads at request scope after synchronizing on the deliver
loop with `drain_events_for`.

## Overview

Two nodes (`respond` then `persist`), one LLM call, three observers
attached at compile time. The pipeline takes a question, calls the
LLM, returns the answer, then synchronizes on the observer queue
and rolls up token cost. The interesting part is the observability
wiring:

- `OTelObserver` attached with an `InMemorySpanExporter`
  (production swaps this for `BatchSpanProcessor` +
  `OTLPSpanExporter` pointed at HyperDX / Honeycomb / Tempo / any
  OTLP backend).
- `LangfuseObserver` attached with an `InMemoryLangfuseClient`
  (production swaps for `LangfuseSDKAdapter(Langfuse(...))`).
- Both observers consume the same `NodeEvent` stream
  independently; node code never knows there are two backends.
- `LangfuseObserver` carries `trace_input_from_state` and
  `trace_output_from_state` caller hooks that derive domain dicts
  like `{"question": ...}` / `{"answer": ..., "model": ...}` from
  State, instead of letting the observer dump the raw State
  object.
- `TimingMiddleware` (canonical, from
  `openarmature.graph.middleware`) wraps the respond node. An
  `on_complete` async callback receives a `TimingRecord` and
  prints a one-line timing summary; production would queue to a
  metrics backend (StatsD, Prometheus pushgateway, OTLP metrics).
- `invoke(metadata={...})` carries `tenantId`, `requestId`, and
  `featureFlag` from the call site. Both observers pick them up:
  OTel attaches them as `openarmature.user.*` span attributes,
  Langfuse merges them as top-level `trace.metadata` keys.

At the end the demo prints what each backend captured so a reader
sees the same logical events represented two ways.

## What it teaches

- **Two observers on one graph**
  ([no double-export between them](../concepts/observability.md)).
  Each consumes the `NodeEvent` stream independently; the engine
  fans events out to all attached observers. Production deployments
  often run both: OTel for infrastructure-side correlation
  (logs, distributed tracing across services), Langfuse for
  LLM-aware generation rendering.
- **Caller hooks for `trace.input` / `trace.output`**
  ([deriving domain dicts from State](../concepts/observability.md)).
  Without the hooks the Langfuse observer either omits the field
  (`disable_state_payload=True` default) or dumps the raw State
  (when `disable_state_payload=False`). The hooks let you return a
  domain dict shaped for the Langfuse UI viewer while keeping PII
  the operator hasn't audited out of trace payloads.
- **`TimingMiddleware`**
  ([reference](../reference/graph.md)).
  Wraps a node's execution and dispatches a
  `TimingRecord(node_name, duration_ms, outcome, exception_category)`
  to an async callback when the chain returns or raises. The
  callback fires inline before the chain's result reaches the
  engine; keep it fast (queue work, defer I/O).
- **`invoke(metadata={...})` propagation across observers**
  ([caller metadata and reserved keys](../concepts/observability.md)).
  One call site, both backends pick it up: OTel attaches each entry
  as `openarmature.user.<key>` cross-cutting span attribute,
  Langfuse merges as top-level `trace.metadata` keys plus
  per-observation metadata.
- **In-memory captures for both backends**
  ([reference](../reference/observability.md)).
  `InMemoryLangfuseClient` records every Trace / Observation;
  `InMemorySpanExporter` records every Span. Production
  deployments swap each for a real exporter / SDK adapter; the
  observer call surface doesn't change.
- **Queryable accumulator + `drain_events_for`**
  ([queryable observer pattern](../concepts/observability.md)).
  A third observer — `LlmUsageAccumulator` — subscribes to the
  same event stream but only records the LLM-namespace events
  carrying an `LlmEventPayload`. It accumulates per-invocation
  token totals in memory, indexed by `current_invocation_id()`.
  The terminal `persist` node calls
  `await graph.drain_events_for(state.invocation_id, timeout=2.0)`
  to synchronize on the deliver loop, then reads the accumulator's
  bucket and drops it. Without the drain, the bucket might be
  missing the most-recent LLM event's tokens (the deliver loop
  hasn't reached them yet). The `Observer` protocol itself stays
  a single-callable shape; the accumulator just exposes its own
  read methods (`get_bucket` / `drop`) that the persist node knows
  about. This is the canonical shape for per-invocation cost
  attribution at request scope, replacing the round-trip-through-
  State workarounds that pre-v0.12.0 deployments used.

## How to run

```bash
uv sync --group examples --all-extras
LLM_API_KEY=sk-... uv run python examples/production-observability/main.py
```

`LLM_MODEL` defaults to `gpt-4o-mini`. The pipeline is single-turn
and doesn't need vision capability.

The demo prints in three blocks: a header (the question and the
caller-supplied tenant/request/feature-flag), the LLM answer, then
two captured-trace summaries (OTel spans + Langfuse Trace tree).

## Reading the output

Numbers shown below (durations, token counts, UUIDs) are illustrative
and vary per run; the shape is what matters.

```
=== openarmature production-observability demo ===
question:    What was the primary objective of Apollo 11?
tenant id:   demo-acme
request id:  <uuid>
feature flag:v2-canary

[timing] respond: 1234.5ms (success)
[persist] LLM usage: prompt=42, completion=38, total=80 across 1 call(s)
answer:      The primary objective of Apollo 11 was ...
model:       gpt-4o-mini-2024-07-18

--- captured OTel spans ---
  [openarmature.invocation] 1240.0ms  openarmature.user.tenantId='demo-acme', ...
  [respond] 1235.0ms  openarmature.node.name='respond', openarmature.user.tenantId='demo-acme', ...
  [openarmature.llm.complete] 1200.0ms  gen_ai.system='openai', gen_ai.usage.input_tokens=42, ...

--- captured Langfuse trace ---
Trace id=<uuid>
      name='respond'
      input={'question': 'What was the primary objective of Apollo 11?'}
      output={'answer': '...', 'model': 'gpt-4o-mini-2024-07-18'}
      metadata={'tenantId': 'demo-acme', 'requestId': '<uuid>', 'featureFlag': 'v2-canary', ...}
  [span] 'respond'
      input={'question': '...'}
      output={'answer': '...', 'model': '...'}
    [generation] 'openarmature.llm.complete'
      input=[{'role': 'system', ...}, {'role': 'user', ...}]
      output='The primary objective of Apollo 11 ...'
      model='gpt-4o-mini-2024-07-18'
      usage={'input_tokens': 42, 'output_tokens': 38}
```

- **`[timing] respond: 1234.5ms (success)`**: emitted by the
  `TimingMiddleware` callback as soon as the respond chain returns.
  `outcome` is `"success"` here; a `ProviderRateLimit` would surface
  as `outcome="exception"` with `exception_category="provider_rate_limit"`.
- **`[persist] LLM usage: ...`**: emitted by the `persist` node
  after it drains the deliver loop and reads the
  `LlmUsageAccumulator`'s bucket for this invocation. If the drain
  times out (slow / hung observer), the persist line is prefixed by
  a `[persist] drain incomplete: N events still pending after 2.0s`
  surface — the production version of that log would also flip an
  SLO-breach metric.
- **OTel spans block**: one line per captured span, sorted by
  start time. The relevant attributes shown are a curated subset
  for readability; the full attribute set is on each `Span` object
  for any reader inspecting them programmatically. Note the
  `openarmature.user.*` attributes appearing on every span (the
  cross-cutting attribute propagation from `invoke(metadata=...)`).
- **Langfuse trace block**: the same invocation as seen by the
  Langfuse data model. `trace.input` / `trace.output` come from the
  caller hooks (`{"question": ...}` / `{"answer": ..., "model": ...}`)
  rather than the raw State. The Observation tree shows
  `[span]` for the node and `[generation]` for the LLM call;
  production Langfuse renders these as nested cards in the UI.
- **Identical `correlation_id`** (not shown by the formatter but
  present in both captures' metadata): the cross-system join key.
  Find a slow Generation in Langfuse, grep for the
  `correlation_id` in OTel logs, see the surrounding infrastructure
  activity.

## Swapping to production backends

```python
# OTel: real OTLP exporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

otel_observer = OTelObserver(
    span_processor=BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="https://your-collector/v1/traces",
            headers={"authorization": os.environ["OTLP_AUTH"]},
        )
    ),
    resource=Resource.create({"service.name": "lunar-briefing"}),
)

# Langfuse: real SDK adapter
from langfuse import Langfuse
from openarmature.observability.langfuse import LangfuseSDKAdapter

langfuse_observer = LangfuseObserver(
    client=LangfuseSDKAdapter(
        Langfuse(
            public_key="pk-lf-...",
            secret_key="sk-lf-...",
            host="https://cloud.langfuse.com",
        )
    ),
    trace_input_from_state=_trace_input,
    trace_output_from_state=_trace_output,
    disable_llm_payload=False,
)
```

Same observer call surface, real exporters underneath. Node and
graph code don't change. The observer-hooks example shows the
OTel-only side at finer granularity (`force_flush`, log bridging,
error handling); the langfuse-observability example shows the
Langfuse + `LangfusePromptBackend` prompt-linkage side.
