# Fan-out with retry

!!! info "Source"
    [https://github.com/LunarCommand/openarmature-python/blob/main/examples/fan-out-with-retry/main.py](https://github.com/LunarCommand/openarmature-python/blob/main/examples/fan-out-with-retry/main.py){target="_blank" rel="noopener"}

Summarize a batch of lunar-mission headlines in parallel, with
per-headline retry and timing middleware wrapping each instance's
subgraph run.

## Overview

You have a list of news headlines. Each one needs a one-sentence
summary plus a topic tag. The headlines are independent, so the
work parallelizes naturally: dispatch one per-headline subgraph
run per headline, bounded concurrency, retry transient LLM failures
on a per-instance basis.

The per-instance subgraph is small (`summarize → classify`) and
would also run standalone against a single headline. Fan-out
multiplies it out across the batch.

The `MODE` env var selects the per-instance failure posture. The
default `fail_fast` aborts the batch on the first instance whose
retries exhaust. `collect` and `degrade` both prepend a sentinel
headline that always raises `ProviderUnavailable`, then handle it
differently: `collect` lands the failure in `state.instance_errors`
and finishes the rest of the batch, while `degrade` wraps each
instance in `FailureIsolationMiddleware` so an exhausted instance is
caught and replaced with a placeholder summary, leaving the batch
intact.

## What it teaches

- [`add_fan_out_node`](../concepts/fan-out.md) in `items_field`
  mode: one subgraph invocation per element of `state.headlines`.
  `item_field` names the per-instance input field on the subgraph's
  state.
- `collect_field` and `extra_outputs` for harvesting per-instance
  results into parent lists. The two lists (`summaries`, `topics`)
  end up index-aligned.
- `instance_middleware`: middleware wrapped around each instance's
  subgraph run. `RetryMiddleware` (3 attempts, deterministic
  backoff) plus `TimingMiddleware` (captures duration per
  instance). Retries are per-instance: a transient failure on
  headline 3 doesn't restart 0-2.
- `concurrency=3` capping how many instances run in flight at once.
- `error_policy="fail_fast"` (default, first exhausted-retry
  failure aborts the batch) vs `"collect"` (failures land in
  `errors_field` and the batch produces partial results). The
  `degrade` mode keeps `fail_fast` but adds
  `FailureIsolationMiddleware` as the outermost instance middleware,
  so an exhausted instance is caught and degraded to a placeholder
  before the fan-out ever sees the failure.
- A `fan_out_config_observer` reads
  `NodeEvent.fan_out_config` on the fan-out node's dispatch event,
  recording the resolved `item_count` / `concurrency` /
  `error_policy` at runtime. Inner-instance events carry
  `fan_out_index` but not the config.
- In `degrade` mode, a `failure_isolation_observer` captures each
  `FailureIsolatedEvent` and the demo prints its
  `caught_exception.category`. At a fan-out instance placement the
  category resolves to the originating cause (`provider_unavailable`)
  rather than the masking `node_exception`, so the telemetry names what
  actually failed.

## Composing with checkpointing

This example doesn't register a `Checkpointer`, but the fan-out
pattern composes cleanly with checkpoint resume. When a fan-out
runs under a registered backend, the resume contract is
**per-instance**: instances that completed in the prior run skip
re-execution and their contributions roll forward through the
fan-in step; instances that were `in_flight` at save time re-run
from the subgraph's entry node; not-started instances dispatch
normally. The `append` reducer's no-double-merge guarantee holds
across resume because `completed` is a one-shot accumulator state.

Composition with `instance_middleware` (retry): on resume, an
instance's `attempt_index` resets to 0 (a fresh retry budget). So
a retry-exhausted instance whose `in_flight` state was saved gets
a fresh budget on the resumed run.

See [Resume semantics in fan-out](../concepts/fan-out.md#resume-semantics)
and the [Checkpointing concept page](../concepts/checkpointing.md)
for the full contract.

## How to run

```bash
uv sync --group examples
LLM_API_KEY=sk-... uv run python examples/fan-out-with-retry/main.py
```

To exercise a failure posture with a synthetic failure:

```bash
# record the failure and finish the batch
MODE=collect LLM_API_KEY=sk-... \
  uv run python examples/fan-out-with-retry/main.py

# degrade the failed instance to a placeholder and finish the batch
MODE=degrade LLM_API_KEY=sk-... \
  uv run python examples/fan-out-with-retry/main.py
```

## The graph

```mermaid
flowchart TD
  start([start])
  announce[announce]
  present[present]
  stop([end])

  subgraph headline_runs [headline_runs: fan-out, concurrency=3]
    direction TB
    note["N instances of:<br/>summarize -> classify<br/>(retry + timing middleware)"]
  end

  start --> announce --> headline_runs --> present --> stop
```

`headline_runs` is the fan-out node. At dispatch time it expands
into N copies of the per-instance subgraph, one per headline.
`RetryMiddleware` and `TimingMiddleware` wrap each instance (plus
`FailureIsolationMiddleware` as the outermost layer in `degrade`
mode).

## Reading the output

A clean default-mode run (`fail_fast`, all instances succeed):

```
========================================================================
Summarizing 5 headlines in parallel (concurrency=3)
mode='fail_fast'
========================================================================

  [observer] fan-out node 'headline_runs' dispatching: item_count=5 concurrency=3 error_policy='fail_fast'

Results (in input order):

  [0] Artemis II splashes down in Pacific after ten-day lunar flyby
       summary: <one-sentence rewrite>
       topic:   crew

  [1] NASA pauses Lunar Gateway program in favor of crewed surface base
       summary: <one-sentence rewrite>
       topic:   policy

  ...

Per-instance timings (in completion order):
  #0   812.3 ms  outcome=success
  #1   941.7 ms  outcome=success
  #2   876.2 ms  outcome=success
  #3   903.4 ms  outcome=success
  #4   1012.8 ms  outcome=success

  wall-clock total:         2089.3 ms
  sum of per-instance:      4546.4 ms
  → concurrency speedup:    2.18x
```

- **The observer line** is the `fan_out_config_observer` printing
  the dispatch-time config. Useful when `count` or `concurrency`
  are callable resolvers whose runtime value isn't visible in code.
- **Per-input order vs completion order.** The result loop walks
  `final.headlines` in input order; `final.summaries` and
  `final.topics` are index-aligned with it. The timings list is in
  completion order, not input order (instance 2 may finish before
  instance 1 under concurrency).
- **Concurrency speedup.** `sum of per-instance / wall-clock`. A
  speedup near `concurrency` indicates the work parallelized well;
  a value near 1.0 indicates concurrency didn't help (the upstream
  serialized you, or instances themselves are short).

With `MODE=collect`, the output includes the sentinel headline at
index 0 with a `(failed after retries; ...)` marker, plus a
`Captured 1 per-instance error(s):` block listing the failed
`fan_out_index` and error category. The other instances complete as
usual.

With `MODE=degrade`, the sentinel at index 0 instead shows a
placeholder result (`summary: (unavailable)`, `topic: other`) and
there is no error block: `FailureIsolationMiddleware` caught the
exhausted-retry failure and returned the degraded partial, so the
fan-out recorded the instance as a (degraded) success. The
per-instance timings still show the sentinel's failed attempts, so
you can see the retries happened before the instance was degraded.

The degrade run also prints a `Failure-isolation events` block from the
`failure_isolation_observer`:

```
Failure-isolation events (1):
  event='headline_degraded'  cause=provider_unavailable  attempt_index=2
```

`cause` is the resolved originating category, `provider_unavailable`,
not the masking `node_exception` the engine wraps the failure in before
isolation catches it. `attempt_index` is the final, exhausting attempt
of the three the retry middleware made.
