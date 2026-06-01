# Custom observer: reconciling started → completed pairs

**Problem.** A custom observer needs to thread per-call state
between a node's `started` and `completed` events: measure
duration, capture request/response payloads, attach a custom ID
that downstream uses. The engine doesn't carry a correlation field
across the pair (it doesn't need one for its own logic, since
events arrive serially per spec §6). How does the observer
reconcile which `completed` matches which `started`?

## Approach

The pair identity is the tuple
`(namespace, branch_name, attempt_index, fan_out_index)`. That
tuple is unique within an invocation: the namespace separates
subgraph wrappers from their parents, `branch_name` distinguishes
parallel-branches branches, `attempt_index` distinguishes retried
attempts of the same node, and `fan_out_index` distinguishes
per-instance fan-out copies. Carry per-invocation state in
`dict[invocation_id, dict[tuple, value]]`, look up on `completed`,
and sweep the outer entry when the per-invocation sub-dict
empties.

Both `branch_name` and `fan_out_index` matter even for nodes that
look "the same" by name: a node `score` inside parallel-branches
`branch=fast` vs `branch=slow` produces two distinct pair
identities, and a per-instance fan-out copy at `fan_out_index=3` is
not the same as `fan_out_index=4`.

## Snippet

```python
import time
from typing import NamedTuple

from openarmature.graph import NodeEvent
from openarmature.observability.correlation import current_invocation_id


PairKey = tuple[tuple[str, ...], str | None, int, int | None]


class StepTiming(NamedTuple):
    node_name: str
    namespace: tuple[str, ...]
    branch_name: str | None
    attempt_index: int
    fan_out_index: int | None
    duration_s: float


class StepTimingObserver:
    """Custom observer that records wall-clock duration per node
    attempt. Stitches started -> completed via the per-invocation
    pair-identity dict.
    """

    def __init__(self) -> None:
        # invocation_id -> {pair_key: start_monotonic}
        self._pending: dict[str, dict[PairKey, float]] = {}
        # Final per-call timings, surfaced to whatever consumes them
        # (metrics exporter, log line, in-test assertion).
        self.timings: list[StepTiming] = []

    async def __call__(self, event: NodeEvent) -> None:
        invocation_id = current_invocation_id()
        if invocation_id is None:
            return

        key: PairKey = (
            event.namespace,
            event.branch_name,
            event.attempt_index,
            event.fan_out_index,
        )

        if event.phase == "started":
            self._pending.setdefault(invocation_id, {})[key] = time.monotonic()
            return

        if event.phase == "completed":
            start = self._pending.get(invocation_id, {}).pop(key, None)
            if start is not None:
                self.timings.append(
                    StepTiming(
                        node_name=event.node_name,
                        namespace=event.namespace,
                        branch_name=event.branch_name,
                        attempt_index=event.attempt_index,
                        fan_out_index=event.fan_out_index,
                        duration_s=time.monotonic() - start,
                    )
                )
            # Sweep when the dict empties for this invocation.
            if not self._pending.get(invocation_id):
                self._pending.pop(invocation_id, None)
```

Attach with `graph.attach_observer(StepTimingObserver())`. Run
the invocation; the observer's `timings` list carries one entry
per node attempt with its duration and identifying tuple.

## When this is the right pattern

- A custom observer needs paired-event state that the spec doesn't
  carry across the pair.
- The pair identity needs to be unique across fan-out instances or
  parallel-branches branches; a key shape that omits `branch_name`
  or `fan_out_index` would collide.
- Long-running services need the dict to drain naturally as
  invocations complete. The "sweep when sub-dict empties" pattern
  prevents the outer dict from growing per-invocation forever.

## When it isn't

- You only need a final-summary signal at invocation completion.
  Subscribe to the invocation `completed` event and read the final
  state directly; no per-call reconciliation needed.
- The `OTelObserver` or `LangfuseObserver` already provides what
  you want. Both stitch `started` / `completed` internally to open
  / close spans; you don't need a custom observer to track timings
  if a span carries the duration already.
- The metric is cross-invocation. A pair-identity dict scoped to a
  single invocation_id won't aggregate; use a global counter or
  push to an external metrics backend instead.

## Cross-references

- [Observability concept page](../concepts/observability.md): the
  `NodeEvent` shape, `started` / `completed` lifecycle.
- [Caller-supplied trace identifiers](caller-supplied-trace-identifiers.md):
  adjacent pattern for tagging the events your observer sees.
- Spec: [graph-engine](https://openarmature.org/capabilities/graph-engine/),
  observer events and the uniqueness invariants for
  `(namespace, branch_name, attempt_index, fan_out_index)`.
