# Caller-supplied trace identifiers

**Problem.** A service runs the same graph for many tenants /
requests / feature flag cohorts. How do you tag every span and
trace so downstream observability (Honeycomb, Datadog, Langfuse,
HyperDX, Grafana Tempo) can filter by tenant or join across
services without each node having to thread the identifiers
through manually?

## Approach

Pass a `metadata` dict to `invoke()`. The framework propagates each
entry to every observability backend at once: the OTel observer
emits each entry as an `openarmature.user.<key>` cross-cutting span
attribute on every span (invocation, node, subgraph wrapper,
fan-out instance, LLM provider), and the Langfuse observer merges
each entry as a top-level key into `trace.metadata` AND every
observation's metadata. Backends that consume OTel attributes pick
the entries up for free; backends with typed metadata fields get
them via per-backend propagation.

For metadata that's only known mid-flight (an ID resolved by an
LLM-classification node, a derived feature flag), use
`set_invocation_metadata` from inside a node. The augmentation
respects fan-out / parallel-branches per-instance scoping per
proposal 0045, so each instance's update lives in its own
async-context copy and doesn't leak to siblings.

## Snippet

```python
import asyncio

from openarmature.graph import END, GraphBuilder, State
from openarmature.observability import set_invocation_metadata


class RequestState(State):
    query: str = ""
    answer: str = ""


async def answer(s: RequestState) -> dict:
    # An entry resolved mid-invocation propagates to subsequent spans
    # in the same async-context: this node's `completed`, the LLM
    # provider span if any, and onwards. Sibling fan-out instances
    # and parallel-branches branches see their own copies.
    set_invocation_metadata(modelTier="standard")
    return {"answer": "Apollo 13 aborted due to an O2 tank failure."}


graph = (
    GraphBuilder(RequestState)
    .add_node("answer", answer)
    .add_edge("answer", END)
    .set_entry("answer")
    .compile()
)


async def main() -> None:
    final = await graph.invoke(
        RequestState(query="why did Apollo 13 abort?"),
        metadata={
            "tenantId": "acme-corp",
            "requestId": "req-12345",
            "featureFlag": "v2-canary",
        },
    )
    print(final.answer)


asyncio.run(main())
```

Every span emitted during this `invoke()` carries
`openarmature.user.tenantId="acme-corp"`,
`openarmature.user.requestId="req-12345"`, and
`openarmature.user.featureFlag="v2-canary"`. Spans inside the
`answer` node (and any downstream nodes if the graph had more)
additionally carry `openarmature.user.modelTier="standard"` from
the `set_invocation_metadata` call.

## Boundary validation

Validation runs synchronously, before any node body fires. Both
`invoke(metadata=...)` and `set_invocation_metadata(...)` enforce
the same rules:

- Keys MUST NOT start with `openarmature.` or `gen_ai.` (reserved
  namespaces per the spec).
- Keys MUST NOT collide with the spec's reserved per-trace metadata
  keys (`correlation_id`, `entry_node`, `spec_version`, etc.). The
  set is enforced at the `invoke()` and `set_invocation_metadata`
  boundaries via the validator in
  `openarmature.observability.metadata`; it grows per spec proposals
  0041 / 0042, with the canonical list in the spec's observability
  §3.4.
- Values MUST be OTel-attribute-compatible scalars (`str` / `int` /
  `float` / `bool`) or homogeneous arrays of those.

Violations raise `ValueError` at the boundary. Failing loud at
construction is better than the bare-key silently clobbering a
spec-reserved key in flat Langfuse `trace.metadata`.

## When this is the right pattern

- One service runs the same graph for many distinct callers
  (multi-tenant SaaS, per-customer feature flags, A/B test
  cohorts).
- Downstream observability needs to filter or join on caller-side
  identifiers (tenant ID for billing dashboards, request ID for
  cross-service trace stitching, feature flag for experiment
  analysis).
- You don't want each node to know about tenancy. The metadata
  flows through the framework, not the node bodies.

## When it isn't

- The identifier is a per-node decision, not a per-invocation one.
  If different nodes in the same invocation produce different
  values, that's typed state, not invocation metadata. Put it on
  the `State` schema with a clear reducer.
- The value isn't a scalar or homogeneous array. The boundary
  validation rejects complex shapes; if you need to attach a nested
  object, serialize it to a JSON string before passing.
- The value contains PII you don't want in every span. Metadata is
  unconditionally emitted everywhere the observers run; filter at
  the caller or skip the propagation for those keys.

## Cross-references

- [Observability concept page](../concepts/observability.md): how
  OTel attributes and Langfuse metadata propagate.
- [`examples/10-langfuse-observability`](../examples/10-langfuse-observability.md):
  runnable example exercising the metadata propagation path.
- Spec: [observability](https://openarmature.org/capabilities/observability/),
  the propagation contract for caller-supplied metadata.
