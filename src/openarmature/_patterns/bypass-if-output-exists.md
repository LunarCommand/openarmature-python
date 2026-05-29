# Bypass if output exists

**Problem.** How do I skip a node whose external output already
exists?

## Approach

A small custom [middleware](https://openarmature.ai/concepts/middleware/) wraps the
node. Before calling `next_(state)`, the middleware checks "does
my output already exist?" (a filesystem file, a database row, a
content-addressable store entry). If yes, it returns the cached
output as the partial update directly. If no, it calls `next_`
and returns the result.

The node sees its normal `(state) → partial_update` contract.
The middleware is the only thing that knows about idempotency;
all callers of the node compose with it cleanly.

## Snippet

```python
import os
from collections.abc import Mapping
from typing import Any
from openarmature.graph import GraphBuilder, NextCall, State


class BypassIfRendered:
    """Skip the node if its rendered output already exists on disk."""

    def __init__(self, output_field: str, key_field: str, root: str):
        self.output_field = output_field
        self.key_field = key_field
        self.root = root

    async def __call__(
        self, state: Any, next_: NextCall
    ) -> Mapping[str, Any]:
        key = getattr(state, self.key_field)
        path = f"{self.root}/{key}.bin"
        if os.path.exists(path):
            with open(path, "rb") as f:
                return {self.output_field: f.read()}
        partial = await next_(state)
        # ... persist partial[self.output_field] to path here, or
        #     have the node itself write the file ...
        return partial


class RenderState(State):
    scene_id: str
    rendered_frame: bytes = b""


builder = (
    GraphBuilder(RenderState)
    .add_node(
        "render",
        render_frame_fn,
        middleware=[
            BypassIfRendered(
                output_field="rendered_frame",
                key_field="scene_id",
                root="./renders",
            )
        ],
    )
    # ... rest of graph ...
)
```

The middleware composes with the framework's
[four registration sites](https://openarmature.ai/concepts/middleware/): attach it
per-node (as above), per-graph, per-branch, or
per-fan-out-instance, depending on the scope of the bypass.

## When this is the right pattern

- The node's work is expensive and idempotent given the same key
  (rendering a frame, calling an external API with content-
  addressable output, downloading a file).
- The "does it exist" check is cheap (a filesystem `stat`, a
  Redis `EXISTS`, a database key lookup).
- You're OK with the node being skipped silently — the partial
  update returned by the middleware is indistinguishable from a
  successful node run.

## When it isn't

- The check itself is expensive enough that you'd rather just run
  the node. The cost model inverts; the pattern is wrong.
- You need to *force* re-execution on demand (cache invalidation).
  Add a `force_rerun: bool` field on state that the middleware
  consults — but if you're doing that often, the bypass logic
  belongs in the node itself, gated on a state field, not in
  middleware.
- The cached output's freshness depends on inputs the middleware
  can't see (downstream state, time-of-day, etc.). Use a
  dedicated caching layer instead of reimplementing cache
  invalidation in the middleware.

## Cross-references

- [Middleware](https://openarmature.ai/concepts/middleware/) — middleware shape, the
  four registration sites, composition.
- Spec: [pipeline-utilities](https://openarmature.org/capabilities/pipeline-utilities/)

This pattern is explicitly called out in proposal 0008's
*Alternatives considered* section as a userland recipe rather than
spec'd behavior — this page is its canonical home.
