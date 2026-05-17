# Concepts

Each page is a focused take on one idea. Read top-to-bottom for a tour of
the framework, or jump to whichever concept you need.

- [State and reducers](state-and-reducers.md): typed state, per-field
  merge policies, what makes nodes safe to write.
- [Graphs: nodes, edges, build, invoke](graphs.md): the four moves you
  make to turn a state schema into a runnable pipeline.
- [Composition: conditional edges, subgraphs, projection](composition.md):
  routing decisions, encapsulated sub-pipelines, the parent ↔ subgraph
  data seam.
- [Fan-out](fan-out.md): running the same subgraph many times in
  parallel, results merged back deterministically.
- [Parallel branches](parallel-branches.md): dispatching M
  heterogeneous subgraphs concurrently with per-branch state schemas
  and middleware.
- [LLMs](llms.md): how LLM calls fit into nodes, structured output,
  routing on parsed fields, errors at the LLM boundary.
- [Observability](observability.md): node-boundary hooks, OTel mapping,
  log correlation.
- [Checkpointing](checkpointing.md): save state at each node boundary,
  resume from a prior point.

If you're brand-new, [Quickstart](../getting-started/index.md) is the
faster entry; under a minute to a running graph. Come back here when
you want to know *why* things are shaped the way they are.
