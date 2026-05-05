# `harness/runtime/`

Phase 0 (typed parser) lives in the parent `harness/` package; this directory
is the home for the **runtime** — the code that takes a parsed fixture and
actually executes it against the engine. Implementations land here in
Phases 1–6, one capability or directive at a time.

Phase 0 deliberately ships an empty `runtime/` to lock in the boundary:
parsing is fixed (every fixture lands as a typed config validated once), and
phases that follow only add interpretation, never re-touch the parsing.
