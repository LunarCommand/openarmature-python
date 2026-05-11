# Changelog

All notable changes to `openarmature-python` are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package follows [Semantic Versioning](https://semver.org/); pre-1.0 minor bumps may carry behavioral changes per [spec governance](https://github.com/LunarCommand/openarmature-spec/blob/main/GOVERNANCE.md).

## [Unreleased]

## [0.5.0] — 2026-05-10

First release on real PyPI. Catches the implementation up from spec v0.5.x to v0.10.0 across six phases — the spec accepted eight proposals while the python lib was at v0.3.1, and v0.5.0 lands all of them in one curated drop.

### Added

- **Typed conformance harness (Phase 0).** Single parametrised test target driving all 68 spec fixtures under discriminated-union YAML parsers. Replaces the earlier hand-rolled per-fixture wiring.
- **Observer pair model (Phase 1, spec v0.6.0 / proposal 0005 §6).** `Observer` Protocol (async callable), `SubscribedObserver` with phase subscription set (`{"started", "completed", "checkpoint_saved"}`), `RemoveHandle.remove()`, and a serial delivery queue per spec §6 ordering. Observer exceptions don't propagate; reported via `warnings.warn`.
- **Middleware (Phase 2, proposal 0004).** `Middleware` Protocol with the canonical `(state, next) → partial_update` shape, `compose_chain` runtime, and five stdlib middlewares: `RetryMiddleware`, `TimingMiddleware`, `ErrorRecoveryMiddleware`, `ShortCircuitMiddleware`, `TraceRecorderMiddleware`. Per-graph and per-node middleware composition.
- **Fan-out runtime (Phase 3, proposal 0005 pipeline-utilities side).** `FanOutNode` for parallel fan-out over an `items_field` or a `count` (int or callable resolver). Configurable concurrency, error policy (`fail_fast` / `collect`), `inputs` / `extra_outputs` projection, optional `errors_field` collection. Composes with retry middleware on the fan-out node and on per-instance subgraphs.
- **LLM provider (Phase 4, proposal 0006).** New `openarmature.llm` package: `Provider` Protocol with `ready()` / `complete(messages, tools=None, config=None)`; `OpenAIProvider` (HTTPX-based, OpenAI-compatible wire); typed `Message` / `ToolCall` / `Tool` / `Response` / `RuntimeConfig`; seven error categories (`ProviderAuthentication`, `ProviderUnavailable`, `ProviderInvalidRequest`, `ProviderInvalidResponse`, `ProviderInvalidModel`, `ProviderModelNotLoaded`, `ProviderRateLimit` with `retry_after`). Tool-call ids preserved verbatim through the wire.
- **Checkpointing (Phase 5, proposal 0008).** `Checkpointer` Protocol (`save` / `load` / `list` / `delete`) with `CheckpointRecord` and `NodePosition` shapes; `InMemoryCheckpointer` reference impl; `CheckpointNotFound` / `CheckpointRecordInvalid` / `CheckpointSaveFailed` error categories; `checkpoint_saved` observer phase; resume-from-checkpoint semantics for fan-out and subgraph compositions.
- **Observability / OTel (Phase 6, proposal 0007).** `OTelObserver` mapping observer events → OpenTelemetry spans with private `TracerProvider` (no global pollution); §4.4 detached subgraph + detached fan-out trace mode; §5.5 LLM-provider span emission with `disable_llm_spans` opt-out; §5.6 cross-cutting `openarmature.correlation_id` on every span; §10.8 `checkpoint_saved` zero-duration span. `install_log_bridge` wires the stdlib root logger through OTel's Logs Bridge (deprecation-aware via `opentelemetry-instrumentation-logging`) so log records emitted within an invocation carry the active span's `trace_id`/`span_id` plus `openarmature.correlation_id`. `prepare_sync` synchronous observer hook so logs emitted on the FIRST line of a node body (before any `await`) pick up the right span. Fan-out per-instance dispatch span synthesis (§5.4) with `parent_node_name` cached and applied per-instance.
- **`current_correlation_id()` public API.** Read the per-invocation cross-backend join key from anywhere within the invocation's async call tree.
- **Subgraph configuration plural form.** Builder accepts `subgraphs:` alongside `subgraph:` for fixture compatibility.

### Changed

- **Pinned spec version: 0.5.x → 0.10.0.** Lands proposals 0004 (middleware), 0005 (fan-out + observer pair model), 0006 (llm-provider), 0007 (observability/OTel), 0008 (checkpointing), 0011 (`prepare_sync` hook), 0012 (`completed` event after edge eval), 0013 (`fan_out_config` on `NodeEvent`).
- **Edge-resolution failures share the preceding node's event pair (spec v0.9.0 / proposal 0012).** `routing_error` and `edge_exception` populate `error` on the preceding node's `completed` event with `post_state=None` instead of producing a separate pair. All five §4 runtime error categories now land via the same uniform mechanism.
- **Observer protocol contract.** Async-only callable; phase-filtered delivery via `SubscribedObserver.phases`; serial single-task delivery worker; observer errors isolated via `warnings.warn`.

### Fixed

- **Log bridge filter placement.** Phase 6.0's `_CorrelationIdFilter` lived on the root logger; Python's logging propagation walks ancestor handlers but **not** ancestor filters, so child-logger records (the normal `logging.getLogger("module")` pattern) were missed. Replaced with a process-global `LogRecord` factory that fires uniformly at record construction.
- **OTelObserver concurrency-safe state scoping.** Per-invocation span state now keyed by `invocation_id` so concurrent invocations sharing one observer instance don't collide on the in-flight span maps.
- **Spec submodule pin sync.** Internal `spec_version` matched the submodule HEAD across phase boundaries; tracked via `tests/test_smoke.py`.

### Notes

- **First real PyPI publish.** Pre-release verification continues to flow through TestPyPI per `docs/RELEASING.md`. The `pypi` GitHub Environment requires a manual approval click before any real-PyPI upload — keep it on.
- **Pre-1.0 SemVer.** Behavioral changes may land in MINOR bumps. Several Phase 1+ contracts changed shape vs. v0.4.0 — most user-visible: the observer pair model in Phase 1, the edge-resolution failure mechanism in Phase 6.1.
- **Cross-language posture.** This release tracks spec v0.10.0; the OpenArmature TypeScript implementation will land separately under the same spec.
