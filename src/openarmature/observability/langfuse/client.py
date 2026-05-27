# Spec mapping (observability §8):
# - Captures the minimal Langfuse SDK-shaped surface the LangfuseObserver
#   calls — `trace(...)` returning a handle with `.span(...)` /
#   `.generation(...)` / `.update(...)` / `.end()`.
# - Protocol-typed so the observer is decoupled from any concrete SDK
#   version; users plug in `langfuse.Langfuse()` directly (or a thin
#   adapter) for production, and the bundled `InMemoryLangfuseClient`
#   for tests / conformance fixtures.
# - All record types are plain dataclasses so the in-memory client's
#   captured data is trivially inspectable from test code.

"""Langfuse client Protocol + in-memory recorder.

The :class:`LangfuseObserver` consumes the §6 OA event stream and
emits Langfuse Trace + Observation entities through a
:class:`LangfuseClient`. The Protocol is intentionally narrow: it
declares only the methods the observer calls. Concrete sinks:

- :class:`InMemoryLangfuseClient` — captures everything in dataclass
  records. Used by the conformance harness; useful for unit tests.
- A real :class:`langfuse.Langfuse` instance — Protocol-compatible
  given the SDK's current shape. Pass it directly to the observer
  in production code.

Future PRs MAY ship a thin adapter for SDK versions whose shape
diverges from the Protocol; for now the in-memory client is the
reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

ObservationType = Literal["span", "generation", "event"]

# Langfuse-supported `level` values per spec §8.4.2 (statusMessage pair).
ObservationLevel = Literal["DEFAULT", "DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass
class LangfuseUsage:
    """Langfuse Generation `usage` record. Field names match Langfuse SDK."""

    input: int | None = None
    output: int | None = None
    total: int | None = None


@dataclass
class LangfuseObservation:
    """A single Langfuse Observation captured by an in-memory client.

    Carries the observation's type-discriminated shape — Spans hold
    timing + metadata; Generations add model/parameters/usage/input/
    output/prompt-entity link; Events are point-in-time markers
    (reserved per spec §8.2 — not used by this version of the mapping).
    """

    id: str
    type: ObservationType
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    parent_observation_id: str | None = None
    level: ObservationLevel = "DEFAULT"
    status_message: str | None = None
    ended: bool = False

    # Generation-specific (None / empty on Span and Event observations)
    model: str | None = None
    model_parameters: dict[str, Any] = field(default_factory=dict[str, Any])
    input: Any = None
    output: Any = None
    usage: LangfuseUsage | None = None
    # Opaque reference set when §8.4.4 case 1 triggers — equals
    # ``Prompt.observability_entities["langfuse_prompt"]`` from the
    # prompt-management capability (proposal 0033). Production
    # adapters surface this as a real Langfuse SDK Prompt link; the
    # in-memory client just records the value verbatim for inspection.
    prompt_entity_link: Any = None


@dataclass
class LangfuseTrace:
    """A single Langfuse Trace captured by an in-memory client.

    The Trace owns its Observation tree. Observations carry their own
    `parent_observation_id`; callers MAY walk to render a tree view.
    """

    id: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict[str, Any])
    observations: list[LangfuseObservation] = field(default_factory=list[LangfuseObservation])

    def find_observation(self, observation_id: str) -> LangfuseObservation | None:
        for obs in self.observations:
            if obs.id == observation_id:
                return obs
        return None

    def children_of(self, parent_id: str | None) -> list[LangfuseObservation]:
        return [o for o in self.observations if o.parent_observation_id == parent_id]


# Handle Protocols ---------------------------------------------------------
# The Langfuse SDK exposes stateful handles (StatefulSpanClient /
# StatefulGenerationClient) returned from create-calls. The observer
# pins these to update / end observations as the corresponding OA span
# closes. The Protocols below declare only the methods the observer
# touches; SDK clients satisfy them structurally.


@runtime_checkable
class LangfuseSpanHandle(Protocol):
    """In-flight Span observation handle returned by `client.span(...)`."""

    @property
    def id(self) -> str: ...

    def update(self, **fields: Any) -> None: ...

    def end(self, **fields: Any) -> None: ...


@runtime_checkable
class LangfuseGenerationHandle(Protocol):
    """In-flight Generation observation handle returned by
    `client.generation(...)`."""

    @property
    def id(self) -> str: ...

    def update(self, **fields: Any) -> None: ...

    def end(self, **fields: Any) -> None: ...


@runtime_checkable
class LangfuseClient(Protocol):
    """Minimal client surface the LangfuseObserver requires.

    Method shape mirrors the Langfuse Python SDK's low-level API:
    `client.trace(...)` creates a Trace; `client.span(trace_id=..., ...)`
    opens a Span observation; `client.generation(trace_id=..., ...)`
    opens a Generation observation. Each returns a stateful handle
    the observer keeps and `.end()`s when the corresponding OA span
    closes.

    The Protocol does NOT define `event(...)` — Event observations
    are reserved by §8.2 but not used in v0.23.0 of the mapping.
    """

    def trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create a new Trace.

        The Trace `id` MUST be the OA invocation_id verbatim (§8.4.1).
        Implementations track Traces internally; observation calls
        pass `trace_id` to associate.
        """
        ...

    # The current observer doesn't invoke this method — it sets the
    # Trace's full metadata + entry-node-name fallback at creation.
    # The Protocol still declares it because the caller-supplied
    # invocation-label path (proposal 0034, PR 4) may need to swap
    # the trace name AFTER the first node event opens the Trace; in
    # that case the observer calls update_trace mid-invocation to
    # apply the label. SDK adapters implement this for forward
    # compatibility with PR 4's wiring.
    def update_trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Update an existing Trace's mutable fields after creation.

        Used by the observer when the caller-supplied invocation
        label (§8.6) lands later than the Trace's open call, or when
        additional metadata becomes available mid-invocation.
        """
        ...

    def span(
        self,
        *,
        trace_id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_observation_id: str | None = None,
        level: ObservationLevel = "DEFAULT",
        status_message: str | None = None,
    ) -> LangfuseSpanHandle: ...

    def generation(
        self,
        *,
        trace_id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_observation_id: str | None = None,
        level: ObservationLevel = "DEFAULT",
        status_message: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any = None,
        output: Any = None,
        usage: LangfuseUsage | None = None,
        prompt: Any = None,
    ) -> LangfuseGenerationHandle: ...

    def force_flush(self, timeout_ms: int = 30_000) -> bool:
        """Flush any pending outbound buffer in the underlying sink.

        Returns ``True`` when the flush completes within the deadline,
        ``False`` otherwise. The semantics mirror OTel's
        ``TracerProvider.force_flush``: cover the export-buffer half
        of fast-teardown races. The bundled
        :class:`InMemoryLangfuseClient` has no buffer and returns
        ``True`` immediately; SDK adapters delegate to the underlying
        client's flush.

        **Deadline honor is best-effort.** Adapters wrapping SDKs
        that don't expose a timeout-propagation surface (the v4
        Langfuse SDK is one such case — its ``flush()`` blocks on the
        SDK's own internal defaults) may ignore ``timeout_ms`` and
        return ``True`` once the underlying call returns. Callers
        with a hard deadline should layer their own wall-clock guard
        around this method rather than relying solely on the return
        value.
        """
        ...


# Concrete in-memory implementation ---------------------------------------
# Used by tests and the conformance harness. Stores everything the
# observer pushes verbatim so assertions can inspect the captured
# shape directly.


@dataclass
class _InMemorySpanHandle:
    """Stateful handle pinned by an InMemoryLangfuseClient."""

    observation: LangfuseObservation

    @property
    def id(self) -> str:
        return self.observation.id

    def update(self, **fields: Any) -> None:
        _apply_fields(self.observation, fields)

    def end(self, **fields: Any) -> None:
        _apply_fields(self.observation, fields)
        self.observation.ended = True


@dataclass
class _InMemoryGenerationHandle:
    """Stateful handle pinned by an InMemoryLangfuseClient."""

    observation: LangfuseObservation

    @property
    def id(self) -> str:
        return self.observation.id

    def update(self, **fields: Any) -> None:
        _apply_fields(self.observation, fields)

    def end(self, **fields: Any) -> None:
        _apply_fields(self.observation, fields)
        self.observation.ended = True


def _apply_fields(observation: LangfuseObservation, fields: dict[str, Any]) -> None:
    # Merge SDK-style kwargs into the captured observation. Maps the
    # SDK's argument names onto LangfuseObservation's attribute names
    # (e.g. `model_parameters` -> `model_parameters`; `usage` stays as
    # `usage`). Unknown kwargs are stored on `metadata` so the in-memory
    # client doesn't silently drop SDK extensions.
    direct = {
        "name",
        "level",
        "status_message",
        "model",
        "model_parameters",
        "input",
        "output",
        "usage",
        "prompt_entity_link",
    }
    metadata_update: dict[str, Any] = {}
    for key, value in fields.items():
        if key in direct:
            setattr(observation, key, value)
        elif key == "metadata":
            if value is not None:
                observation.metadata.update(value)
        elif key == "prompt":
            # The SDK accepts a `prompt=<Prompt entity>` kwarg on
            # `generation(...)`. Mirror that into our explicit
            # ``prompt_entity_link`` slot.
            observation.prompt_entity_link = value
        else:
            metadata_update[key] = value
    if metadata_update:
        observation.metadata.update(metadata_update)


@dataclass
class InMemoryLangfuseClient:
    """In-memory recorder satisfying :class:`LangfuseClient`.

    Captures every Trace / Span / Generation the observer creates as
    plain dataclass records reachable via :attr:`traces`. Tests assert
    against the records directly rather than mocking SDK methods.

    The recorder mints observation IDs internally via a simple
    counter; production callers (with a real `langfuse.Langfuse()`
    client) get SDK-minted UUIDs instead.
    """

    traces: dict[str, LangfuseTrace] = field(default_factory=dict[str, LangfuseTrace])
    _next_observation_id: int = 0

    def _mint_observation_id(self) -> str:
        # Sequential integer suffixes keep test diffs stable across runs.
        oid = f"obs-{self._next_observation_id}"
        self._next_observation_id += 1
        return oid

    def trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.traces[id] = LangfuseTrace(
            id=id,
            name=name,
            metadata=dict(metadata) if metadata is not None else {},
        )

    def update_trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        trace = self.traces.get(id)
        if trace is None:
            # Treat update-before-create as a create; this shouldn't
            # happen under the observer's emission order but stays
            # defensive against re-ordered events.
            self.trace(id=id, name=name, metadata=metadata)
            return
        if name is not None:
            trace.name = name
        if metadata is not None:
            trace.metadata.update(metadata)

    def span(
        self,
        *,
        trace_id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_observation_id: str | None = None,
        level: ObservationLevel = "DEFAULT",
        status_message: str | None = None,
    ) -> LangfuseSpanHandle:
        trace = self._get_trace(trace_id)
        observation = LangfuseObservation(
            id=self._mint_observation_id(),
            type="span",
            name=name,
            metadata=dict(metadata) if metadata is not None else {},
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
        )
        trace.observations.append(observation)
        return _InMemorySpanHandle(observation=observation)

    def generation(
        self,
        *,
        trace_id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        parent_observation_id: str | None = None,
        level: ObservationLevel = "DEFAULT",
        status_message: str | None = None,
        model: str | None = None,
        model_parameters: dict[str, Any] | None = None,
        input: Any = None,
        output: Any = None,
        usage: LangfuseUsage | None = None,
        prompt: Any = None,
    ) -> LangfuseGenerationHandle:
        trace = self._get_trace(trace_id)
        observation = LangfuseObservation(
            id=self._mint_observation_id(),
            type="generation",
            name=name,
            metadata=dict(metadata) if metadata is not None else {},
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
            model=model,
            model_parameters=dict(model_parameters) if model_parameters is not None else {},
            input=input,
            output=output,
            usage=usage,
            prompt_entity_link=prompt,
        )
        trace.observations.append(observation)
        return _InMemoryGenerationHandle(observation=observation)

    def force_flush(self, timeout_ms: int = 30_000) -> bool:
        # In-memory recorder has no outbound buffer; every observation
        # is captured synchronously on its create call. The ``timeout_ms``
        # parameter is accepted for Protocol compatibility but unused.
        del timeout_ms
        return True

    def _get_trace(self, trace_id: str) -> LangfuseTrace:
        trace = self.traces.get(trace_id)
        if trace is None:
            # Auto-create on first observation call. Real SDKs require
            # the Trace to exist first; we tolerate observer-side
            # ordering quirks by creating-on-demand.
            trace = LangfuseTrace(id=trace_id)
            self.traces[trace_id] = trace
        return trace


__all__ = [
    "InMemoryLangfuseClient",
    "LangfuseClient",
    "LangfuseGenerationHandle",
    "LangfuseObservation",
    "LangfuseSpanHandle",
    "LangfuseTrace",
    "LangfuseUsage",
    "ObservationLevel",
    "ObservationType",
]
