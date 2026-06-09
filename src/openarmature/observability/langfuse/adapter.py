# Bridges the langfuse Python SDK (v4.6+) onto the LangfuseClient
# Protocol. Validated against langfuse==4.7.0; the [langfuse] extras
# pin to `>=4.6,<5`. SDK churn before v4 (v2/v3 API removed in v4) is
# not supported — projects on v2/v3 should write their own adapter or
# upgrade.
#
# Shape mismatch the adapter handles:
#   - v4 has no explicit `client.trace(...)` — traces are auto-created
#     when the first observation starts. We cache the trace name +
#     metadata on `.trace()` and apply them via `propagate_attributes`
#     around EVERY observation under that trace_id. Propagating on
#     every observation (not just the first) keeps v4's
#     last-attribute-wins display logic from clobbering the trace's
#     display name when later observations land without the attribute
#     set.
#   - v4 unifies span and generation under `start_observation(as_type=)`.
#     The adapter routes `.span()` to `as_type="span"` and
#     `.generation()` to `as_type="generation"`.
#   - v4's `propagate_attributes(metadata=...)` requires Dict[str, str]
#     (not Dict[str, Any]). Non-string values are JSON-serialized at
#     the boundary.
#
# `update_trace` merges into the persistent trace_info cache so
# subsequent observations under the trace_id pick up the new values
# via `propagate_attributes`. Existing observations are NOT
# retroactively updated. The current OA LangfuseObserver doesn't
# actually invoke `update_trace` today — caller-supplied
# invocation-label lands in PR 4 via the trace_info cache before the
# first observation creates the trace — but the merge-then-propagate
# path is wired for forward compat.

"""LangfuseSDKAdapter: bridge langfuse>=4.6 onto the LangfuseClient Protocol."""

from __future__ import annotations

import json
from contextlib import ExitStack
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from .client import LangfuseGenerationHandle, LangfuseSpanHandle, LangfuseUsage, ObservationLevel
from .trace_id import _is_uuid, _to_otel_trace_id

if TYPE_CHECKING:
    from langfuse import Langfuse

try:
    from langfuse import propagate_attributes
    from langfuse.types import TraceContext
except ImportError as exc:  # pragma: no cover - exercised by extras-not-installed path
    raise ImportError(
        "openarmature.observability.langfuse.adapter requires the optional `langfuse` extras. "
        "Install with: pip install 'openarmature[langfuse]'"
    ) from exc


def _stringify_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    """Coerce metadata values to strings for v4's propagate_attributes,
    which only accepts ``Dict[str, str]``. Non-string scalars stringify
    via ``str()``; dicts and lists serialize via JSON with sorted keys
    so the round-trip is deterministic."""
    if metadata is None:
        return {}
    out: dict[str, str] = {}
    for key, value in metadata.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, dict | list):
            out[key] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        else:
            out[key] = str(value)
    return out


class _SpanHandle:
    """Wraps a langfuse LangfuseSpan / LangfuseGeneration to satisfy
    :class:`LangfuseSpanHandle` / :class:`LangfuseGenerationHandle`.

    The SDK's ``update(**fields)`` and ``end()`` shapes match our
    Protocol; the only translation is the ``status_message`` /
    ``level`` kwarg pass-through and the ``usage_details`` rename for
    Generation usage fields.
    """

    def __init__(self, langfuse_obs: Any) -> None:
        self._obs = langfuse_obs

    @property
    def id(self) -> str:
        # v4's LangfuseObservationWrapper exposes ``id`` as a property
        # backed by the underlying OTel span context. Cast to str so
        # static analysis sees the right shape.
        return cast("str", self._obs.id)

    def update(self, **fields: Any) -> None:
        kwargs: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "metadata":
                kwargs["metadata"] = value
            elif key == "status_message":
                kwargs["status_message"] = value
            elif key == "level":
                kwargs["level"] = value
            elif key == "usage":
                # Translate our LangfuseUsage record to v4's
                # usage_details dict shape. v4 expects integers.
                if isinstance(value, LangfuseUsage):
                    usage_details: dict[str, int] = {}
                    if value.input is not None:
                        usage_details["input"] = value.input
                    if value.output is not None:
                        usage_details["output"] = value.output
                    if value.total is not None:
                        usage_details["total"] = value.total
                    kwargs["usage_details"] = usage_details
            elif key == "output":
                kwargs["output"] = value
            elif key == "input":
                kwargs["input"] = value
            elif key == "model":
                kwargs["model"] = value
            elif key == "model_parameters":
                kwargs["model_parameters"] = value
            elif key == "prompt":
                kwargs["prompt"] = value
            else:
                # Unknown kwargs fall through to v4's update kwargs —
                # the SDK accepts arbitrary kwargs via its **kwargs
                # parameter.
                kwargs[key] = value
        self._obs.update(**kwargs)

    def end(self, *, end_time: datetime | None = None, **fields: Any) -> None:
        # Apply any field updates first (so they're set BEFORE the
        # observation closes), then call end(). v4's end() takes only
        # an optional ``end_time``; field mutation happens via update().
        if fields:
            self.update(**fields)
        if end_time is not None:
            self._obs.end(end_time=end_time)
        else:
            self._obs.end()


class LangfuseSDKAdapter:
    """Adapts a ``langfuse.Langfuse`` client (v4.6+) to the
    :class:`~openarmature.observability.langfuse.LangfuseClient`
    Protocol the :class:`LangfuseObserver` consumes.

    Usage::

        from langfuse import Langfuse
        from openarmature.observability.langfuse import (
            LangfuseObserver,
            LangfuseSDKAdapter,
        )

        client = Langfuse(
            public_key="pk-lf-...",
            secret_key="sk-lf-...",
            host="https://cloud.langfuse.com",
        )
        observer = LangfuseObserver(client=LangfuseSDKAdapter(client))
        compiled.attach_observer(observer)

    The adapter is stateful per-instance: it caches trace info keyed
    by trace_id and applies it to every observation under that trace
    via ``propagate_attributes``. The cache persists across the
    observation lifecycle so the trace name + metadata stay consistent
    instead of being clobbered by later observations under "last-
    attribute-wins" Langfuse-side processing. Cache cleanup is
    future-PR work (a `close_trace(trace_id)` hook on the Protocol);
    until then the cache grows linearly with unique trace_ids, which
    is bounded in practice by how many invocations a process runs.

    Safe to share across concurrent invocations on one ``Langfuse``
    client; the cache is keyed by trace_id.

    **Trace ID format.** OA uses standard UUID4 invocation_ids
    (8-4-4-4-12 dashed hex); Langfuse v4 is OTel-based and expects
    32-char lowercase hex (no dashes). The adapter converts on the
    way out via :func:`_to_otel_trace_id`. Same 128 bits, different
    representation — so a trace shows in Langfuse under
    ``b24eda93d06d4eaa9891ca5e56f35722`` while OA's
    ``correlation_id`` / ``invocation_id`` log line emits
    ``b24eda93-d06d-4eaa-9891-ca5e56f35722``. Strip the dashes when
    querying Langfuse for a specific invocation.
    """

    def __init__(self, client: Langfuse) -> None:
        self._client = client
        # Trace info cache, applied via propagate_attributes around
        # EVERY observation (not just the first). Langfuse v4's trace
        # name/metadata processing uses last-attribute-wins semantics,
        # so propagating only on the first observation lets later
        # observations clobber the trace's display name (the LAST
        # observation's name becomes the trace name). Propagating on
        # every observation under the same trace_id keeps the value
        # consistent. Cache cleanup is deferred to a future PR.
        self._trace_info: dict[str, dict[str, Any]] = {}

    def trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        # v4 has no explicit trace creation; cache the info and apply
        # it via propagate_attributes on every observation under this
        # trace_id so the trace's display name + metadata stay
        # consistent under v4's last-wins semantics.
        md: dict[str, Any] = dict(metadata) if metadata is not None else {}
        # Non-UUID invocation_id: the derived trace.id is a hash, not
        # reversible to the caller's id, so surface the raw id under
        # trace.metadata.invocation_id for lookup (§8.4.1). The key is
        # reserved (proposal 0041), so no caller metadata collides.
        if not _is_uuid(id):
            md.setdefault("invocation_id", id)
        self._trace_info[id] = {"name": name, "metadata": md}

    def update_trace(
        self,
        *,
        id: str,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        input: Any | None = None,
        output: Any | None = None,
    ) -> None:
        # Merge into the trace_info cache so subsequent observations
        # (and the first one if not yet created) pick up the updated
        # values. ``name`` / ``metadata`` are propagated via
        # ``propagate_attributes`` around every observation under
        # ``id``; ``input`` / ``output`` follow the SDK's
        # ``set_trace_io`` path (per proposal 0043 + the
        # empirically-validated v4.7.1 behaviour — see CHANGELOG).
        #
        # ``input`` is staged on the cache; applied to the FIRST real
        # observation that opens under this trace_id (``_start_observation``
        # below). Piggybacks on a real span so the trace tree gains no
        # extra observation in the common case.
        #
        # ``output`` is applied immediately via a synthetic short-lived
        # observation. By the time the LangfuseObserver dispatches the
        # invocation-completed event all real spans have ended, so a
        # synthetic span is the only path that has an active OTel span
        # context for ``set_trace_io`` to find.
        entry = self._trace_info.get(id)
        if entry is None:
            entry = {
                "name": name,
                "metadata": dict(metadata) if metadata is not None else {},
            }
            self._trace_info[id] = entry
        else:
            if name is not None:
                entry["name"] = name
            if metadata is not None:
                entry["metadata"].update(metadata)
        if input is not None:
            entry["pending_input"] = input
        if output is not None:
            self._emit_trace_output_synthetic(id, output)

    def _emit_trace_output_synthetic(self, trace_id: str, output: Any) -> None:
        # Open a synthetic short-lived observation, set
        # ``trace.output`` on it via ``set_trace_io``, end immediately.
        # The synthetic span shows in the trace as a small observation
        # named ``openarmature.trace_io``; the value lands on the
        # Langfuse Trace's ``output`` headline field through the
        # ``langfuse.trace.output`` OTel attribute set inside.
        #
        # Edge case: if no real node observation ever opened for this
        # trace (e.g., a resume-path validation failure aborted the
        # invocation before any node fired), the cached ``pending_input``
        # has no real span to piggyback on. Apply it here so the input
        # still lands — the synthetic observation becomes the sole
        # carrier for both fields. Pops the cache so we don't re-apply
        # if ``update_trace`` is called more than once.
        entry = self._trace_info.get(trace_id)
        pending_input = entry.pop("pending_input", None) if entry is not None else None

        trace_context: TraceContext = {"trace_id": _to_otel_trace_id(trace_id)}
        with ExitStack() as stack:
            if entry is not None:
                stack.enter_context(
                    propagate_attributes(
                        trace_name=entry["name"],
                        metadata=_stringify_metadata(entry["metadata"]),
                    )
                )
            obs = cast(
                "Any",
                self._client.start_observation(
                    name="openarmature.trace_io",
                    as_type="span",
                    trace_context=trace_context,
                ),
            )
            try:
                # Deprecation rationale on the equivalent call in
                # ``_start_observation``.
                obs.set_trace_io(input=pending_input, output=output)  # pyright: ignore[reportDeprecated]
            finally:
                obs.end()

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
        obs = self._start_observation(
            as_type="span",
            trace_id=trace_id,
            name=name,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
        )
        return _SpanHandle(obs)

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
        start_time: datetime | None = None,
    ) -> LangfuseGenerationHandle:
        extra_kwargs: dict[str, Any] = {
            "model": model,
            "model_parameters": model_parameters,
            "input": input,
            "output": output,
            "prompt": prompt,
        }
        # v4 expects usage_details (Dict[str, int]); translate from
        # our LangfuseUsage record.
        if usage is not None:
            usage_details: dict[str, int] = {}
            if usage.input is not None:
                usage_details["input"] = usage.input
            if usage.output is not None:
                usage_details["output"] = usage.output
            if usage.total is not None:
                usage_details["total"] = usage.total
            extra_kwargs["usage_details"] = usage_details
        if start_time is not None:
            extra_kwargs["start_time"] = start_time
        obs = self._start_observation(
            as_type="generation",
            trace_id=trace_id,
            name=name,
            metadata=metadata,
            parent_observation_id=parent_observation_id,
            level=level,
            status_message=status_message,
            **{k: v for k, v in extra_kwargs.items() if v is not None},
        )
        return _SpanHandle(obs)

    def force_flush(self, timeout_ms: int = 30_000) -> bool:
        """Best-effort flush of the underlying Langfuse client.

        ``timeout_ms`` is accepted for Protocol compatibility but
        **ignored**: the v4 Langfuse SDK's ``flush()`` method takes
        no timeout parameter and discards the underlying
        ``TracerProvider.force_flush()`` return value. The call is
        nonetheless synchronous — internally ``flush()`` waits on
        OTel's ``force_flush`` (default 30 s) and then ``.join()`` on
        the SDK's score and media ingestion queues — so by the time
        we return the OTel batch processor and ingestion queues have
        either drained or hit the SDK's internal default deadlines.

        Returns ``True`` once the SDK call completes without raising;
        a tight-deadline caller should pair this with its own
        wall-clock guard rather than relying on the return value.
        """
        del timeout_ms
        self._client.flush()
        return True

    def _start_observation(
        self,
        *,
        as_type: str,
        trace_id: str,
        name: str | None,
        metadata: dict[str, Any] | None,
        parent_observation_id: str | None,
        level: ObservationLevel,
        status_message: str | None,
        **extra: Any,
    ) -> Any:
        # Read the cached trace info (no pop — propagate on every
        # observation so v4's last-wins display logic keeps the
        # trace name + metadata stable across all observations under
        # this trace_id).
        trace_entry = self._trace_info.get(trace_id)

        # Build the start_observation kwargs. parent_observation_id is
        # threaded via trace_context (v4's TraceContext TypedDict
        # supports trace_id + parent_span_id). Convert OA's UUID4
        # invocation_id to OTel's hex form for the trace_id; the
        # parent_observation_id was minted by Langfuse on a prior
        # call and is already OTel-formatted.
        trace_context: TraceContext = {"trace_id": _to_otel_trace_id(trace_id)}
        if parent_observation_id is not None:
            trace_context["parent_span_id"] = parent_observation_id

        kwargs: dict[str, Any] = {
            "name": name or "observation",
            "as_type": as_type,
            "trace_context": trace_context,
            "metadata": metadata,
        }
        if level != "DEFAULT":
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = status_message
        kwargs.update(extra)

        with ExitStack() as stack:
            if trace_entry is not None:
                stack.enter_context(
                    propagate_attributes(
                        trace_name=trace_entry["name"],
                        metadata=_stringify_metadata(trace_entry["metadata"]),
                    )
                )
            obs = cast("Any", self._client.start_observation(**kwargs))
            # Proposal 0043 (PR 8.5a): apply any pending ``trace.input``
            # cached by ``update_trace`` to the FIRST real observation
            # under this trace. ``set_trace_io`` needs an active OTel
            # span context — piggybacking on the just-created
            # observation is the lowest-overhead path. ``pop`` so
            # subsequent observations under the same trace_id don't
            # re-apply (the value is one-shot per trace).
            #
            # The Langfuse SDK marks ``set_trace_io`` deprecated as of
            # v4.6 ("removal in a future major version"); per the
            # empirical verification in PR 8.5a it remains the only
            # path that surfaces ``trace.input`` in the Langfuse UI's
            # Traces list view. See CHANGELOG for the deprecation note.
            if trace_entry is not None:
                pending_input = trace_entry.pop("pending_input", None)
                if pending_input is not None:
                    obs.set_trace_io(input=pending_input)  # pyright: ignore[reportDeprecated]
            return obs


__all__ = ["LangfuseSDKAdapter"]
