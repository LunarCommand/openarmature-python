"""The call machinery the bundled LLM providers share.

A wire mapping subclasses :class:`_ProviderBase` and supplies the parts that
differ between wire formats: the request body, the response parser, the HTTP
error mapping and the structured-output path. Validation, call-level retry,
reask and the typed LLM events are the same for every mapping and live here.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, cast

import httpx
from pydantic import BaseModel

from openarmature.graph.events import (
    LlmCompletionEvent,
    LlmFailedEvent,
    LlmRetryAttemptEvent,
)
from openarmature.observability.correlation import (
    current_attempt_index,
    current_branch_name,
    current_branch_name_chain,
    current_correlation_id,
    current_dispatch,
    current_fan_out_index,
    current_fan_out_index_chain,
    current_invocation_id,
    current_namespace_prefix,
)
from openarmature.observability.diagnostics import STRUCTURED_OUTPUT_SCHEMA_NOT_HONORED, diagnostic
from openarmature.observability.metadata import AttributeValue, current_invocation_metadata

from .._event_fields import _event_request, _EventRequest
from ..errors import (
    LlmProviderError,
    ProviderInvalidResponse,
    ProviderUnavailable,
    StructuredOutputInvalid,
)
from ..messages import AssistantMessage, Message, Tool, ToolChoice, UserMessage
from ..provider import validate_message_list, validate_tool_choice, validate_tools
from ..response import Response, RuntimeConfig, Usage
from ._common import _as_sent, _normalize_response_schema, _schema_not_honored

# ``current_prompt_group`` / ``current_prompt_result`` are imported
# lazily inside :meth:`_ProviderBase.complete` to avoid a module-load
# cycle: ``openarmature.prompts.prompt`` imports ``RuntimeConfig`` from
# this package (for the ``SamplingConfig`` subclass), so a top-level
# import here would re-enter prompts.prompt before its types finish
# defining.

if TYPE_CHECKING:
    from openarmature.graph.middleware import RetryConfig


@dataclass(frozen=True)
class _StructuredPlan:
    """How one call obtains structured output: the messages its first attempt
    sends and the path it reports. A mapping subclasses this to carry what its
    body builder and response parser need."""

    messages: Sequence[Message]
    path: str | None


class _ProviderBase[PlanT: _StructuredPlan](ABC):
    """Shared ``complete()`` for the bundled wire mappings.

    A concrete subclass sets the ``_completions_path`` and ``_logger_name``
    class attributes, sets ``model``, ``_genai_system``,
    ``_populate_caller_metadata`` and ``_client`` in its constructor, and
    implements the abstract methods.
    """

    # The endpoint path ``complete()`` posts to, relative to the client's base URL.
    _completions_path: ClassVar[str]
    # The logger the shared code's warnings go to. Fixed per mapping, so a
    # caller's subclass still logs under ``openarmature``.
    _logger_name: ClassVar[str]

    model: str
    _genai_system: str
    _populate_caller_metadata: bool
    _client: httpx.AsyncClient

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ABCMeta sets __abstractmethods__ only after this runs, so abstractness
        # is read from the members. A missing attribute would otherwise surface
        # as an AttributeError on the first call, outside the typed-event path.
        abstract = any(getattr(getattr(cls, name, None), "__isabstractmethod__", False) for name in dir(cls))
        if abstract:
            return
        missing = [name for name in ("_completions_path", "_logger_name") if not hasattr(cls, name)]
        if missing:
            raise TypeError(f"{cls.__name__} must set the class attribute(s) {', '.join(missing)}")

    @property
    def _log(self) -> logging.Logger:
        return logging.getLogger(self._logger_name)

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Optional; async clients
        garbage-collect cleanly, but explicit close is RECOMMENDED in
        long-lived services to release the connection pool promptly."""
        await self._client.aclose()

    @abstractmethod
    def _plan_structured_output(
        self,
        messages: Sequence[Message],
        schema_dict: dict[str, Any] | None,
        *,
        tools: Sequence[Tool] | None,
        tool_choice: ToolChoice | None,
    ) -> PlanT: ...

    @abstractmethod
    def _build_request_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None,
        config: RuntimeConfig | None,
        schema_dict: dict[str, Any] | None,
        *,
        plan: PlanT | None = None,
        tool_choice: ToolChoice | None = None,
        collision_hint: Callable[[str], str | None] | None = None,
    ) -> dict[str, Any]: ...

    @abstractmethod
    def _stop_sequence_sent(self, body: Mapping[str, Any]) -> bool: ...

    @abstractmethod
    def _parse_response(
        self,
        payload: dict[str, Any],
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
        *,
        plan: PlanT | None = None,
    ) -> Response: ...

    @abstractmethod
    def _classify_http_error(self, resp: httpx.Response) -> LlmProviderError: ...

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] | None = None,
        config: RuntimeConfig | None = None,
        response_schema: dict[str, Any] | type[BaseModel] | None = None,
        tool_choice: ToolChoice | None = None,
        retry: RetryConfig | None = None,
    ) -> Response:
        """Single completion call.

        Pre-send validation runs first (per-message Pydantic +
        list-level invariants + response_schema shape check +
        ``tool_choice`` validation). HTTP errors map to canonical
        provider-error categories. The successful 200 body is parsed
        into a :class:`Response`; failure to parse raises
        ``provider_invalid_response``; failure to validate the response
        content against ``response_schema`` raises
        ``structured_output_invalid``.

        When ``response_schema`` is supplied as a Pydantic BaseModel
        subclass, ``Response.parsed`` is a validated instance of that
        class; when supplied as a JSON Schema dict,
        ``Response.parsed`` is the deserialized dict.

        ``tool_choice`` is validated against ``tools``:
        ``"required"`` and the ``ForceTool`` record both demand
        non-empty ``tools``, and ``ForceTool.name`` must appear in the
        supplied list. Violations raise ``provider_invalid_request``
        BEFORE any HTTP request is sent.

        When ``retry`` is supplied, the wire call is retried on
        transient provider errors per the config's classifier and
        backoff (defaulting to the canonical transient categories with
        exponential jittered backoff). The call is validated once and
        each attempt's body built from that attempt's config and
        messages; an error raised before a request is sent is never
        retried. Exactly one terminal observability event
        (LlmCompletionEvent or LlmFailedEvent) fires for the call's
        terminal outcome regardless of attempt count. It describes the
        request of the attempt whose outcome it reports, and its
        ``latency_ms`` covers the whole call, retries and backoff
        included. A per-attempt LlmRetryAttemptEvent also fires for
        each attempt that sends its request (driving the per-attempt
        LLM span surface), each describing that attempt's request and
        carrying just its latency. The ``on_retry`` hook is not
        exception-isolated (mirroring ``RetryMiddleware``); an exception
        raised by it propagates out of the call.
        """
        # §5.5: emit the typed completion / failure event around the wire call
        # so a backend can build the span. Queue-mediated so §6 serial ordering
        # holds across event sources. ``current_dispatch()`` is ``None`` outside
        # an invocation (direct provider use), and the call then emits nothing.
        #
        # ``call_id`` is minted per ``complete()`` call, which gives a failed
        # call an id distinct from any retry-attempt sibling (0058), since the
        # retry middleware re-enters ``complete()`` per attempt.
        dispatch = current_dispatch()
        call_id = str(uuid.uuid4())
        # Capture prompt context AT DISPATCH TIME (in the node task's
        # context). The delivery worker (asyncio.create_task'd at
        # ``invoke()`` entry, before any node body runs) has a stale
        # ContextVar snapshot — reading ``current_prompt_result()``
        # from inside the observer in the worker task returns ``None``
        # even when a node body opened a ``with_active_prompt`` block.
        # Snapshot here; the observer reads from the event payload.
        # Lazy import: see module-level comment for the cycle reason.
        from openarmature.prompts.context import (
            current_prompt_group,
            current_prompt_result,
        )

        active_prompt = current_prompt_result()
        active_prompt_group = current_prompt_group()
        # graph-engine §6: the call's typed event describes the request of the
        # attempt whose outcome it reports, and a call rejected before any
        # request was sent describes the caller's own. So this starts as the
        # caller's request, which is what a call rejected in validation reports,
        # and ``build_body`` moves it to each attempt's request in turn. Only a
        # non-Message object makes rendering raise, which is no §7 failure for
        # the event path below to report, so it sits outside the try.
        reported = _event_request(messages, config)
        # Wall-clock latency measured at the adapter boundary per
        # proposal 0049's LlmCompletionEvent.latency_ms contract /
        # proposal 0058's LlmFailedEvent.latency_ms contract. Window
        # spans from "just before validation runs" through to either
        # "parsed Response in hand" (success) or "the §7 exception
        # was raised" (failure). perf_counter is the monotonic high-
        # resolution clock for elapsed-time measurements.
        adapter_start = time.perf_counter()
        try:
            # Per proposal 0058: pre-send validation raises (e.g.,
            # provider_invalid_request, provider_unsupported_content_
            # block) MUST dispatch LlmFailedEvent before the exception
            # propagates. The whole validation + wire-call sequence
            # runs inside the protected scope so any §7 category
            # exception (pre-send OR adapter-caught) flows through the
            # same emission path.
            validate_message_list(messages)
            validate_tools(tools)
            # ``validate_tool_choice`` runs after ``validate_tools``
            # so the name-membership check sees a structurally valid
            # tools list.
            validate_tool_choice(tool_choice, tools)
            schema_dict, schema_class = _normalize_response_schema(response_schema)
            plan = self._plan_structured_output(messages, schema_dict, tools=tools, tool_choice=tool_choice)
            stop_sent = False

            def build_body(
                attempt: int,
                attempt_config: RuntimeConfig | None,
                attempt_messages: Sequence[Message],
                inherited_extras: frozenset[str] | None,
            ) -> dict[str, Any]:
                # Proposal 0095: the wire body is assembled PER ATTEMPT so a
                # retry can vary sampling (0095a per-attempt override) and/or the
                # messages (0095b reask transcript). tools / schema are fixed
                # across attempts. Absent an override and a reask, every attempt
                # passes the base config + base messages, so a retry resends a
                # byte-identical body.
                #
                # A collision reads as a bug in one config, and on a retry it
                # can instead be a mismatch between two: the base supplied the
                # extras key and the override declared the field. Attempt 0 sends
                # cleanly in that case, so the only report the caller gets has to
                # name the channel they should change.
                #
                # Keyed on whether the COLLIDING key was inherited, not on
                # whether a merge happened. An override that declares a field and
                # names it in its own extras collides on a merged attempt too,
                # and sending that caller to inspect a base config they never
                # wrote is worse than saying nothing.
                def hint(key: str) -> str | None:
                    # None means no override applied to this attempt, so the
                    # collision is between a caller's declared field and their own
                    # extras key in one config. There is no second channel to
                    # explain, and the mapping's message already names both values.
                    if inherited_extras is None:
                        return None
                    if key in inherited_extras:
                        return (
                            "this attempt merges the base config with a per-attempt "
                            f"override, and {key!r} was inherited from the base's extras "
                            "while the override declares the field; set it in the channel "
                            "the base used"
                        )
                    return (
                        f"the per-attempt override sets {key!r} as a declared field and as "
                        "an extras key; remove one"
                    )

                nonlocal reported, stop_sent
                # graph-engine §6: a retry rejected before sending reports the
                # request it was building, with no path since none went out. A
                # first attempt rejected here leaves the caller's own in place.
                request = _event_request(attempt_messages, attempt_config)
                if attempt > 0:
                    reported = request
                body = self._build_request_body(
                    attempt_messages,
                    tools,
                    attempt_config,
                    schema_dict,
                    plan=plan,
                    tool_choice=tool_choice,
                    collision_hint=hint,
                )
                reported = replace(request, structured_output_path=plan.path)
                stop_sent = self._stop_sequence_sent(body)
                return body

            # Per-attempt LLM span surface (observability §5.5 under
            # llm-provider §7.1 call-level retry): dispatch one
            # LlmRetryAttemptEvent per attempt so the observer can render
            # one openarmature.llm.complete span per attempt. Each describes
            # its attempt's own request, which build_body recorded before the
            # request went out. Outside an invocation (``dispatch`` is None)
            # nothing is dispatched.
            def emit_attempt(
                idx: int,
                attempt_latency_ms: float,
                attempt_response: Response | None,
                attempt_exc: LlmProviderError | None,
                retry_reason: str | None = None,
            ) -> None:
                request = _as_sent(reported, attempt_exc)
                # observability §5.5.7: one record per wire attempt however many
                # observers are attached, none included, so it precedes the
                # dispatch check.
                if _schema_not_honored(request, attempt_exc, stop_sent=stop_sent):
                    self._warn_schema_not_honored(call_id, idx)
                if dispatch is None:
                    return
                dispatch(
                    self._build_llm_retry_attempt_event(
                        llm_attempt_index=idx,
                        latency_ms=attempt_latency_ms,
                        call_id=call_id,
                        request=request,
                        active_prompt=active_prompt,
                        active_prompt_group=active_prompt_group,
                        token_budget=(active_prompt.token_budget if active_prompt is not None else None),
                        response=attempt_response,
                        exc=attempt_exc,
                        retry_reason=retry_reason,
                    ),
                )

            response = await self._do_complete_with_retry(
                build_body, config, plan, schema_dict, schema_class, retry, emit_attempt
            )
        except LlmProviderError as exc:
            # Failure path: dispatch a typed LlmFailedEvent per
            # proposal 0058. Only §7 category exceptions
            # (LlmProviderError subclasses) trigger a typed event —
            # internal bugs / unexpected exceptions raise without
            # event emission because they aren't part of the §7
            # contract. The caller-side exception flow is unchanged:
            # the exception still raises out of complete().
            latency_ms_failed = (time.perf_counter() - adapter_start) * 1000.0
            if dispatch is not None:
                dispatch(
                    self._build_llm_failed_event(
                        exc,
                        latency_ms_failed,
                        call_id=call_id,
                        request=_as_sent(reported, exc),
                        active_prompt=active_prompt,
                        active_prompt_group=active_prompt_group,
                        token_budget=(active_prompt.token_budget if active_prompt is not None else None),
                    ),
                )
            raise
        latency_ms = (time.perf_counter() - adapter_start) * 1000.0

        if dispatch is not None:
            # Success path: emit the typed LlmCompletionEvent.
            # External custom observers consuming LLM events MUST
            # filter via ``isinstance(event, LlmCompletionEvent)``;
            # the sentinel namespace pattern is retired in this
            # release.
            dispatch(
                self._build_llm_completion_event(
                    response,
                    latency_ms,
                    call_id=call_id,
                    request=reported,
                    active_prompt=active_prompt,
                    active_prompt_group=active_prompt_group,
                    token_budget=(active_prompt.token_budget if active_prompt is not None else None),
                ),
            )
        return response

    @staticmethod
    def _config_for_attempt(
        base: RuntimeConfig | None,
        overrides: list[RuntimeConfig] | None,
        attempt: int,
    ) -> tuple[RuntimeConfig | None, frozenset[str]]:
        """Resolve the RuntimeConfig for one call-level retry attempt.

        Attempt 0 uses the caller's base config unmodified; retry ``i``
        (attempt ``i > 0``) merges ``overrides[i-1]`` onto the base -- the
        override's non-None fields replace, a None field (like an absent one)
        inherits the base -- and the last entry carries forward when the
        schedule is shorter than the retry count. ``extras`` merges by the same
        per-key rule: a key the override sets replaces that key, a key it does
        not mention inherits the base's. Attempt 0 and the no-override
        case return the caller's base config as-is; only the override path
        returns a fresh ``model_copy``. The caller's config is never mutated
        either way -- the base path relies on the downstream body build reading
        it read-only.

        Returns the attempt's config and the set of ``extras`` keys it inherited
        from the base rather than receiving from the override.
        """
        # Proposal 0095 per-attempt override. A None override field inheriting
        # the base is the §6 null-skip rule (None means unset, not "clear to
        # null"); the never-mutate guarantee honors §5 immutability.
        if attempt == 0 or not overrides:
            return base, frozenset()
        override = overrides[min(attempt - 1, len(overrides) - 1)]
        base_or_empty = base if base is not None else RuntimeConfig()
        # exclude_none (not exclude_unset): a None override field inherits the
        # base per §6 null-skip, rather than an explicit None clobbering it.
        #
        # `extras` is handled separately because it is the one field whose
        # default is not None: an empty container survives `exclude_none`, so
        # leaving it in the dump would clear the caller's vendor knobs on any
        # override that never mentioned them.
        #
        # §7.1: "Undeclared extras (§6) merge by the same per-key rule." The
        # per-key rule is the one the DECLARED fields follow, and the analogue of
        # a field is an extras KEY rather than the container: a key set in the
        # override replaces the base's value for that key, a key the override does
        # not mention inherits the base's. So the container merges.
        #
        # The merge does not resolve an inherited key that collides with a field
        # the override declares; the mapping rejects it like any other managed-field
        # collision. Dropping the key instead would make one config mean two
        # things depending on how it was assembled, rejected when a caller writes
        # both and accepted when a merge produces both, and §6 separately forbids
        # silently discarding a conflicting extras value. A caller who wants the
        # override to change such a value keeps it in the same channel the base
        # used.
        update = override.model_dump(exclude_none=True, exclude={"extras"})
        update["extras"] = {**base_or_empty.extras, **override.extras}
        # A key present in the base and unmentioned by the override arrived here
        # by inheritance rather than by the caller writing it alongside the
        # field, which is what decides whether a collision crossed channels.
        inherited = frozenset(base_or_empty.extras) - frozenset(override.extras)
        return base_or_empty.model_copy(update=update), inherited

    @staticmethod
    def _append_reask_pair(
        transcript: Sequence[Message],
        model_output: str,
        correction: str,
    ) -> list[Message]:
        """Append a structured-output reask pair to a COPY of the transcript:
        the model's verbatim raw output as a fresh ``assistant`` message, then
        the caller's correction as a ``user`` message. The caller's messages are
        never mutated -- a new list plus new message objects.
        """
        # Proposal 0095b. Always appends a fresh assistant message (never
        # continues a trailing one): §3 requires the last message before a call
        # to be user/tool, so the transcript never ends in assistant. Proposal
        # 0110 removed the continuation clause 0095 had added for that
        # (unreachable) case.
        return [
            *transcript,
            AssistantMessage(content=model_output),
            UserMessage(content=correction),
        ]

    async def _do_complete_with_retry(
        self,
        build_body: Callable[
            [int, RuntimeConfig | None, Sequence[Message], frozenset[str] | None], dict[str, Any]
        ],
        base_config: RuntimeConfig | None,
        plan: PlanT,
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
        retry: RetryConfig | None,
        emit_attempt: Callable[[int, float, Response | None, LlmProviderError | None, str | None], None]
        | None = None,
    ) -> Response:
        """Run the wire call with optional call-level retry.

        Loops the underlying wire call on transient provider errors per the
        retry config. ``build_body`` assembles the outbound body for an attempt
        from its index, config and messages (proposal 0095: retries MAY vary sampling
        via ``per_attempt_override`` and/or the messages via ``reask``); absent
        both, every attempt uses ``base_config`` + ``plan.messages`` unchanged,
        preserving the byte-identical replay. ``emit_attempt`` (when supplied)
        fires once per attempt with that attempt's index, latency, outcome
        (response on success, exception on failure), and ``retry_reason`` (None
        on the base attempt, else "transient" | "reask") so the caller can
        dispatch the per-attempt LLM span event. The terminal outcome (success,
        retry exhaustion, or a non-retryable error) is what reaches
        ``complete()`` for the single terminal typed event, so exactly one
        terminal event still fires per ``complete()`` call.
        """
        base_messages = plan.messages
        if retry is None:
            body = build_body(0, base_config, base_messages, None)
            attempt_start = time.perf_counter()
            try:
                response = await self._do_complete(body, schema_dict, schema_class, plan)
            except LlmProviderError as exc:
                if emit_attempt is not None:
                    emit_attempt(0, (time.perf_counter() - attempt_start) * 1000.0, None, exc, None)
                raise
            if emit_attempt is not None:
                emit_attempt(0, (time.perf_counter() - attempt_start) * 1000.0, response, None, None)
            return response
        # Lazy import avoids a module-load cycle: graph.middleware.retry
        # imports llm.errors. Resolve None config fields to the canonical
        # defaults, mirroring RetryMiddleware.
        from openarmature.graph.middleware.retry import (
            default_classifier,
            exponential_jitter_backoff,
        )

        classifier = retry.classifier or default_classifier
        backoff = retry.backoff or exponential_jitter_backoff
        # Proposal 0095: the per-attempt override schedule + reask builder live
        # only on the llm-provider-scoped LlmRetryConfig superset; a plain
        # RetryConfig has neither (getattr -> None), so every attempt replays
        # base_config + base_messages.
        overrides = getattr(retry, "per_attempt_override", None)
        reask = getattr(retry, "reask", None)
        # The working transcript accumulates reask pairs across retries: it
        # starts as base_messages and each reask REPLACES it with a fresh list
        # (from _append_reask_pair), never mutating in place. On a
        # structured-output call base_messages is itself already an augmented
        # copy, so the caller's messages are never mutated either way.
        transcript: Sequence[Message] = base_messages
        # retry_reason for the CURRENT attempt = why the PREVIOUS attempt was
        # retried (None for the base attempt). Set at each ``continue``.
        retry_reason: str | None = None
        attempt = 0
        while True:
            attempt_config, inherited_extras = self._config_for_attempt(base_config, overrides, attempt)
            # Attempt 0 runs the caller's config untouched, so no override is in
            # play and a collision there has only one channel to report.
            merged = attempt > 0 and bool(overrides)
            body = build_body(attempt, attempt_config, transcript, inherited_extras if merged else None)
            attempt_start = time.perf_counter()
            try:
                response = await self._do_complete(body, schema_dict, schema_class, plan)
            except LlmProviderError as exc:
                # Per-attempt event fires for every attempt, including this
                # failed one (the terminal failed attempt's span carries the
                # final category).
                if emit_attempt is not None:
                    emit_attempt(
                        attempt, (time.perf_counter() - attempt_start) * 1000.0, None, exc, retry_reason
                    )
                # Reask (0095b) makes a structured_output_invalid retryable FOR
                # THIS CALL when a builder is present -- a call-level convenience,
                # NOT via the classifier (structured_output_invalid stays
                # non-transient). No graph state at the call boundary; the
                # default classifier ignores the None state.
                reasked = reask is not None and isinstance(exc, StructuredOutputInvalid)
                retryable = reasked or classifier(exc, None)
                # Re-raise on exhaustion or a non-retryable error so complete()
                # emits the single terminal LlmFailedEvent.
                if attempt + 1 >= retry.max_attempts or not retryable:
                    raise
                # Re-test inline (equivalent to ``reasked``) so the type checker
                # narrows exc -> StructuredOutputInvalid and reask -> non-None.
                if reask is not None and isinstance(exc, StructuredOutputInvalid):
                    # Feed the model its own output + the caller's correction so
                    # the next attempt is informed, not a byte-identical replay.
                    # The reask builder is caller code and is NOT exception-
                    # isolated: if it raises, or returns a non-str that fails
                    # message construction, re-raise the original
                    # structured_output_invalid (with the builder error chained)
                    # so it propagates through the terminal-event path -- exactly
                    # one terminal LlmFailedEvent still fires -- rather than
                    # leaking a non-§7 error and masking the real failure.
                    try:
                        transcript = self._append_reask_pair(transcript, exc.output_content, reask(exc))
                    except Exception as reask_error:
                        # The chained cause carries this, but the surfaced error
                        # reads as a bad model rather than a broken builder, and
                        # a caller who never inspects __cause__ debugs the wrong
                        # thing. Say which one failed where it will be seen.
                        self._log.warning(
                            "reask builder raised on attempt %d (%s: %s); "
                            "reask is disabled for this call and the "
                            "structured_output_invalid failure stands",
                            attempt,
                            type(reask_error).__name__,
                            reask_error,
                        )
                        raise exc from reask_error
                    next_retry_reason = "reask"
                else:
                    next_retry_reason = "transient"
                # on_retry is not exception-isolated (matches RetryMiddleware);
                # a raise propagates out of the call.
                if retry.on_retry is not None:
                    await retry.on_retry(exc, attempt)
                await asyncio.sleep(backoff(attempt))
                attempt += 1
                retry_reason = next_retry_reason
                continue
            if emit_attempt is not None:
                emit_attempt(
                    attempt, (time.perf_counter() - attempt_start) * 1000.0, response, None, retry_reason
                )
            return response

    def _build_llm_completion_event(
        self,
        response: Response,
        latency_ms: float,
        *,
        call_id: str,
        request: _EventRequest,
        active_prompt: Any,
        active_prompt_group: Any,
        token_budget: Any,
    ) -> LlmCompletionEvent:
        """Construct the typed LlmCompletionEvent for the success path.

        Sources identity / scoping fields from the calling-node
        ContextVars and outcome fields from the response. Request-side
        fields come from ``request``, the request the event describes;
        the prompt-context snapshots were taken at dispatch time and the
        call-id minted at the call's start.

        The calling-node namespace is the FULL namespace tuple (not
        the legacy sentinel pseudo-namespace); node_name is the last
        element of the namespace (the user-defined node that issued
        the call). Outside any node body (namespace empty), node_name
        is the empty string.
        """

        namespace = current_namespace_prefix()
        node_name = namespace[-1] if namespace else ""
        # invocation_id is normally always present once invoke() entry
        # has run, but the LLM provider can be exercised in test
        # fixtures outside an invocation. Spec proposal 0049's field
        # table types invocation_id as a non-nullable string, so we
        # fall back to empty string rather than None to keep the event
        # constructable. Downstream observers using invocation_id as a
        # correlation key should treat "" as "not in an invocation"
        # and either skip or special-case those events; collisions
        # across multiple out-of-invocation calls are theoretically
        # possible but not a path production code should hit.
        invocation_id = current_invocation_id() or ""
        caller_metadata: Mapping[str, AttributeValue] | None = None
        if self._populate_caller_metadata:
            # Snapshot via dict() so downstream consumers see a stable
            # frozen view; if a node body mutates metadata after the
            # snapshot, the event still carries the at-emission view.
            caller_metadata = dict(current_invocation_metadata())
        # ``output_content`` is None on tool-call-only assistant
        # messages per llm-provider §6 mutual-exclusion: the
        # tool-call path and structured-content path are mutually
        # exclusive at the response level, and provider.complete()
        # leaves the AssistantMessage.content as the empty string on
        # the tool-call path (which we project to None per the
        # typed-event contract).
        output_content = response.message.content or None
        return LlmCompletionEvent(
            invocation_id=invocation_id,
            correlation_id=current_correlation_id(),
            node_name=node_name,
            namespace=namespace,
            attempt_index=current_attempt_index(),
            fan_out_index=current_fan_out_index(),
            branch_name=current_branch_name(),
            # Proposal 0084: enclosing fan-out / branch lineage chains.
            fan_out_index_chain=current_fan_out_index_chain(),
            branch_name_chain=current_branch_name_chain(),
            provider=self._genai_system,
            model=self.model,
            response_id=response.response_id,
            response_model=response.response_model,
            usage=response.usage,
            latency_ms=latency_ms,
            finish_reason=response.finish_reason,
            input_messages=request.input_messages,
            output_content=output_content,
            # Proposal 0076: the model's output tool calls, populated
            # unconditionally (empty list on a no-tool completion).
            output_tool_calls=list(response.message.tool_calls or []),
            request_params=request.request_params,
            request_extras=request.request_extras,
            structured_output_path=request.structured_output_path,
            active_prompt=active_prompt,
            active_prompt_group=active_prompt_group,
            # Proposal 0083: the active prompt's advisory token budget, sourced
            # by the caller from active_prompt.token_budget at dispatch time.
            token_budget=token_budget,
            call_id=call_id,
            caller_invocation_metadata=caller_metadata,
        )

    def _build_llm_failed_event(
        self,
        exc: LlmProviderError,
        latency_ms: float,
        *,
        call_id: str,
        request: _EventRequest,
        active_prompt: Any,
        active_prompt_group: Any,
        token_budget: Any,
    ) -> LlmFailedEvent:
        """Construct the typed LlmFailedEvent for the failure path.

        Sources identity / scoping fields from the calling-node
        ContextVars, failure fields from the raised exception, and
        request-side fields from ``request``, the request the event
        describes. Field set mirrors LlmCompletionEvent (identity +
        request-side) plus the three failure-specific fields.

        ``error_type`` defaults to the exception class name — falls
        into the "upstream exception class name" style documented in
        the field table. Providers that have a vendor error code
        available (e.g. ``rate_limit_exceeded`` for OpenAI) can
        override with vendor-specific detail in a future proposal; for
        now the class name is the safest default since every
        LlmProviderError subclass carries one.
        """

        namespace = current_namespace_prefix()
        node_name = namespace[-1] if namespace else ""
        invocation_id = current_invocation_id() or ""
        caller_metadata: Mapping[str, AttributeValue] | None = None
        if self._populate_caller_metadata:
            caller_metadata = dict(current_invocation_metadata())
        # Response-side surface, populated only for structured_output_invalid
        # (proposal 0082): the failure is a downstream validation step on an
        # intact wire response, so the model's output genuinely exists and was
        # attached to the error at the parse/validate call site. None for every
        # other category (no response received).
        output_content: str | None = None
        finish_reason: str | None = None
        usage: Usage | None = None
        response_id: str | None = None
        response_model: str | None = None
        error_message = str(exc)
        if isinstance(exc, StructuredOutputInvalid):
            # Empty content projects to None (as the success path does with
            # ``content or None``), so both observers omit it identically
            # rather than one rendering "" and the other dropping it.
            output_content = exc.output_content or None
            finish_reason = exc.finish_reason
            usage = exc.usage
            response_id = exc.response_id
            response_model = exc.response_model
            # error_message carries the failing locator (proposal 0082): the
            # terse category summary alone drops the which-field detail, so
            # combine it with the error_message observers triage on.
            error_message = f"{exc}: {exc.error_message}"
        return LlmFailedEvent(
            invocation_id=invocation_id,
            correlation_id=current_correlation_id(),
            node_name=node_name,
            namespace=namespace,
            attempt_index=current_attempt_index(),
            fan_out_index=current_fan_out_index(),
            branch_name=current_branch_name(),
            # Proposal 0084: enclosing fan-out / branch lineage chains.
            fan_out_index_chain=current_fan_out_index_chain(),
            branch_name_chain=current_branch_name_chain(),
            provider=self._genai_system,
            model=self.model,
            latency_ms=latency_ms,
            input_messages=request.input_messages,
            request_params=request.request_params,
            request_extras=request.request_extras,
            structured_output_path=request.structured_output_path,
            active_prompt=active_prompt,
            active_prompt_group=active_prompt_group,
            # Proposal 0083: advisory budget carried on the failure event too;
            # None-carrying keeps it inert for non-response failures, while a
            # structured_output_invalid failure (which does carry usage) drives
            # the same reactive budget eval as a completion.
            token_budget=token_budget,
            call_id=call_id,
            error_category=exc.category,
            error_type=type(exc).__name__,
            error_message=error_message,
            caller_invocation_metadata=caller_metadata,
            output_content=output_content,
            finish_reason=finish_reason,
            usage=usage,
            response_id=response_id,
            response_model=response_model,
        )

    def _build_llm_retry_attempt_event(
        self,
        *,
        llm_attempt_index: int,
        latency_ms: float,
        call_id: str,
        request: _EventRequest,
        active_prompt: Any,
        active_prompt_group: Any,
        token_budget: Any,
        response: Response | None = None,
        exc: LlmProviderError | None = None,
        retry_reason: str | None = None,
    ) -> LlmRetryAttemptEvent:
        """Construct an LlmRetryAttemptEvent for one in-call attempt.

        Identity / scoping fields mirror the terminal events, and the
        request-side fields come from ``request``, the request this
        attempt sent. The outcome comes from ``response`` (a successful
        attempt) or ``exc`` (a failed attempt). Exactly one of the two
        is supplied. ``llm_attempt_index`` is the call-level retry
        index, distinct from the node-level ``attempt_index`` sourced
        from the ContextVar.
        """
        namespace = current_namespace_prefix()
        node_name = namespace[-1] if namespace else ""
        invocation_id = current_invocation_id() or ""
        caller_metadata: Mapping[str, AttributeValue] | None = None
        if self._populate_caller_metadata:
            caller_metadata = dict(current_invocation_metadata())
        # Identity / scoping + request-side fields are shared; only the
        # outcome fields differ between the success and failure branches.
        base: dict[str, Any] = {
            "invocation_id": invocation_id,
            "correlation_id": current_correlation_id(),
            "node_name": node_name,
            "namespace": namespace,
            "attempt_index": current_attempt_index(),
            "fan_out_index": current_fan_out_index(),
            "branch_name": current_branch_name(),
            # Proposal 0084: enclosing fan-out / branch lineage chains.
            "fan_out_index_chain": current_fan_out_index_chain(),
            "branch_name_chain": current_branch_name_chain(),
            "provider": self._genai_system,
            "model": self.model,
            "call_id": call_id,
            "llm_attempt_index": llm_attempt_index,
            "latency_ms": latency_ms,
            "input_messages": request.input_messages,
            "request_params": request.request_params,
            "request_extras": request.request_extras,
            "structured_output_path": request.structured_output_path,
            "active_prompt": active_prompt,
            "active_prompt_group": active_prompt_group,
            # Proposal 0083: advisory per-prompt token budget snapshot.
            "token_budget": token_budget,
            "caller_invocation_metadata": caller_metadata,
            # Proposal 0095: "transient" | "reask" on a retry attempt, None on
            # the base attempt.
            "retry_reason": retry_reason,
        }
        if response is not None:
            return LlmRetryAttemptEvent(
                **base,
                response_id=response.response_id,
                response_model=response.response_model,
                usage=response.usage,
                finish_reason=response.finish_reason,
                output_content=response.message.content or None,
                # Proposal 0076: the attempt's output tool calls — the
                # OTel observer renders the output tool-call span
                # attributes from this per-attempt event.
                output_tool_calls=list(response.message.tool_calls or []),
            )
        if exc is None:
            raise ValueError("_build_llm_retry_attempt_event requires response or exc")
        # Proposal 0082: a structured_output_invalid failed attempt carries the
        # response-side surface (the wire response was intact); the OTel observer
        # renders it on the error span and the §11 token metric records from it.
        # None for every other category. error_message carries the failing locator.
        error_message = str(exc)
        response_fields: dict[str, Any] = {}
        if isinstance(exc, StructuredOutputInvalid):
            response_fields = {
                # Empty content -> None (as the success path projects), so both
                # observers omit it identically.
                "output_content": exc.output_content or None,
                "finish_reason": exc.finish_reason,
                "usage": exc.usage,
                "response_id": exc.response_id,
                "response_model": exc.response_model,
            }
            error_message = f"{exc}: {exc.error_message}"
        return LlmRetryAttemptEvent(
            **base,
            error_category=exc.category,
            error_type=type(exc).__name__,
            error_message=error_message,
            **response_fields,
        )

    def _warn_schema_not_honored(self, call_id: str, attempt_index: int) -> None:
        self._log.warning(
            "%s returned content that is not a JSON object on the native structured-output "
            "path (call %s, attempt %d): the schema constraint did not reach or bind the model",
            self.model,
            call_id,
            attempt_index,
            extra={
                **diagnostic(STRUCTURED_OUTPUT_SCHEMA_NOT_HONORED),
                "openarmature.llm.provider": self._genai_system,
                # observability §5.5.7: the bound model, as the span's
                # openarmature.llm.model, not the one the response reports.
                "openarmature.llm.model": self.model,
                "openarmature.llm.call_id": call_id,
                "openarmature.llm.attempt_index": attempt_index,
            },
        )

    async def _do_complete(
        self,
        body: dict[str, Any],
        schema_dict: dict[str, Any] | None,
        schema_class: type[BaseModel] | None,
        plan: PlanT,
    ) -> Response:
        """Wire-call helper: separated from ``complete()`` so the
        LLM-provider span hook in ``complete()`` can wrap success and
        failure paths uniformly."""
        try:
            resp = await self._client.post(self._completions_path, json=body)
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(str(exc)) from exc

        if resp.status_code != 200:
            raise self._classify_http_error(resp)

        try:
            payload_raw = resp.json()
        except ValueError as exc:
            raise ProviderInvalidResponse(f"POST {self._completions_path} returned non-JSON body") from exc
        if not isinstance(payload_raw, dict):
            raise ProviderInvalidResponse(f"POST {self._completions_path} returned a non-object body")
        return self._parse_response(cast("dict[str, Any]", payload_raw), schema_dict, schema_class, plan=plan)


__all__ = ["_ProviderBase", "_StructuredPlan"]
