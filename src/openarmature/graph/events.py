# Spec: realizes graph-engine §6 (started/completed event pair model
# from proposal 0005, v0.6.0). FanOutEventConfig is the fan-out node
# event payload added by proposal 0013 (v0.10.0).

"""Node-boundary observer events.

Each node attempt produces a started/completed event PAIR. The engine
dispatches the started event before invoking the wrapped node function
and the completed event after the reducer merge succeeds (with
``post_state`` populated) or after the node, reducer, or state
validation fails (with ``error`` populated).

Frozen dataclass; observers receive a snapshot, not a live handle.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from openarmature.observability.metadata import AttributeValue

from .cause_chain import CaughtException
from .errors import RuntimeGraphError
from .state import State

# TYPE_CHECKING import — the runtime Usage class lives in the llm
# package, which transitively imports from graph.events (the
# OpenAI provider imports NodeEvent). Using a TYPE_CHECKING import
# plus a string annotation on LlmCompletionEvent.usage avoids the
# circular runtime import while keeping pyright type-safe.
if TYPE_CHECKING:
    from openarmature.llm.messages import ToolCall
    from openarmature.llm.response import Usage
    from openarmature.retrieval.response import EmbeddingUsage, RerankUsage, ScoredDocument

# Sentinel empty metadata mapping for events constructed without a
# live caller-metadata snapshot (test helpers, synthetic events).
# Read-only proxy keeps the default allocation-free.
_EMPTY_METADATA: MappingProxyType[str, AttributeValue] = MappingProxyType({})


# Spec: realizes observability §5.4 fan-out attributes via the
# event-payload mechanism added by proposal 0013 (v0.10.0). Backend
# observers cache ``parent_node_name`` off the fan-out node's
# started event and apply it on every per-instance span they
# synthesize (observability §5.4 mandates
# ``openarmature.fan_out.parent_node_name`` on per-instance spans).
@dataclass(frozen=True)
class FanOutEventConfig:
    """Resolved fan-out configuration carried on a fan-out node's
    own events.

    Fan-out node events carry the resolved configuration so backend
    observers can attribute the fan-out node span (``item_count`` /
    ``concurrency`` / ``error_policy``) and synthesize per-instance
    spans with the right ``parent_node_name``.

    Populated ONLY on ``started`` and ``completed`` events for a
    fan-out node itself (partition by node type, not event category;
    INCLUDES retried attempts of a fan-out node when retry middleware
    wraps it). All other events leave ``NodeEvent.fan_out_config``
    null.

    Field shapes:

    - ``item_count``: non-negative int. The resolved instance count
      (matches ``count_field`` value when configured; matches
      ``len(items_field)`` in items_field mode).
    - ``concurrency``: positive int OR ``None`` (unbounded). Zero or
      negative is rejected at config resolution time as
      ``fan_out_invalid_concurrency``. Backend mappings may translate
      ``None`` to a sentinel at the attribute layer (e.g.,
      ``openarmature.fan_out.concurrency = 0``); that translation is
      observer-internal, not engine-internal.
    - ``error_policy``: one of ``"fail_fast"`` or ``"collect"``.
    - ``parent_node_name``: the fan-out node's name in the parent
      graph. Carried here for caching by backend observers when
      attributing per-instance spans.

    All four fields MUST be present when ``fan_out_config`` is
    populated. Only ``concurrency`` is nullable.
    """

    item_count: int
    concurrency: int | None
    error_policy: str
    parent_node_name: str


# Spec: realizes observability §5.7 parallel-branches attributes via
# the event-payload mechanism added by proposal 0044 (v0.36.0).
# Backend observers cache ``parent_node_name`` off the parallel-
# branches node's started event and apply it on every per-branch
# dispatch span they synthesize (observability §5.7 mandates
# ``openarmature.parallel_branches.parent_node_name`` on per-branch
# dispatch spans). Mirrors :class:`FanOutEventConfig`'s shape.
@dataclass(frozen=True)
class ParallelBranchesEventConfig:
    """Resolved parallel-branches configuration carried on a parallel-
    branches node's own events.

    Parallel-branches node events carry the resolved configuration so
    backend observers can attribute the parallel-branches node span
    (``branch_count`` / ``error_policy``) and synthesize per-branch
    dispatch spans with the right ``branch_names`` + ``parent_node_name``.

    Populated ONLY on ``started`` and ``completed`` events for a
    parallel-branches node itself (partition by node type, not event
    category; INCLUDES retried attempts of a parallel-branches node
    when retry middleware wraps it). All other events leave
    ``NodeEvent.parallel_branches_config`` null.

    Field shapes:

    - ``branch_names``: non-empty ordered tuple of strings. The branch
      identifiers in declaration / dispatch order, as configured on
      the parallel-branches node.
    - ``branch_count``: positive int. Equals ``len(branch_names)``.
      Surfaced explicitly so observers don't have to derive it.
    - ``error_policy``: one of ``"fail_fast"`` or ``"collect"``.
    - ``parent_node_name``: the parallel-branches node's name in the
      parent graph. Carried here for caching by backend observers
      when attributing per-branch dispatch spans.

    All four fields MUST be present when ``parallel_branches_config``
    is populated.
    """

    branch_names: tuple[str, ...]
    branch_count: int
    error_policy: str
    parent_node_name: str


# Spec: realizes graph-engine §6 NodeEvent (started/completed pair
# model from proposal 0005, v0.6.0). The ``checkpoint_saved`` phase
# is the Python shape for §10.8 save events (§10.8 SHOULDs an event
# emit but leaves the shape implementation-defined). ``fan_out_config``
# is the observability §5.4 / proposal 0013 (v0.10.0) addition.
@dataclass(frozen=True)
class NodeEvent:
    """A single node-boundary event delivered to observers.

    - ``phase`` is ``"started"`` (dispatched before the node runs) or
      ``"completed"`` (dispatched after the node returns or raises
      and the merge runs/fails). Each node attempt produces exactly
      one of each in that order. The engine ALSO dispatches a
      ``"checkpoint_saved"`` event on the same shape after a
      successful ``Checkpointer.save`` call; observers MUST opt in
      explicitly via ``phases={"checkpoint_saved"}`` to receive these
      (default subscription is ``{"started", "completed"}`` only, so
      legacy observers don't see them).
    - ``node_name`` is the name under which this node was registered
      in its immediate containing graph.
    - ``namespace`` is an ordered sequence of node names from the
      outermost graph down to this node. For a node in the outermost
      graph, ``namespace`` is ``(node_name,)``. For nested subgraphs,
      the chain extends.
    - ``step`` is a monotonically-increasing counter starting at 0,
      scoped to a single outermost invocation. Subgraph-internal nodes
      increment the same counter. The started/completed pair for one
      attempt share the same step.
    - ``pre_state`` is the state the node received, before reducer
      merge. Populated on both phases (identical across the pair).
    - ``post_state`` is the state after the node's partial update
      merged successfully. Populated only on ``completed`` events
      that succeeded.
    - ``error`` is the wrapped runtime error (``NodeException``,
      ``ReducerError``, or ``StateValidationError``) when the node
      failed. Populated only on ``completed`` events that failed.
    - ``parent_states`` carries one state snapshot per containing
      graph, outermost first; for a node in the outermost graph it's
      an empty tuple. Invariant:
      ``len(parent_states) == len(namespace) - 1``.
    - ``attempt_index`` is the 0-based index of this attempt among
      any retries. ``0`` for nodes not wrapped by retry middleware.
    - ``fan_out_index`` is the 0-based index of this fan-out instance
      among its siblings. ``None`` for nodes not inside a fan-out.
    - ``fan_out_config`` carries resolved fan-out configuration on
      events from a fan-out NODE itself. See
      :class:`FanOutEventConfig`. ``None`` on every other event.
    - ``branch_name`` is the non-empty string name of the
      parallel-branches branch this event came from. ``None`` for
      nodes outside any branch. The combination of ``namespace``,
      ``branch_name``, ``fan_out_index``, ``attempt_index``, and
      ``phase`` jointly uniquely identifies an event source.
      ``branch_name`` and ``fan_out_index`` are independent; both
      MAY be present when a branch's subgraph contains a fan-out
      (or a fan-out instance contains a parallel-branches node).

    Invariants:

    - On ``started`` events, ``post_state`` and ``error`` MUST both
      be ``None``.
    - On ``completed`` events, exactly one of ``post_state`` and
      ``error`` is populated.

    **Synthetic phases.** ``"checkpoint_saved"`` and
    ``"checkpoint_migrated"`` repurpose this dataclass for non-node
    events. Both
    are opt-in via ``phases={...}`` on observer registration;
    default subscriptions are ``{"started", "completed"}`` only, so
    legacy observers never see them. Conventions on synthetic
    events:

    - ``checkpoint_saved``: ``pre_state`` carries the saved
      post-merge state (still a real ``State`` instance for this
      phase), ``post_state`` is ``None``. ``step`` matches the
      saving node's step.
    - ``checkpoint_migrated``: ``step=-1`` (no graph-step
      sequencing; migrations run before any node fires).
      ``node_name="openarmature.checkpoint.migrate"`` and
      ``namespace=("openarmature.checkpoint.migrate",)`` are
      dotted-pseudo identifiers, not real node names. ``pre_state``
      carries a private ``_MigrationSummary`` dataclass with
      ``from_version`` / ``to_version`` / ``chain_length``, NOT a
      ``State`` instance. ``parent_states`` is the empty tuple.

    Because ``pre_state`` is no longer guaranteed to be a ``State``
    on the synthetic phases, its type is declared as ``Any`` and
    observer authors who subscribe to those phases MUST narrow
    per-phase before reading ``pre_state``.
    """

    node_name: str
    namespace: tuple[str, ...]
    step: int
    phase: Literal[
        "started",
        "completed",
        "checkpoint_saved",
        # Synthetic phase per spec §6 cross-ref in proposal 0014:
        # fires once at the start of a versioned resume to carry
        # the migration chain's metadata. ``pre_state`` on this
        # phase carries a ``_MigrationSummary`` (not a ``State``);
        # the field type stays permissive on this dataclass and
        # the OTel observer narrows defensively via ``isinstance``.
        "checkpoint_migrated",
    ]
    pre_state: Any
    post_state: State | None
    error: RuntimeGraphError | None
    parent_states: tuple[State, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    fan_out_config: FanOutEventConfig | None = None
    # Per observability §5.7 / proposal 0044 (v0.36.0): resolved
    # parallel-branches configuration carried on the parallel-branches
    # NODE's own events (mirroring ``fan_out_config`` on a fan-out
    # NODE's events). Populated on both ``started`` and ``completed``
    # events for a parallel-branches NODE (including retried
    # attempts); absent on every other event. Carries the §5.7
    # branch_count + error_policy + parent_node_name surface so
    # backend observers can attribute the parallel-branches NODE span
    # and synthesize per-branch dispatch spans without re-reading the
    # graph's static config.
    parallel_branches_config: ParallelBranchesEventConfig | None = None
    # Per pipeline-utilities §11 / graph-engine §6 (proposal 0011):
    # optional non-empty string populated only on events from nodes
    # that execute inside a parallel-branches branch. The
    # combination of ``namespace``, ``branch_name``,
    # ``fan_out_index``, ``attempt_index``, and ``phase`` jointly
    # uniquely identifies an event source. ``branch_name`` and
    # ``fan_out_index`` are independent; both MAY be present
    # simultaneously when a branch's subgraph contains a fan-out
    # (and vice versa).
    branch_name: str | None = None
    # Per proposal 0045 (v0.37.0): per-depth lineage chains parallel
    # to ``namespace``.  Position ``i`` is the fan_out_index (or
    # branch_name) at the dispatch boundary leading to namespace
    # depth ``i+1`` — or ``None`` when that boundary is a subgraph
    # wrapper (not a fan-out, not a parallel-branches branch).
    # ``fan_out_index`` and ``branch_name`` above carry the
    # INNERMOST values; the chains carry the full lineage so
    # observers can apply the §3.4 lineage-aware boundary rule
    # without re-deriving it from successive events.
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()
    # Per observability §5.3 + the coord-thread
    # ``clarify-subgraph-name-semantics`` resolution: chain of
    # compiled-subgraph identities parallel to the wrapper-depth
    # positions of ``namespace``. Index ``i`` is the identity for
    # the wrapper at ``namespace[i]`` (or ``None`` when that
    # wrapper has no tracked identity); chain length equals the
    # depth of wrapper nesting (always ``< len(namespace)`` since
    # the last element of ``namespace`` is the current node, not
    # a wrapper). Observers read by depth and emit it as
    # ``observation.metadata.subgraph_name`` (Langfuse) /
    # ``openarmature.subgraph.name`` (OTel), falling back to the
    # empty string when ``None`` per §5.3's "if the implementation
    # tracks one" clause.
    subgraph_identities: tuple[str | None, ...] = ()
    # Per observability §3.4 + §5.6 (proposal 0034): snapshot of the
    # caller-supplied invocation metadata at event-construction
    # time. The engine reads ``current_invocation_metadata()`` when
    # it constructs the event (in the engine task / node body's
    # Context); the observer reads from the snapshot on the event
    # rather than re-reading the ContextVar at observer time —
    # critical because the observer runs on the engine's
    # ``deliver_loop`` task whose Context is frozen at invoke time
    # (asyncio.create_task copies the parent Context at task
    # creation), so the live ContextVar value in the deliver_loop
    # would NOT reflect mid-invocation augmentations made by node
    # bodies running in the main engine task. Observers emit each
    # entry as ``openarmature.user.<key>`` (OTel, §5.6) /
    # ``metadata.<key>`` (Langfuse, §8.4.1+§8.4.2).
    caller_invocation_metadata: Mapping[str, AttributeValue] = field(default_factory=lambda: _EMPTY_METADATA)


# Spec: realizes observability §3.4 + graph-engine §6 augmentation
# event mechanism (proposal 0040). Emitted by
# ``set_invocation_metadata`` when called mid-invocation; carries the
# delta + the augmenting context's lineage identity so observers can
# resolve which of their open observations belong to the augmenting
# context's subtree and apply the entries in place.
@dataclass(frozen=True)
class MetadataAugmentationEvent:
    """A metadata-augmentation event delivered to observers.

    Emitted by :func:`openarmature.observability.metadata.set_invocation_metadata`
    when called mid-invocation. Carries:

    - ``entries``: the delta merged into the per-async-context
      invocation metadata mapping by the call. Read-only view.
    - ``namespace`` / ``attempt_index`` / ``fan_out_index`` /
      ``branch_name``: the four lineage fields that jointly identify
      the augmenting execution context (the calling node's identity
      tuple). When ``set_invocation_metadata`` is called from outside
      a node body, ``namespace`` is the empty tuple, ``attempt_index``
      is ``0``, and both ``fan_out_index`` and ``branch_name`` are
      ``None`` — the invocation-level identity.

    Distinct from :class:`NodeEvent` because there is no node phase,
    no pre/post state, and no error: this event reports a side-channel
    augmentation, not a node-attempt boundary. The event is NOT
    subject to the observer ``phases`` filter (which only governs
    ``NodeEvent`` phases); the delivery worker forwards it to every
    subscribed observer. Observers that handle it iterate their open
    observations whose lineage is an ancestor of (or equal to) the
    augmenting context's lineage and apply the entries as
    ``openarmature.user.<key>`` (OTel) / ``metadata.<key>``
    (Langfuse).
    """

    entries: Mapping[str, AttributeValue]
    namespace: tuple[str, ...]
    attempt_index: int = 0
    fan_out_index: int | None = None
    branch_name: str | None = None
    # Per proposal 0045 (v0.37.0): the augmenter's per-depth lineage
    # chain.  Two parallel tuples indexed by namespace position —
    # position ``i`` is the fan_out_index (or branch_name) at
    # namespace depth ``i+1``, or ``None`` if that depth's dispatch
    # boundary is not a fan-out instance (not a parallel-branches
    # branch).  Required by §3.4's lineage-aware boundary rule so
    # observers can identify the augmenter's call-stack ancestor
    # chain rather than only the innermost dispatch.
    fan_out_index_chain: tuple[int | None, ...] = ()
    branch_name_chain: tuple[str | None, ...] = ()


# Spec: realizes observability §8.4.1 *Trace input/output sourcing*
# (proposal 0043). Emitted by the engine at invocation entry, BEFORE
# any node fires. Carries the initial state observers can use to
# resolve trace.input via the three-lever decision tree (caller hook
# → raw state when disable_state_payload is OFF → privacy-safe
# minimal stub). Distinct from NodeEvent because there is no node
# context — the event is invocation-scoped.
@dataclass(frozen=True)
class InvocationStartedEvent:
    """An invocation-entry event delivered to observers.

    Emitted once per invocation, before any node fires. Observers that
    populate Trace-level input fields (the Langfuse observer, today)
    consume it to resolve ``trace.input`` per the three-lever decision
    tree. Observers without a Trace-level input concept (the OTel
    observer) treat it as a no-op.

    Carries:

    - ``initial_state``: the raw state object the engine constructed
      from ``invoke()``'s arguments (the typed-state instance).
    - ``invocation_id``: the invocation id (caller-supplied or
      framework-generated).
    - ``correlation_id``: the correlation id when present.
    - ``entry_node``: the outermost-graph entry node name.

    The event is NOT subject to the observer ``phases`` filter (which
    only governs ``NodeEvent`` phases); the delivery worker forwards it
    to every subscribed observer.
    """

    initial_state: Any
    invocation_id: str
    correlation_id: str | None
    entry_node: str


# Spec: realizes observability §8.4.1 *Trace input/output sourcing*
# (proposal 0043). Emitted by the engine at invocation exit, on both
# the success path (status="completed") and the failure path
# (status="failed"). Carries the final state observers can use to
# resolve trace.output via the three-lever decision tree, plus the
# closed status enum for the privacy-safe minimal stub.
@dataclass(frozen=True)
class InvocationCompletedEvent:
    """An invocation-exit event delivered to observers.

    Emitted once per invocation, after the last node has fired (and
    after a failure boundary on the failure path). Observers that
    populate Trace-level output fields (the Langfuse observer, today)
    consume it to resolve ``trace.output`` per the three-lever
    decision tree. Observers without a Trace-level output concept (the
    OTel observer) treat it as a no-op.

    Carries:

    - ``final_state``: the state at invocation exit (the engine's
      returned state on the success path; the state at point-of-
      failure on the failure path).
    - ``status``: closed enum ``"completed"`` (END reached) or
      ``"failed"`` (any node, edge, reducer, or boundary validator
      raised before END).
    - ``final_node``: the name of the node whose execution preceded
      the END-reached transition on the success path, or the node
      that raised on the failure path.
    - ``invocation_id`` / ``correlation_id``: the run + correlation ids.

    The event is NOT subject to the observer ``phases`` filter; the
    delivery worker forwards it to every subscribed observer.
    """

    final_state: Any
    status: Literal["completed", "failed"]
    final_node: str
    invocation_id: str
    correlation_id: str | None


# Spec: realizes proposal 0049's first spec-normatively-typed event
# variant on the observer event union (graph-engine §6 +
# observability §5.5.7). Dispatched on every LLM provider call that
# returns a structured response, alongside the calling node's
# NodeEvent pair. Failure cases (provider exceptions, malformed
# responses) flow through the existing exception path and do NOT
# emit this variant. Not subject to the §6 ``phases`` subscription
# filter (matches MetadataAugmentationEvent / InvocationStartedEvent
# / InvocationCompletedEvent treatment).
#
# Field naming matches the spec-canonical names verbatim per the spec
# Q5 ack — Python snake_case happens to match the spec table 1:1.
#
# Spec proposal 0057 (v0.51.0) extension: adds 8 additive request-side
# fields (input_messages, output_content, request_params,
# request_extras, active_prompt, active_prompt_group, call_id,
# response_model) and renames request_id → response_id to match the
# response-side data the field carries. Inline image bytes in
# input_messages MUST be redacted per observability §5.5.5 before
# population — the provider reuses _serialize_messages_for_payload
# which already enforces the redaction. The three payload-bearing
# fields (input_messages, output_content, request_extras) are
# populated unconditionally on the typed event per §5.5.7; observer-
# side privacy gates (OTel disable_provider_payload, Langfuse equivalents)
# apply at rendering, symmetric with the §5.5.1 span attribute path.
# Custom queryable observers (per observability §9) own their own
# redaction posture — gating belongs at rendering with the consumer's
# awareness.
@dataclass(frozen=True)
class LlmCompletionEvent:
    """A typed LLM provider call event delivered to observers.

    Carries identity, scoping, and outcome data for an LLM call as
    structured fields. Observer code filters by type discrimination
    (``isinstance(event, LlmCompletionEvent)``) rather than by the
    impl-current sentinel-namespace string match the legacy
    NodeEvent pattern uses.

    Field set:

    - ``invocation_id``: the outer invocation's identifier.
    - ``correlation_id``: cross-backend correlation id when present.
    - ``node_name``: the user-defined node that issued the call.
    - ``namespace``: the calling node's namespace tuple (NOT the
      legacy sentinel namespace).
    - ``attempt_index``: retry-attempt index (0 on first attempt).
    - ``fan_out_index``: fan-out instance index when the calling
      node ran inside a fan-out instance; ``None`` otherwise.
    - ``branch_name``: parallel-branches branch name when the
      calling node ran inside a branch; ``None`` otherwise.
    - ``provider``: provider identifier; matches ``gen_ai.system``.
    - ``model``: the model identifier the call targeted (the
      request-side bound model; distinct from ``response_model``).
    - ``response_id``: provider-returned response id; ``None`` when
      the provider didn't return one.
    - ``response_model``: provider-returned model identifier;
      distinct from ``model`` (the provider may return a more
      specific identifier than the one requested). ``None`` when
      the provider didn't return one.
    - ``usage``: token-accounting record reusing the existing
      ``openarmature.llm.response.Usage`` class. ``None`` when the
      call returned no usage at all.
    - ``latency_ms``: wall-clock latency measured at the adapter
      boundary, in milliseconds. ``None`` when latency was not
      measured.
    - ``finish_reason``: the call's finish reason; ``None`` when
      the call did not complete normally.
    - ``input_messages``: the message list the call was made with,
      serialized to the plain-dict shape. Non-nullable; empty list
      when the call had no history. Inline image bytes are
      redacted before population (see the comment block above for
      the redaction contract).
    - ``output_content``: the assistant message's content string
      from the response. ``None`` on tool-call-only responses
      (the structured-response and tool-call paths are mutually
      exclusive at the response level).
    - ``output_tool_calls``: the assistant message's output tool
      calls (the ``ToolCall`` records). Populated unconditionally;
      empty list when the response carried no tool calls. The output
      tool calls live here rather than in ``output_content`` (which
      is the response text and is empty on a tool-call-only response).
    - ``request_params``: the GenAI request-parameter set the
      caller supplied. Absence-is-meaningful: only caller-supplied
      keys appear; empty mapping when none supplied. Keys are the
      cross-vendor parameter names without the ``gen_ai.request.``
      prefix (e.g. ``temperature``, ``max_tokens``).
    - ``request_extras``: the ``RuntimeConfig`` extras pass-
      through bag in native mapping form (not JSON-encoded).
      Empty mapping when no extras supplied.
    - ``active_prompt``: 5-field identity snapshot of the active
      ``PromptResult`` at LLM-call time (``name`` / ``version`` /
      ``label`` / ``template_hash`` / ``rendered_hash``).
      ``None`` when the call ran outside any prompt-context
      binding. Typed as ``Any`` because the prompts package
      imports State indirectly; observer-side narrowing reads
      the attribute names directly.
    - ``active_prompt_group``: ``{group_name}`` snapshot when the
      call ran inside a ``PromptGroup`` context; ``None``
      otherwise. Same ``Any`` typing rationale as
      ``active_prompt``.
    - ``call_id``: per-call disambiguator minted by the
      implementation. Always present, freshly minted per
      ``provider.complete()`` call, stable for the call's
      lifetime, unique within the run. Distinct from
      ``response_id``.
    - ``caller_invocation_metadata``: optional snapshot of caller-
      supplied invocation metadata at LLM-call time. OPTIONAL; the
      python OpenAIProvider populates it by default so
      the bundled OTel/Langfuse observers can emit the
      ``openarmature.user.<key>`` span-attribute family without an
      extra opt-in. Pass ``populate_caller_metadata=False`` to suppress
      the snapshot. Future non-OpenAI providers MAY default to
      ``None``.
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    response_id: str | None
    response_model: str | None
    # Usage is a string-typed forward reference per the TYPE_CHECKING
    # import above — keeps the runtime import direction graph → llm
    # off the module-load path while preserving pyright resolution.
    usage: "Usage | None"
    latency_ms: float | None
    finish_reason: str | None
    # Proposal 0057 (spec v0.51.0) additive request-side fields.
    # Non-nullable for input_messages / request_params /
    # request_extras — absence is represented as empty list / empty
    # mapping, not None. output_content stays nullable for tool-
    # call-only assistant messages.
    input_messages: list[dict[str, Any]]
    output_content: str | None
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None
    # Proposal 0076 (spec v0.67.0): the assistant message's output tool
    # calls in typed-event-native form (the ToolCall records, not a
    # pre-serialized shape — they carry no inline-image bytes, so the
    # input_messages redaction-driven pre-serialization doesn't apply).
    # Populated unconditionally by the provider; empty list when the
    # response carried no tool calls. Source for the §5.5.1 gated
    # ``openarmature.llm.output.tool_calls`` serialization + the §5.5.10
    # ungated ``.count`` / ``.names`` / ``.ids`` identity projections.
    # Defaulted (default_factory) so existing kwargs-constructors that
    # predate this field keep working; the provider always populates it.
    output_tool_calls: list["ToolCall"] = field(default_factory=list["ToolCall"])


# Spec: realizes proposal 0058's second spec-normatively-typed event
# variant on the observer event union (graph-engine §6 +
# observability §5.5.7), accepted at spec v0.53.0. Dispatched on the
# observer delivery queue whenever a provider.complete() call raises
# a §7 category exception — covers BOTH the adapter-caught provider-
# exception path AND the pre-send validation raise path
# (provider_invalid_request / provider_unsupported_content_block
# raise before any provider contact). The event is dispatched
# ALONGSIDE the exception, not in place of it; caller-side exception
# flow is unchanged.
#
# Mutual exclusion with LlmCompletionEvent on the same
# provider.complete() call — implementations MUST NOT emit both for
# the same call. Conformance fixture 072 locks this down.
#
# Privacy posture identical to LlmCompletionEvent: input_messages /
# request_params / request_extras are populated unconditionally per
# §5.5.7; observer-side privacy gates (OTel disable_provider_payload,
# Langfuse equivalents) apply at rendering. Inline image bytes are
# redacted per observability §5.5.5 before population. Custom
# queryable observers own their own redaction posture.
@dataclass(frozen=True)
class LlmFailedEvent:
    """A typed LLM provider call failure event delivered to observers.

    Carries identity, scoping, and failure-context data for an LLM
    call that raised a llm-provider category exception. Observer
    code filters by type discrimination (``isinstance(event,
    LlmFailedEvent)``) rather than by the impl-current sentinel-
    namespace string match.

    Identity / scoping / request-side field set mirrors
    ``LlmCompletionEvent`` 1:1 — same field semantics, same nullability
    rules. Response-side fields (``response_id``, ``response_model``,
    ``usage``, ``output_content``, ``finish_reason``) are ABSENT from
    this variant — no response was received.

    Failure-specific fields:

    - ``error_category``: the llm-provider normative error category
      the call raised. One of the 9 canonical strings
      (``provider_authentication``, ``provider_unavailable``,
      ``provider_invalid_model``, ``provider_model_not_loaded``,
      ``provider_rate_limit``, ``provider_invalid_response``,
      ``provider_invalid_request``,
      ``provider_unsupported_content_block``,
      ``structured_output_invalid``). Always present.
    - ``error_type``: OPTIONAL impl-level / vendor-specific error
      type or code. Two acceptable styles:
      vendor error code (e.g. ``"rate_limit_exceeded"``) OR
      upstream exception class name (e.g. ``"RateLimitError"``).
      ``None`` when no impl-side type is available.
    - ``error_message``: human-readable message from the raised
      exception. Always present (empty string when the exception
      carried no message).
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    latency_ms: float | None
    input_messages: list[dict[str, Any]]
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    error_category: str
    error_message: str
    error_type: str | None = None
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Python-internal per-attempt LLM event. NOT a spec-normative event type
# (unlike LlmCompletionEvent / LlmFailedEvent): it is the observer-side
# vehicle for the observability §5.5 per-attempt span surface under
# llm-provider §7.1 call-level retry. One is dispatched per in-call
# attempt (including the single attempt of a no-retry call); the OTel
# observer renders one openarmature.llm.complete span from each, while
# the terminal LlmCompletionEvent / LlmFailedEvent stay one-per-call
# (payload/latency, Langfuse mapping, the fixture-072 mutual exclusion).
@dataclass(frozen=True)
class LlmRetryAttemptEvent:
    """One LLM-call attempt delivered to observers for per-attempt span
    rendering.

    Carries the full request-side surface plus that attempt's outcome.
    ``error_category`` discriminates the outcome: ``None`` for a
    successful attempt (the response-side fields are populated), a
    category string for a failed attempt (the response-side fields are
    ``None`` — no response was received).

    Field set:

    - ``llm_attempt_index``: the call-level retry-attempt index, ``0``
      for the first attempt and ``0..N-1`` across the N attempts of a
      call-level retry. Distinct from ``attempt_index`` (the node-level
      retry index used for calling-span resolution); the two are
      independent.
    - identity / scoping (``invocation_id`` ... ``call_id``) and the
      request side (``input_messages`` / ``request_params`` /
      ``request_extras`` / ``active_prompt`` / ``active_prompt_group``)
      mirror :class:`LlmCompletionEvent`, carried on every attempt.
    - response side (``response_id`` / ``response_model`` / ``usage`` /
      ``finish_reason`` / ``output_content`` / ``output_tool_calls``):
      populated on a successful attempt; ``None`` / empty list on a
      failed attempt. ``output_tool_calls`` is the source the OTel
      observer renders the output tool-call attributes
      from (this is the per-attempt event that drives the LLM span).
    - failure side (``error_category`` / ``error_message`` /
      ``error_type``): populated on a failed attempt; ``None`` on a
      successful one.
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    call_id: str
    llm_attempt_index: int
    latency_ms: float | None
    input_messages: list[dict[str, Any]]
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    response_id: str | None = None
    response_model: str | None = None
    usage: "Usage | None" = None
    finish_reason: str | None = None
    output_content: str | None = None
    error_category: str | None = None
    error_message: str | None = None
    error_type: str | None = None
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None
    # Proposal 0076: the attempt's output tool calls (ToolCall records),
    # mirroring LlmCompletionEvent.output_tool_calls. Populated on a
    # successful attempt; empty list on a failed one (no response). The
    # OTel observer renders the observability §5.5.1 / §5.5.10 output
    # tool-call span attributes from this field (the per-attempt event is
    # the LLM-span source).
    output_tool_calls: list["ToolCall"] = field(default_factory=list["ToolCall"])


# Spec: realizes graph-engine §6 + observability §5.5.9 -- the typed
# EmbeddingEvent / EmbeddingFailedEvent pair (proposal 0059,
# retrieval-provider capability). Dispatched on the observer delivery
# queue per EmbeddingProvider.embed() call: the success variant after the
# response is parsed + validated, the failure variant alongside a raised
# §7 category exception (mutually exclusive per call). Scalar
# fan_out_index / branch_name only; the lineage chains arrive uniformly
# across the provider events with proposal 0084 (v0.81.0). input_strings /
# request_extras are payload-bearing, populated unconditionally; observer-
# side privacy gates (OTel disable_provider_payload, Langfuse equivalents)
# apply at rendering, symmetric with LlmCompletionEvent.
@dataclass(frozen=True)
class EmbeddingEvent:
    """A typed embedding provider call event delivered to observers.

    Carries identity, scoping, and outcome data for a successful
    ``EmbeddingProvider.embed()`` call. Observer code filters by type
    discrimination (``isinstance(event, EmbeddingEvent)``).

    The identity / scoping / request-side fields mirror
    ``LlmCompletionEvent``'s convention; the outcome fields are
    embedding-specific:

    - ``input_strings``: the input strings the call was made with;
      non-nullable, populated unconditionally (privacy gating is
      observer-side at rendering).
    - ``input_count``: ``len(input_strings)``; a convenience field.
    - ``dimensions``: the output vector dimensionality from the response;
      ``None`` when the response surfaced no determinate dimensionality.
    - ``response_model`` / ``response_id``: the provider-returned model
      and response identifiers; ``None`` when the provider returned none.
    - ``usage``: the embedding token record; ``None`` when the call
      returned no usage.
    - ``request_params``: the embedding request parameters the caller
      supplied (e.g. ``dimensions``). Absence-is-meaningful: only supplied
      keys appear; an empty mapping when none.
    - ``request_extras``: the runtime-config extras pass-through bag.
    - ``active_prompt`` / ``active_prompt_group``: prompt-context
      snapshots at embed-call time; ``None`` outside a binding.
    - ``call_id``: a per-call disambiguator, always present, freshly
      minted per ``embed()`` call.
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    response_id: str | None
    response_model: str | None
    # EmbeddingUsage is a string-typed forward reference per the
    # TYPE_CHECKING import -- keeps the runtime import direction
    # graph -> retrieval off the module-load path.
    usage: "EmbeddingUsage | None"
    latency_ms: float | None
    input_strings: list[str]
    input_count: int
    dimensions: int | None
    # Spec graph-engine §6 (proposal 0089, v0.84.0): the output embedding
    # vectors (sourced from EmbeddingResponse.vectors at dispatch). Populated
    # unconditionally on the success event -- payload-bearing like
    # input_strings, gated observer-side at the rendering boundary (OTel
    # disable_provider_payload, Langfuse equivalents) per observability
    # §5.5.9. The §8.4.5 embedding.output mapping reads this field, not the
    # response object. No output field on EmbeddingFailedEvent (no response).
    output_vectors: list[list[float]]
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Spec: the failure sibling of EmbeddingEvent (proposal 0059). Dispatched
# whenever EmbeddingProvider.embed() raises a §7 category exception --
# covers both the provider-caught path and the pre-send validation raise
# (provider_invalid_request on an empty input list). Dispatched ALONGSIDE
# the exception, not in place of it; mutually exclusive with EmbeddingEvent
# on the same call. The response-side fields are absent (no response).
@dataclass(frozen=True)
class EmbeddingFailedEvent:
    """A typed embedding provider call failure event delivered to observers.

    Carries identity, scoping, and failure-context data for an ``embed()``
    call that raised a retrieval-provider category exception. Observer code
    filters by type discrimination
    (``isinstance(event, EmbeddingFailedEvent)``).

    The identity / scoping / request-side field set mirrors
    ``EmbeddingEvent``; the response-side fields are absent. Failure-
    specific fields:

    - ``error_category``: the error category the call raised (one of the
      embedding-applicable provider categories). Always present.
    - ``error_type``: an optional impl-level / vendor-specific type or
      code; ``None`` when unavailable.
    - ``error_message``: a human-readable message; always present (the
      empty string when the exception carried no message).
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    latency_ms: float | None
    input_strings: list[str]
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    error_category: str
    error_message: str
    error_type: str | None = None
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Spec: realizes graph-engine §6 -- the typed RerankEvent / RerankFailedEvent
# pair (proposal 0060, retrieval-provider rerank capability), the rerank
# sibling to the EmbeddingEvent / EmbeddingFailedEvent pair. Dispatched on the
# observer delivery queue per RerankProvider.rerank() call: the success variant
# after the response is parsed + validated (retrieval-provider §6), the failure
# variant alongside a raised §7 category exception (mutually exclusive per
# call). Scalar fan_out_index / branch_name only, matching the embedding pair
# (the lineage chains arrive uniformly across the provider events with proposal
# 0084). query / documents / request_extras / output_results are payload-
# bearing, populated unconditionally; observer-side privacy gates (OTel
# disable_provider_payload, Langfuse equivalents) apply at rendering, symmetric
# with EmbeddingEvent.
@dataclass(frozen=True)
class RerankEvent:
    """A typed rerank provider call event delivered to observers.

    Carries identity, scoping, and outcome data for a successful
    ``RerankProvider.rerank()`` call. Observer code filters by type
    discrimination (``isinstance(event, RerankEvent)``).

    The identity / scoping / request-side fields mirror ``EmbeddingEvent``'s
    convention; the outcome fields are rerank-specific:

    - ``query``: the query string the call was made with; non-nullable,
      populated unconditionally (privacy gating is observer-side at
      rendering).
    - ``documents``: the input documents list; non-nullable, populated
      unconditionally. Same privacy posture as ``query``.
    - ``document_count``: ``len(documents)``; a convenience field.
    - ``top_k``: the caller-supplied ``top_k`` value; ``None`` when the
      caller passed ``None``.
    - ``result_count``: ``len(output_results)``; a convenience field.
    - ``output_results``: the scored results the call returned; populated
      unconditionally on the success event (privacy gating is observer-side
      at rendering).
    - ``response_model`` / ``response_id``: the provider-returned model and
      response identifiers; ``None`` when the provider returned none.
    - ``usage``: the rerank usage record; ``None`` when the call returned
      no usage.
    - ``request_params``: the rerank request parameters the caller supplied
      (e.g. ``return_documents``). Absence-is-meaningful: only supplied keys
      appear; an empty mapping when none.
    - ``request_extras``: the runtime-config extras pass-through bag.
    - ``active_prompt`` / ``active_prompt_group``: prompt-context snapshots
      at rerank-call time; ``None`` outside a binding.
    - ``call_id``: a per-call disambiguator, always present, freshly minted
      per ``rerank()`` call.
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    response_id: str | None
    response_model: str | None
    # RerankUsage is a string-typed forward reference per the TYPE_CHECKING
    # import -- keeps the runtime import direction graph -> retrieval off the
    # module-load path.
    usage: "RerankUsage | None"
    latency_ms: float | None
    query: str
    documents: list[str]
    document_count: int
    top_k: int | None
    result_count: int
    # Spec graph-engine §6 (proposal 0089): the output scored results (sourced
    # from RerankResponse.results at dispatch). Populated unconditionally on the
    # success event -- payload-bearing like query / documents, gated observer-
    # side at the rendering boundary. The §8.4.7 rerank output mapping reads
    # this field. No output field on RerankFailedEvent (no response).
    output_results: list["ScoredDocument"]
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Spec: the failure sibling of RerankEvent (proposal 0060). Dispatched
# whenever RerankProvider.rerank() raises a §7 category exception -- covers
# both the provider-caught path and the pre-send validation raise
# (provider_invalid_request on an empty query / documents list / non-positive
# top_k). Dispatched ALONGSIDE the exception, not in place of it; mutually
# exclusive with RerankEvent on the same call. The response-side fields
# (usage / response_id / response_model / result_count / output_results) are
# absent (no response).
@dataclass(frozen=True)
class RerankFailedEvent:
    """A typed rerank provider call failure event delivered to observers.

    Carries identity, scoping, and failure-context data for a ``rerank()``
    call that raised a retrieval-provider category exception. Observer code
    filters by type discrimination (``isinstance(event, RerankFailedEvent)``).

    The identity / scoping / request-side field set mirrors ``RerankEvent``;
    the response-side fields are absent. Failure-specific fields:

    - ``error_category``: the error category the call raised (one of the
      rerank-applicable provider categories). Always present.
    - ``error_type``: an optional impl-level / vendor-specific type or code;
      ``None`` when unavailable.
    - ``error_message``: a human-readable message; always present (the empty
      string when the exception carried no message).
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    provider: str
    model: str
    latency_ms: float | None
    query: str
    documents: list[str]
    document_count: int
    top_k: int | None
    request_params: Mapping[str, Any]
    request_extras: Mapping[str, Any]
    active_prompt: Any
    active_prompt_group: Any
    call_id: str
    error_category: str
    error_message: str
    error_type: str | None = None
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Spec: realizes pipeline-utilities §6.3 failure-isolation middleware
# (proposal 0050). Emitted by FailureIsolationMiddleware when it
# catches an exception escaping the inner chain and substitutes a
# degraded partial update. A distinct framework-emitted event kind
# (NOT a NodeEvent — does not reuse node_name / namespace / error),
# mirroring the proposal 0040 MetadataAugmentationEvent mechanism:
# enqueued on the engine's serial observer-delivery queue via
# ``current_dispatch()`` and NOT subject to the observer ``phases``
# filter (matches MetadataAugmentationEvent / InvocationStartedEvent /
# InvocationCompletedEvent / LlmCompletionEvent / LlmFailedEvent
# treatment).
@dataclass(frozen=True)
class FailureIsolatedEvent:
    """A failure-isolation event delivered to observers.

    Reports that ``FailureIsolationMiddleware`` caught an exception at
    a node and substituted a degraded partial update for the node's
    output. Observer code filters by type discrimination
    (``isinstance(event, FailureIsolatedEvent)``).

    Field set:

    - ``event_name``: the caller-supplied identifier for this catch
      site, from the middleware's configuration.
    - ``namespace`` / ``attempt_index`` / ``fan_out_index`` /
      ``branch_name``: the wrapped node's lineage identity, surfaced
      for correlation with the node's other events.
    - ``pre_state``: the state the wrapped node received.
    - ``post_state``: the degraded partial update the middleware
      returned in place of the node's output.
    - ``caught_exception``: a :class:`CaughtException` record of the
      caught exception (its derived category / message and the full
      cause ``chain``).
    """

    event_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    pre_state: Any
    post_state: Mapping[str, Any]
    caught_exception: CaughtException


# Spec: realizes graph-engine §6 tool-execution observer events
# (proposal 0063, spec v0.69.0), the success+failure pair the
# tool-call instrumentation scope (``with_tool_call``) dispatches around
# a caller's tool execution. OA observes the execution; it does NOT run,
# select, loop, or feed back tools (llm-provider §1). Mirrors the
# 0049/0058 LLM completion/failure pairing and the §6 dispatch treatment
# (no ``phase`` discriminator, not subject to the ``phases`` filter;
# observers filter by type discrimination). ``arguments`` / ``result``
# are payload-bearing: populated unconditionally here, observer-side
# gating applies at rendering per observability §5.5.4
# (``disable_provider_payload``).
@dataclass(frozen=True)
class ToolCallEvent:
    """A successful tool execution delivered to observers.

    Dispatched by the tool-call instrumentation scope when a caller's
    tool execution returns a result. Observer code filters by type
    discrimination (``isinstance(event, ToolCallEvent)``).

    Field set:

    - ``invocation_id`` / ``correlation_id`` / ``node_name`` /
      ``namespace`` / ``attempt_index`` / ``fan_out_index`` /
      ``branch_name``: the scope-entry identity of the node that ran
      the tool (captured when the scope was entered).
    - ``call_id``: per-execution disambiguator minted when the scope is
      entered. Always present; distinct from ``tool_call_id`` (this is
      OA's own correlation token for the execution).
    - ``tool_name``: the name of the tool / function executed.
    - ``tool_call_id``: the ``ToolCall.id`` of the
      ``LlmCompletionEvent.output_tool_calls`` entry this execution
      satisfies (the linkage back to the requesting LLM call); ``None``
      when the instrumented function did not originate from an LLM tool
      request.
    - ``arguments``: the arguments the tool was invoked with; ``None``
      when the tool takes no arguments. Payload-bearing.
    - ``result``: the tool's return value as the tool produced it
      (pre-serialization, language-idiomatic; opaque to OA).
      Payload-bearing.
    - ``latency_ms``: wall-clock latency measured at the scope boundary,
      in milliseconds; ``None`` when not measured.
    - ``caller_invocation_metadata``: optional snapshot of caller-
      supplied invocation metadata, same opt-in semantics as on
      :class:`LlmCompletionEvent`.
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    call_id: str
    tool_name: str
    tool_call_id: str | None
    arguments: Mapping[str, Any] | None
    result: Any
    latency_ms: float | None
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


# Spec: the failure variant (proposal 0063). Mirrors ToolCallEvent's
# identity / scoping / request-side fields with ``result`` absent and
# two failure fields. DELIBERATELY carries NO ``error_category``: tool
# execution is arbitrary user / third-party code with no closed
# llm-provider §7 failure taxonomy (the departure from LlmFailedEvent /
# EmbeddingFailedEvent). Mutually exclusive with ToolCallEvent per
# execution; dispatched ALONGSIDE the re-raised exception, not in place
# of it.
@dataclass(frozen=True)
class ToolCallFailedEvent:
    """A failed tool execution delivered to observers.

    Dispatched by the tool-call instrumentation scope when a caller's
    tool execution raises (the exception then re-raises out of the
    scope; the event is dispatched alongside it, not in place of it).
    Observer code filters by type discrimination.

    Field set: the identity / scoping / request-side fields of
    :class:`ToolCallEvent` (``tool_name`` / ``tool_call_id`` /
    ``arguments`` / ``latency_ms`` / ``call_id``), the success-only
    ``result`` absent, plus:

    - ``error_type``: the exception class name (e.g. ``"TimeoutError"``)
      or a tool-defined error code; ``None`` when no type is available.
    - ``error_message``: the message from the raised exception; always
      present (empty string when the exception carried no message).

    There is no ``error_category`` (the deliberate departure from the
    provider failure events).
    """

    invocation_id: str
    correlation_id: str | None
    node_name: str
    namespace: tuple[str, ...]
    attempt_index: int
    fan_out_index: int | None
    branch_name: str | None
    call_id: str
    tool_name: str
    tool_call_id: str | None
    arguments: Mapping[str, Any] | None
    latency_ms: float | None
    error_type: str | None
    error_message: str
    caller_invocation_metadata: Mapping[str, AttributeValue] | None = None


__all__ = [
    "EmbeddingEvent",
    "EmbeddingFailedEvent",
    "FailureIsolatedEvent",
    "FanOutEventConfig",
    "InvocationCompletedEvent",
    "InvocationStartedEvent",
    "LlmCompletionEvent",
    "LlmFailedEvent",
    "LlmRetryAttemptEvent",
    "MetadataAugmentationEvent",
    "NodeEvent",
    "ParallelBranchesEventConfig",
    "RerankEvent",
    "RerankFailedEvent",
    "ToolCallEvent",
    "ToolCallFailedEvent",
]
