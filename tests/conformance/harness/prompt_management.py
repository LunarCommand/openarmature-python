"""Typed YAML models for prompt-management conformance fixtures.

Fixture shape (different from the llm-provider / graph shapes):

- ``backends:`` — list of mock backend specs (each with ``name``,
  optional ``simulate_unavailable``, and a list of ``prompts``).
- ``manager:`` — optional manager composition (a list of backend
  names, in fallback order).
- ``calls:`` — list of operations to drive. Each call has a
  ``target`` (``{backend: <name>}`` for direct backend operations,
  or ``manager`` for manager operations), an ``operation``, inputs,
  optional ``capture_as`` (binds the operation's result to a name
  usable by later calls / final expectations), and optional
  per-call ``expected``.
- ``expected:`` — optional top-level expectation block for
  PromptGroup shape or cross-call result-equivalence assertions
  that need access to ``capture_as`` bindings.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _PermissiveModel(BaseModel):
    """For fixture sub-shapes that vary across fixtures and don't
    warrant a per-shape enumeration."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Backend / manager configuration
# ---------------------------------------------------------------------------


class FixturePromptSpec(_StrictModel):
    name: str
    label: str
    version: str
    template: str
    template_hash: str


class FixtureBackendSpec(_StrictModel):
    name: str
    prompts: list[FixturePromptSpec] = []
    simulate_unavailable: bool = False


class FixtureManagerSpec(_StrictModel):
    backends: list[str]


# ---------------------------------------------------------------------------
# Call targets, operations, and expectations
# ---------------------------------------------------------------------------


class BackendTarget(_StrictModel):
    backend: str


CallTarget = BackendTarget | Literal["manager", "construct_prompt_group"]


class FixtureExpectedRaises(_PermissiveModel):
    category: str
    # Optional extra carries. Fixture 005 surfaces
    # ``description_mentions`` / ``name`` / ``version`` / ``label``
    # here. Permissive on this shape so fixtures evolve;
    # per-backend call-count assertions live on the parent
    # ``FixtureExpectedPerCall`` (see ``secondary_backend_call_count``
    # and ``backend_call_counts`` below), not in ``carries``.
    carries: dict[str, Any] | None = None


class FixtureExpectedPrompt(_PermissiveModel):
    """Per-call ``expected.prompt`` shape (fetch ops)."""


class FixtureExpectedPromptResult(_PermissiveModel):
    """Per-call ``expected.prompt_result`` shape (render / get ops)."""


class FixtureExpectedPerCall(_StrictModel):
    prompt: FixtureExpectedPrompt | None = None
    prompt_result: FixtureExpectedPromptResult | None = None
    raises: FixtureExpectedRaises | None = None
    # Fixture 008's extra: assert how many times the secondary
    # backend's fetch was called. Lives alongside ``raises``.
    secondary_backend_call_count: int | None = None
    # Fixture 009's extra: assert per-backend call counts (named
    # backends → expected call count) after a fetch that exhausts
    # all of them.
    backend_call_counts: dict[str, int] | None = None


class FixtureCall(_StrictModel):
    target: CallTarget
    # ``operation`` is required for fetch / render / get calls. The
    # ``construct_prompt_group`` shape uses the target as the operation
    # indicator (no separate operation field on the call).
    operation: Literal["fetch", "render", "get"] | None = None
    name: str | None = None
    label: str | None = None
    variables: dict[str, Any] | None = None
    # Render-only inputs — either an inline ``fetched_prompt`` (which
    # the harness fetches first, then renders) or a ``fetched_prompt_ref``
    # pointing at an earlier ``capture_as``.
    fetched_prompt: dict[str, str] | None = None
    fetched_prompt_ref: str | None = None
    # construct_prompt_group-only inputs.
    group_name: str | None = None
    members_refs: list[str] | None = None
    capture_as: str | None = None
    expected: FixtureExpectedPerCall | None = None


# ---------------------------------------------------------------------------
# Top-level expected
# ---------------------------------------------------------------------------


class FixtureExpectedPromptGroup(_PermissiveModel):
    """Top-level ``expected.prompt_group`` shape (fixture 011)."""

    of: str
    group_name: str
    member_count: int
    member_order_preserved: bool | None = None
    member_names: list[str] | None = None


class FixtureExpectedResultEquivalence(_PermissiveModel):
    """Top-level ``expected.result_equivalence`` shape (fixtures 006,
    010, 012). Asserts equality across two or more captured results on
    a configurable set of fields."""

    of: list[str]
    fields_must_match: list[str]
    fields_may_differ: list[str] = []
    # fixture 012 — assert two different captures have a DIFFERENT
    # value on a given field.
    fields_must_differ: list[str] = []


class FixtureExpectedTopLevel(_StrictModel):
    prompt_group: FixtureExpectedPromptGroup | None = None
    result_equivalence: FixtureExpectedResultEquivalence | None = None
    # Some fixtures (012) have multiple result-equivalence blocks; keep
    # a plural list-form too. Empty by default.
    result_equivalences: list[FixtureExpectedResultEquivalence] = []
    # Fixture 012's per-pair rendered_hash equality / inequality
    # assertions. Each entry is a 2-element list of capture names; the
    # pair MUST share (resp. differ on) ``rendered_hash``.
    rendered_hash_equal: list[list[str]] = []
    rendered_hash_different: list[list[str]] = []


# ---------------------------------------------------------------------------
# Fixture root
# ---------------------------------------------------------------------------


class PromptManagementFixture(_StrictModel):
    backends: list[FixtureBackendSpec]
    manager: FixtureManagerSpec | None = None
    calls: list[FixtureCall]
    expected: FixtureExpectedTopLevel | None = None
