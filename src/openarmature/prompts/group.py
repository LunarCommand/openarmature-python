"""PromptGroup; composition pattern for tracing related prompts together."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from .prompt import PromptResult


class PromptGroup(BaseModel):
    """An ordered N≥2 sequence of PromptResult instances under one
    logical observability grouping.

    The group is a structural hint to observability, not a control-flow
    primitive. User code is responsible for executing each member's
    LLM call. The group's contribution is the ``group_name`` that
    observability propagates onto every member call's span so trace
    UIs can render them as one unit.

    Attributes:
        group_name: Stable identifier for this group pattern.
        members: Ordered sequence of at least two PromptResult
            instances. Order matches the application's intended call
            sequence; sequential execution is not required.
    """

    model_config = ConfigDict(extra="forbid")

    group_name: str
    members: list[PromptResult]

    @model_validator(mode="after")
    def _check_min_two_members(self) -> PromptGroup:
        if len(self.members) < 2:
            raise ValueError("prompt group: members MUST contain at least two PromptResult instances")
        return self
