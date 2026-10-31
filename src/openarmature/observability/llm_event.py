"""Helpers the LLM providers and the observability backends share."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from openarmature.llm.messages import ToolCall


# Spec observability §5.5.5: tool-call serialization.
def serialize_tool_calls(tool_calls: Sequence[ToolCall]) -> list[dict[str, Any]]:
    """Serialize tool calls to ``[{id, name, arguments}, ...]``.

    The single home for the encoding, shared by the input-message
    payload (the provider's ``input.messages`` serialization, where the
    model's tool calls appear inside replayed assistant history) and the
    output tool-call attribute (the OTel observer's gated
    ``openarmature.llm.output.tool_calls``). Lives here rather than in a
    provider or observer module so both sides import one definition and
    the encoding can't drift between them.
    """
    return [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls]


def _token_budget_evaluations(token_budget: Any, usage: Any) -> list[dict[str, Any]]:
    # §5.5.15 / §11.2 (proposal 0083): the per-bound token-budget evaluation
    # shared by the OTel span attrs, the metric recorder, and the Langfuse
    # WARNING / §7 log surfaces. Returns one entry per DECLARED bound that also
    # has a usable actual to compare against, so a bound whose count the provider
    # did not report (None) is omitted -- mirroring the token.usage gate rather
    # than coercing a missing count to 0. A declared bound of 0 IS evaluated: the
    # exceeded test is a strict ``actual > max``, so a 0 bound is exceeded by any
    # positive usage per §5.5.15; only the utilization ratio is undefined for a 0
    # denominator, and the metric recorder skips that one sample. Empty when there
    # is no budget, no usage, or no bound with a reported count; the caller then
    # leaves the exceeded signal absent (not false).
    if token_budget is None or usage is None:
        return []
    evaluations: list[dict[str, Any]] = []
    input_max = getattr(token_budget, "input_max_tokens", None)
    if input_max is not None and usage.prompt_tokens is not None:
        evaluations.append({"kind": "input", "actual": usage.prompt_tokens, "max": input_max})
    total_max = getattr(token_budget, "total_max_tokens", None)
    if total_max is not None:
        if usage.total_tokens is not None:
            total_actual = usage.total_tokens
        elif usage.prompt_tokens is not None and usage.completion_tokens is not None:
            total_actual = usage.prompt_tokens + usage.completion_tokens
        else:
            total_actual = None
        if total_actual is not None:
            evaluations.append({"kind": "total", "actual": total_actual, "max": total_max})
    return evaluations


__all__ = [
    "_token_budget_evaluations",
    "serialize_tool_calls",
]
