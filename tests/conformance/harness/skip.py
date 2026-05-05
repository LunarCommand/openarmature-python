"""Structured skip-reason values for fixtures whose directives the current
phase doesn't yet support.

Phase 0 ships parsing only — every Phase 1+ runtime test that consumes a
fixture marks itself skipped if the fixture references directives not yet
implemented. The skip-reason is structured (capability + directive list +
phase mapping) so test output makes the next-phase pickup obvious rather
than asking the reader to grep the implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass

# Mapping from spec directive names → the phase that lands their runtime
# implementation. Sourced from the implementation-plan agreement (see
# `_docs/implementation-plan.md` if/when that lands; for now the phase
# numbers are: 1 = engine pair model, 2 = llm-provider, 3 = middleware,
# 4 = fan-out, 5 = OTel observability, 6 = checkpointing).
DIRECTIVE_PHASE: dict[str, int] = {
    # Phase 1 — graph-engine (proposals 0001 + 0002 + 0003 + 0005's §6 revision)
    "observers": 1,
    "phases_subscription": 1,
    "phase": 1,
    "fan_out_index": 1,
    "attempt_index": 1,
    # Phase 2 — llm-provider (proposal 0006)
    "mock_provider": 2,
    "calls_llm": 2,
    "expected_wire_request": 2,
    # Phase 3 — pipeline-utilities middleware (proposal 0004)
    "middleware": 3,
    "flaky": 3,
    "clock_stub": 3,
    # Phase 4 — pipeline-utilities fan-out (proposal 0005)
    "fan_out": 4,
    "flaky_by_index": 4,
    "flaky_instance_only": 4,
    "subgraph_with_idx": 4,
    # Phase 5 — observability (proposal 0007)
    "caller_correlation_id": 5,
    "detached_subgraphs": 5,
    "detached_fan_outs": 5,
    "disable_llm_spans": 5,
    "mock_llm": 5,
    "caller_global_otel_active": 5,
    "invocations": 5,
    "emits_log": 5,
    "also_emits_via_global_tracer": 5,
    # Phase 6 — checkpointing (proposal 0008)
    "checkpointer": 6,
    "first_run_expected_error": 6,
    "saved_record_assertions": 6,
    "resume": 6,
    "populate_checkpointer_via_runs": 6,
    "flaky_per_index": 6,
    "flaky_resume_aware": 6,
    "update_pure_from_state": 6,
}

PHASE_TITLE: dict[int, str] = {
    1: "engine pair-model + fan-out scaffolding",
    2: "llm-provider",
    3: "pipeline-utilities middleware",
    4: "pipeline-utilities fan-out",
    5: "observability (OTel)",
    6: "checkpointing",
}


@dataclass(frozen=True)
class SkipReason:
    """Why a runtime test is skipped at the current phase.

    Render via :meth:`format` for pytest skip-message output. The
    rendered string is action-readable ("phase 1 hasn't shipped yet,
    look at directives X, Y") rather than generic ("not implemented").
    """

    fixture: str  # capability/path-relative identifier, e.g. "graph-engine/012-..."
    current_phase: int
    missing_directives: tuple[str, ...]

    def format(self) -> str:
        """Render the skip message shown in pytest's `-v` output."""
        if not self.missing_directives:
            return f"{self.fixture}: nothing to skip on (current phase {self.current_phase})"
        # Group directives by their landing phase so the message reads
        # "lands in phase 4 (fan-out): [fan_out, flaky_by_index]".
        by_phase: dict[int, list[str]] = {}
        unknown: list[str] = []
        for directive in self.missing_directives:
            phase = DIRECTIVE_PHASE.get(directive)
            if phase is None:
                unknown.append(directive)
            else:
                by_phase.setdefault(phase, []).append(directive)

        parts = [
            f"phase {phase} ({PHASE_TITLE[phase]}): {sorted(directives)}"
            for phase, directives in sorted(by_phase.items())
        ]
        if unknown:
            parts.append(f"unmapped: {sorted(unknown)}")
        return (
            f"{self.fixture}: needs directives not yet supported at phase "
            f"{self.current_phase} — {'; '.join(parts)}"
        )


__all__ = ["DIRECTIVE_PHASE", "PHASE_TITLE", "SkipReason"]
