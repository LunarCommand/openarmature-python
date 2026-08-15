"""The `requires_capability` gate as wired into the fixture runners."""

# Spec basis: conformance-adapter §5.5 (proposal 0116). The gate's own logic is
# covered by tests/unit/test_conformance_capability_gate.py; what is covered here
# is that the RUNNERS consult it, which their fixtures cannot yet demonstrate --
# no fixture at the current spec pin carries `requires_capability` (157 / 158
# arrive with the pin bump). Rather than let the wiring sit unexercised until
# then, these tests inject a gate into a real activated fixture.

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from . import test_observability as otel_runner
from . import test_observability_langfuse as langfuse_runner
from .harness.capabilities import RecognizedSkip

# A two-case activated Langfuse fixture; both cases pass ungated, so any failure
# below is the gate's doing rather than the fixture's.
_FIXTURE = "023-langfuse-generation-rendering"
_CASES = ("generation_rendering", "generation_rendering_truncated")


def _fixture_path() -> Path:
    return langfuse_runner.CONFORMANCE_DIR / f"{_FIXTURE}.yaml"


def _patch_load(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    inject: dict[str, dict[str, Any]],
) -> None:
    """Load the real fixture, then graft `requires_capability` onto named cases."""
    original = module._load

    def _loader(path: Path) -> dict[str, Any]:
        spec = original(path)
        for case in spec.get("cases", []):
            gate = inject.get(case.get("name", ""))
            if gate is not None:
                case["requires_capability"] = gate
        return spec

    monkeypatch.setattr(module, "_load", _loader)


def _spy_run_case(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the names of the cases the runner actually executes."""
    ran: list[str] = []
    original = langfuse_runner._run_case

    async def _recorder(case: Any, **kwargs: Any) -> None:
        ran.append(case.get("name", "<unnamed>"))
        await original(case, **kwargs)

    monkeypatch.setattr(langfuse_runner, "_run_case", _recorder)
    return ran


async def test_mismatched_gate_skips_only_that_case(monkeypatch: pytest.MonkeyPatch) -> None:
    # One case gated out, its sibling still runs -- the in-loop `continue`, not a
    # pytest.skip that would abandon the sibling silently. Asserted on the cases
    # that actually executed, so "it did not raise" cannot stand in for it.
    _patch_load(
        monkeypatch,
        langfuse_runner,
        {_CASES[0]: {"langfuse_bound_provider_detection": False}},
    )
    ran = _spy_run_case(monkeypatch)
    await langfuse_runner.test_langfuse_fixture(_fixture_path())
    assert ran == [_CASES[1]]


async def test_gated_case_is_reported_on_a_passing_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # §5.5 calls a gated-out case a recognized skip, not a silently-omitted one.
    # The run below PASSES -- the surviving sibling carries it -- so without the
    # warning there would be no output anywhere distinguishing this from a run
    # that asserted both cases.
    _patch_load(
        monkeypatch,
        langfuse_runner,
        {_CASES[0]: {"langfuse_bound_provider_detection": False}},
    )
    with pytest.warns(RecognizedSkip, match=_CASES[0]):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_no_skip_is_reported_when_every_case_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-vacuity for the test above: the warning tracks actual exclusions rather
    # than firing on every fixture, so its presence carries information.
    _patch_load(
        monkeypatch,
        langfuse_runner,
        dict.fromkeys(_CASES, {"langfuse_bound_provider_detection": True}),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await langfuse_runner.test_langfuse_fixture(_fixture_path())
    assert [w for w in caught if issubclass(w.category, RecognizedSkip)] == []


async def test_matching_gate_still_runs_the_case(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-vacuity for the test above: with a MATCHING gate both cases run, so the
    # skip there came from the mismatch and not from the injection itself.
    _patch_load(
        monkeypatch,
        langfuse_runner,
        dict.fromkeys(_CASES, {"langfuse_bound_provider_detection": True}),
    )
    ran = _spy_run_case(monkeypatch)
    await langfuse_runner.test_langfuse_fixture(_fixture_path())
    assert ran == list(_CASES)


async def test_gating_every_case_fails_rather_than_passing_vacuously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(
        monkeypatch,
        langfuse_runner,
        dict.fromkeys(_CASES, {"langfuse_bound_provider_detection": False}),
    )
    with pytest.raises(AssertionError, match="asserted nothing"):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_undeclared_capability_reaches_the_runner_as_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_load(monkeypatch, langfuse_runner, {_CASES[0]: {"some_future_capability": True}})
    with pytest.raises(AssertionError, match="does not declare"):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_deferred_sibling_does_not_mask_the_vacuity_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One case deferred, the other gated out: nothing ran, and the check must say
    # so. Counting only gated cases would let a single deferral put any fixture
    # permanently out of its reach.
    monkeypatch.setattr(langfuse_runner, "_DEFERRED_CASES", frozenset({(_FIXTURE, _CASES[0])}))
    _patch_load(
        monkeypatch,
        langfuse_runner,
        {_CASES[1]: {"langfuse_bound_provider_detection": False}},
    )
    with pytest.raises(AssertionError, match="asserted nothing"):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_duplicate_case_names_cannot_hide_an_empty_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # End-to-end guard on how the RUNNER counts. Both cases carry the same name
    # and both are gated out, so they collapse into ONE entry in `excluded`. A
    # count derived from (total - len(excluded)) reads 1 and passes; only
    # counting actual executions reaches 0 and fails. Nothing in the fixture set
    # at this pin has colliding names, so without this the runner's arithmetic is
    # untested against the case it was rewritten for.
    original = langfuse_runner._load

    def _loader(path: Path) -> dict[str, Any]:
        spec = original(path)
        for case in spec["cases"]:
            case["name"] = "same-name"
            case["requires_capability"] = {"langfuse_bound_provider_detection": False}
        return spec

    monkeypatch.setattr(langfuse_runner, "_load", _loader)
    with pytest.raises(AssertionError, match="asserted nothing"):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_wholly_deferred_fixture_does_not_pass_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No capability gate involved at all -- the deferral hook on its own could
    # empty a fixture and report success. A fixture with nothing left to run
    # belongs in the fixture-level deferral table, which skips visibly.
    monkeypatch.setattr(langfuse_runner, "_DEFERRED_CASES", frozenset({(_FIXTURE, name) for name in _CASES}))
    with pytest.raises(AssertionError, match="asserted nothing"):
        await langfuse_runner.test_langfuse_fixture(_fixture_path())


async def test_one_deferral_still_lets_the_sibling_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Non-vacuity for the two above: a deferral is still a deferral, so a fixture
    # with one live case runs it and passes.
    monkeypatch.setattr(langfuse_runner, "_DEFERRED_CASES", frozenset({(_FIXTURE, _CASES[0])}))
    ran = _spy_run_case(monkeypatch)
    await langfuse_runner.test_langfuse_fixture(_fixture_path())
    assert ran == [_CASES[1]]


async def test_hand_built_134_runner_consults_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    # 134 dispatches to a hand-built runner that returns before the generic case
    # loop, so it needs the gate wired separately -- otherwise a gated case there
    # would run the arm belonging to a different adapter class.
    fixture = langfuse_runner._FIXTURE_134
    ran: list[str] = []
    original = langfuse_runner._run_langfuse_134

    async def _recorder(case: Any) -> None:
        ran.append(case.get("name", "<unnamed>"))
        await original(case)

    monkeypatch.setattr(langfuse_runner, "_run_langfuse_134", _recorder)

    path = langfuse_runner.CONFORMANCE_DIR / f"{fixture}.yaml"
    all_cases = [c["name"] for c in langfuse_runner._load(path)["cases"]]
    _patch_load(
        monkeypatch,
        langfuse_runner,
        {all_cases[0]: {"langfuse_bound_provider_detection": False}},
    )
    await langfuse_runner.test_langfuse_fixture(path)
    assert ran == all_cases[1:]


def test_otel_runner_rejects_a_gate_it_does_not_implement() -> None:
    # The OTel runner implements no gate, so an appearance must fail rather than
    # be ignored -- ignoring it would assert an arm meant for another adapter.
    #
    # Called directly rather than through test_observability_fixture: that entry
    # point runs four pytest.skip branches first, and pytest.skip raises Skipped,
    # an OutcomeException deriving from BaseException that pytest.raises does not
    # catch. Driving it end-to-end would therefore report SKIPPED (green) rather
    # than FAILED the day the chosen fixture leaves _SUPPORTED_FIXTURES.
    gated = {"cases": [{"name": "c", "requires_capability": {"langfuse_bound_provider_detection": True}}]}
    with pytest.raises(AssertionError, match="does not implement"):
        otel_runner._reject_unsupported_capability_gate("007-x", gated)

    # The single-case shape (no `cases` key) has to be rejected too.
    single = {"name": "c", "requires_capability": {"langfuse_bound_provider_detection": True}}
    with pytest.raises(AssertionError, match="does not implement"):
        otel_runner._reject_unsupported_capability_gate("007-x", single)

    # Non-vacuity: an ungated spec passes through both shapes.
    otel_runner._reject_unsupported_capability_gate("007-x", {"cases": [{"name": "c"}]})
    otel_runner._reject_unsupported_capability_gate("007-x", {"name": "c"})


def _fixture_files(spec_root: Path) -> list[Path]:
    """Every conformance fixture under a `<capability>/conformance/` tree."""
    return sorted(spec_root.glob("*/conformance/*.yaml"))


def _gated_fixtures(spec_root: Path) -> list[str]:
    """Fixture paths, relative to spec_root, carrying `requires_capability`."""
    found: list[str] = []
    for fixture in _fixture_files(spec_root):
        try:
            loaded: Any = yaml.safe_load(fixture.read_text())
        except yaml.YAMLError:  # pragma: no cover - malformed fixtures are the parse suite's job
            continue
        if not isinstance(loaded, dict):
            continue
        spec = cast("dict[str, Any]", loaded)
        raw = spec.get("cases")
        cases: list[Any] = cast("list[Any]", raw) if isinstance(raw, list) else [spec]
        if any(isinstance(c, dict) and cast("dict[str, Any]", c).get("requires_capability") for c in cases):
            found.append(str(fixture.relative_to(spec_root)))
    return found


def test_gated_fixture_detector_finds_both_case_shapes(tmp_path: Path) -> None:
    # Non-vacuity for the sweep below. At the current pin NO fixture carries
    # `requires_capability` (157/158 arrive with the bump), so the sweep is
    # searching an empty haystack and would pass with a detector that always
    # returned nothing. This pins the detector against a synthetic tree instead,
    # so the sweep starts working the moment a real gated fixture lands.
    (tmp_path / "llm-provider" / "conformance").mkdir(parents=True)
    (tmp_path / "observability" / "conformance").mkdir(parents=True)
    (tmp_path / "checkpointing" / "conformance").mkdir(parents=True)

    (tmp_path / "llm-provider" / "conformance" / "900-multi.yaml").write_text(
        "cases:\n  - name: a\n  - name: b\n    requires_capability: {some_capability: true}\n"
    )
    (tmp_path / "checkpointing" / "conformance" / "901-single.yaml").write_text(
        "name: solo\nrequires_capability: {some_capability: false}\n"
    )
    (tmp_path / "observability" / "conformance" / "902-ungated.yaml").write_text("cases:\n  - name: a\n")

    assert _gated_fixtures(tmp_path) == [
        "checkpointing/conformance/901-single.yaml",
        "llm-provider/conformance/900-multi.yaml",
    ]


def test_every_gated_fixture_belongs_to_a_runner_that_implements_the_gate() -> None:
    # Repo-wide companion to the per-runner rejection. `requires_capability` is a
    # general §5.5 directive and CaseSpec allows extra keys, so a gate landing in
    # an llm-provider / retrieval / checkpoint / prompt-management fixture would
    # parse cleanly and then be dropped on the floor by a runner that never looks
    # for it -- executing an arm meant for a different adapter class, silently.
    # Only the Langfuse runner implements the gate today, so any gated fixture
    # outside observability/ is a wiring gap this must surface.
    #
    # CONFORMANCE_DIR is `<submodule>/spec/observability/conformance`, so the
    # directory holding the capability folders is parents[1] (`<submodule>/spec`),
    # NOT parents[2] (the submodule root, which holds no capability folders at
    # all and globs nothing).
    spec_root = langfuse_runner.CONFORMANCE_DIR.parents[1]
    scanned = _fixture_files(spec_root)
    # Non-vacuity on the sweep's INPUT, not merely on its detector. The detector
    # is pinned separately against a synthetic tree, and that test passing is
    # exactly what made a wrong root look verified: the detector worked while the
    # haystack was empty, so the sweep reported a clean pass over nothing.
    assert langfuse_runner.CONFORMANCE_DIR in {f.parent for f in scanned}, (
        f"the sweep root {spec_root} does not contain the observability conformance directory, "
        f"so it scanned {len(scanned)} fixtures and would report no gated fixtures whatever lands."
    )
    unhandled = [f for f in _gated_fixtures(spec_root) if not f.startswith("observability/")]
    assert not unhandled, (
        f"fixtures {unhandled} carry `requires_capability`, but only the observability Langfuse "
        f"runner implements the gate. Wire it into their runner "
        f"(tests/conformance/harness/capabilities.py) before those fixtures are activated."
    )


# -- invariants-only guard --------------------------------------------------


def test_invariants_only_expected_block_is_rejected() -> None:
    from .harness.capabilities import assert_case_asserts_something

    with pytest.raises(AssertionError, match="only `invariants`"):
        assert_case_asserts_something(
            "999-synthetic",
            "case_with_no_concrete_assertion",
            {"invariants": {"some_absence_claim": True}},
        )


@pytest.mark.parametrize(
    "expected",
    [
        {"span_tree": [], "invariants": {"x": True}},
        {"metrics": [], "invariants": {"x": True}},
        {"langfuse_trace": {}, "invariants": {"x": True}},
        {"invariants": {}},
        {},
    ],
)
def test_a_case_with_any_concrete_directive_is_accepted(expected: dict[str, object]) -> None:
    from .harness.capabilities import assert_case_asserts_something

    assert_case_asserts_something("999-synthetic", "case", expected)


def test_the_guard_is_pointed_at_the_real_corpus() -> None:
    # Non-vacuity for the guard itself. The rejecting test above proves the
    # PREDICATE works; this proves the corpus it is meant to police is actually
    # reachable and non-empty, because a guard verified only against synthetic
    # input is how a sweep ends up globbing an empty directory and reporting a
    # clean result.
    from .test_observability import _fixture_paths

    paths = list(_fixture_paths())
    assert len(paths) > 100, f"expected the observability corpus, found {len(paths)} fixtures"
