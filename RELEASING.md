# Releasing

How to cut a release of `openarmature`. For maintainers.

Releases go through TestPyPI before PyPI **by convention**. The
workflow (`.github/workflows/release.yml`) is tag-driven and dispatches
by tag name; it does not enforce that an rc preceded a real-release
tag. Pushing `v0.7.0` directly publishes to PyPI without consulting
any prior `-rc` tag, so the rc-first flow is a maintainer-side
discipline carried by the pre-release checklist below.

## The release path: rc first, then real

A release happens in two tag steps:

1. **`vX.Y.Z-rc1`** publishes to **TestPyPI only**. No PyPI upload, no
   GitHub Release. Use the rc to verify the artifact installs, the
   examples still run, and the docs reflect what shipped.
2. **`vX.Y.Z`** publishes to **PyPI** and creates a **GitHub Release**
   with auto-generated notes. Fires only after the rc is good.

Tag dispatch is by name:

- Tags containing `-rc` route to TestPyPI.
- Tags with no `-` suffix route to PyPI + GitHub Release.
- Any other suffix (`-beta`, `-alpha`, `-dev`, ...) is a no-op by design;
  a typo in the rc suffix cannot accidentally hit PyPI.

The workflow validates that `pyproject.toml`'s `version` field matches
the tag (modulo PEP 440 normalization: `0.7.0-rc1` ≡ `0.7.0rc1`).
Mismatches fail the test job before any publishing step runs.

## Pre-release checklist

Run through this before tagging the first rc. Everything here is human
judgment; the workflow can't catch a stale doc reference or a missing
changelog entry.

- [ ] **`CHANGELOG.md` is current.** Every commit since the previous
      release that changed user-visible behavior is reflected in the
      upcoming version's section. The date matches the day the rc tag
      is pushed (refresh it again at the real-release step).
- [ ] **`conformance.toml` is current.** Any proposal whose impl
      landed in this cycle has its `[proposals."NNNN"]` entry — either
      newly added (set `since` to the version about to ship) or
      adjusted (e.g., `not-yet` → `implemented`). If the pinned spec
      submodule was bumped, also bump `[manifest].spec_pin` to the new
      tag. The CI guard at `scripts/check_conformance_manifest.py`
      enforces structural consistency, but it can't check that
      semantic status reflects reality — read the diff manually.
- [ ] **Docs sweep for stale references.** For each behavior change in
      the upcoming release, grep the docs for the old wording, file
      paths, and flag descriptions; reconcile in the same PR as the
      version bump. Common spots: `README.md`, `docs/concepts/*`,
      `docs/getting-started/index.md`, `docs/reference/index.md`,
      in-code help text.
- [ ] **`pyproject.toml` version pinned to the tag we're about to push.**
      The release workflow validates that the pyproject version equals
      the tag after PEP 440 normalization. Concretely:
      - For an rc tag like `v0.7.0-rc1`, set `project.version =
        "0.7.0rc1"` (or `"0.7.0-rc1"`; both normalize to `0.7.0rc1`).
      - For the real-release tag `v0.7.0`, set `project.version =
        "0.7.0"`.
      The rc and real-release pyproject bumps are SEPARATE COMMITS —
      one before each tag — because the normalized forms differ.
      Also update `src/openarmature/__init__.py`'s `__version__` and
      `tests/test_smoke.py`'s version assertion in the same commit.
- [ ] **Branch state.** On `main`, clean working tree, latest pulled.
      Release tags should point at commits already on `main`.
- [ ] **CI is green on `main`.** The release workflow's `test` job
      re-runs the suite, but a red `main` is a sign to investigate
      before tagging anything.

Land the version bump + changelog refresh as one commit before tagging.
Convention: `chore(release): vX.Y.Z`.

## Tagging the rc

After the prep commit is on `main`:

```bash
git tag v0.7.0-rc1
git push origin v0.7.0-rc1
```

Watch the workflow run at <https://github.com/LunarCommand/openarmature-python/actions>.
On success, the artifact lands at <https://test.pypi.org/project/openarmature/>.

The `testpypi` GitHub Environment owns the OIDC trust for the upload;
no secrets to plumb manually.

## Verifying the rc

Install from TestPyPI into a fresh venv and exercise the package the
way a downstream user would.

```bash
python -m venv /tmp/oa-rc-verify
source /tmp/oa-rc-verify/bin/activate

# --extra-index-url is required: TestPyPI does not mirror the
# transitive dependency graph, so dependencies pull from real PyPI.
pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  'openarmature==0.7.0rc1'

python -c "import openarmature; print(openarmature.__version__)"
```

Minimum smoke set:

- [ ] Version string matches the rc tag.
- [ ] At least one example runs to completion against a real LLM
      endpoint (`examples/00-hello-world/main.py` is the quickest).
- [ ] The optional `[otel]` extra installs cleanly and
      `import openarmature.observability.otel` succeeds.
- [ ] If any docs changed in this release, the live docs site
      (<https://openarmature.ai/>) builds without warnings.

If any of these fail, see *Iterating on an rc* below. Do not proceed
to the real-release tag with an unverified rc.

## Tagging the real release

After the rc is verified, prepare the real-release bump as a
separate PR. Update:

- `pyproject.toml`: `version = "0.7.0"`
- `src/openarmature/__init__.py`: `__version__ = "0.7.0"`
- `tests/test_smoke.py`: the version assertion
- `CHANGELOG.md`: refresh the date if it drifted from today

Commit as `chore(release): v0.7.0`, open a PR, and merge once CI is
green. Then from a freshly-pulled `main`:

```bash
git tag v0.7.0
git push origin v0.7.0
```

The workflow runs the same test job, builds the artifact, publishes to
PyPI through the `pypi` GitHub Environment, and creates a GitHub
Release with notes auto-generated from commits since the previous tag.

The `pypi` environment is the right place to attach a **required
reviewers** protection rule so the publish step pauses for explicit
manual approval before any real-PyPI upload. Configure it in repo
settings under *Environments → pypi → Required reviewers*.

After the workflow finishes:

- [ ] The new version appears on <https://pypi.org/project/openarmature/>.
- [ ] A GitHub Release exists at
      <https://github.com/LunarCommand/openarmature-python/releases>
      with the wheel and sdist attached.
- [ ] `pip install openarmature` in a fresh venv resolves the new
      version.

## Iterating on an rc

If the rc reveals an issue **after it has published to TestPyPI**,
never move the existing rc tag. PyPI and TestPyPI treat versions as
immutable; bump the rc counter instead. Each rc bump is its own PR
because the pyproject version has to track the new tag. Fix the
bug, then update:

- `pyproject.toml`: `version = "0.7.0rc2"`
- `src/openarmature/__init__.py`: `__version__ = "0.7.0rc2"`
- `tests/test_smoke.py`: the version assertion

Commit as `chore(release): v0.7.0-rc2`, open a PR, and merge once
CI is green. Then from a freshly-pulled `main`:

```bash
git tag v0.7.0-rc2
git push origin v0.7.0-rc2
```

Repeat verification against the new rc. Two or three rc iterations is
fine. If the same issue keeps recurring, that's a signal to step back
and address the design rather than spin more rc tags.

If an rc tag fails the release workflow's pre-publish checks (e.g.,
the version-match validator) so that nothing was uploaded to
TestPyPI, the tag is recoverable: nothing is immutable on PyPI yet.
Land the fix on `main`, delete the tag from origin (`git push origin
:refs/tags/v0.7.0-rcN`), and re-tag the fix commit. Only do this when
you're certain the workflow did not reach a publish step.

## Rollback

PyPI does not allow re-uploading the same version. If a real release
ships and turns out to be broken:

1. **Yank the version** via the PyPI web UI
   (<https://pypi.org/manage/project/openarmature/release/X.Y.Z/>).
   Yanking marks the version as not-installable-by-default; existing
   pinned dependencies still resolve, but a fresh install skips it.
2. **Cut a patch.** Fix the bug, run through the full rc → real cycle
   for `X.Y.(Z+1)`. The yanked version stays in place as a historical
   record; the new patch supersedes it.

Do not try to delete a release. Yanking + patching is the supported
path and what downstream tooling expects.

## Reference

- `.github/workflows/release.yml` — the release workflow; authoritative
  on what happens when each tag shape is pushed.
- `CHANGELOG.md` — release notes go here, in
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
- `pyproject.toml` — `project.version` must match the tag (normalized
  per PEP 440).
- GitHub Environments — `testpypi` and `pypi` own the OIDC trust to
  the respective indexes. Configure required-reviewer rules on `pypi`
  for an extra approval gate before real publishes.
