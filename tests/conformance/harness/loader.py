"""Discovery + parsing for spec conformance fixtures.

Two entry points:

- :func:`discover_fixtures` walks the four capability directories under the
  pinned ``openarmature-spec`` submodule and yields ``(capability, path)``
  pairs sorted by capability then filename. Used by parametrized pytest
  collection.

- :func:`load_fixture` parses one YAML file into a typed
  :data:`fixtures.Fixture` (one of the three discriminated variants).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .fixtures import Fixture

# All four capability directories the spec defines under ``spec/``. Keep this
# list in sync with the spec repo's top-level ``spec/<capability>/``
# layout. Adding a fifth capability is a "knob" change — the discovery and
# parsing already work; you'd only need to extend the per-capability
# expected-block models in :mod:`expectations`.
CAPABILITIES: tuple[str, ...] = (
    "graph-engine",
    "llm-provider",
    "pipeline-utilities",
    "observability",
    "prompt-management",
)

CONFORMANCE_ROOT = Path(__file__).resolve().parents[3] / "openarmature-spec" / "spec"

# pydantic v2 needs an adapter to validate against an Annotated/Union type
# that isn't itself a BaseModel subclass. Built once, reused per call.
_FIXTURE_ADAPTER: TypeAdapter[Fixture] = TypeAdapter(Fixture)


def discover_fixtures() -> Iterator[tuple[str, Path]]:
    """Yield ``(capability, fixture_path)`` for every ``NNN-*.yaml`` under
    each capability's ``conformance/`` directory, sorted deterministically
    so pytest parametrization IDs are stable across runs.
    """
    for capability in CAPABILITIES:
        conformance_dir = CONFORMANCE_ROOT / capability / "conformance"
        if not conformance_dir.is_dir():
            continue
        for path in sorted(conformance_dir.glob("[0-9][0-9][0-9]-*.yaml")):
            yield capability, path


def load_fixture(path: Path) -> Fixture:
    """Parse a fixture YAML into one of the three typed variants.

    The discriminator inspects top-level keys to pick
    :class:`LlmProviderFixture` (when ``mock_provider`` is present),
    :class:`CasesFixture` (when ``cases`` is present and no
    ``mock_provider``), or :class:`GraphFixture` (default).

    Raises ``pydantic.ValidationError`` on schema violations — the
    ``extra="forbid"`` config in :mod:`fixtures` makes any unknown
    top-level key fail loudly, which is how we catch the spec adding
    directives we haven't modelled yet.
    """
    with path.open() as f:
        raw = yaml.safe_load(f)
    return _FIXTURE_ADAPTER.validate_python(raw)


__all__ = ["CAPABILITIES", "CONFORMANCE_ROOT", "discover_fixtures", "load_fixture"]
