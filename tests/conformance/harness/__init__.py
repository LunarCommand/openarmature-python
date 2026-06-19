"""Conformance fixture harness — typed parsing for the four spec capabilities.

Every fixture under
``openarmature-spec/spec/<capability>/conformance/`` lands as a typed pydantic
config. Later stages add runtime interpretation under ``harness/runtime/``;
they never re-touch parsing.

Public surface:

- :func:`loader.load_fixture` — parse one YAML path into a typed fixture.
- :func:`loader.discover_fixtures` — auto-discover fixture paths across the
  four capability directories on the spec submodule.
- :class:`fixtures.Fixture` — the root discriminated union
  (``LlmProviderFixture | CasesFixture | GraphFixture``).
- :class:`skip.SkipReason` — structured "fixture needs directives X, current
  phase doesn't support them" used by the test runner to skip cleanly.
"""

from .fixtures import (
    CasesFixture,
    Fixture,
    GraphFixture,
    LlmProviderFixture,
)
from .loader import discover_fixtures, load_fixture
from .skip import SkipReason
from .wire import (
    assert_error_carries,
    assert_response_format_absent,
    assert_system_references_schema,
    assert_tool_choice_absent,
    match_wire_body,
    request_body,
)

__all__ = [
    "CasesFixture",
    "Fixture",
    "GraphFixture",
    "LlmProviderFixture",
    "SkipReason",
    "assert_error_carries",
    "assert_response_format_absent",
    "assert_system_references_schema",
    "assert_tool_choice_absent",
    "discover_fixtures",
    "load_fixture",
    "match_wire_body",
    "request_body",
]
