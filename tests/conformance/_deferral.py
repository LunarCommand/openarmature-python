"""Shared deferral plumbing for conformance test files.

Each fixture-level conformance test (``test_fixture_parsing``,
``test_llm_provider``, ``test_prompt_management``) maintains its own
``_DEFERRED_FIXTURES`` dict mapping fixture IDs to human-readable
reasons. Entries come and go as proposals land in their implementing
PRs.

This helper centralizes the skip-call site so the format
(``f"{fixture_id}: {reason}"``) stays consistent across files and so
future tweaks to the message shape land in one place.

The case-level skip inside ``_run_fixture_028`` in
``test_observability.py`` deliberately doesn't use this helper — it
executes ``continue`` inside an async loop iterating sub-cases of a
single fixture rather than skipping the outer pytest test.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest


def skip_if_deferred(fixture_id: str, deferrals: Mapping[str, str]) -> None:
    """Raise :func:`pytest.skip` if ``fixture_id`` appears in ``deferrals``.

    Standard call site at the top of a conformance test that processes
    one fixture per pytest test::

        @pytest.mark.parametrize(...)
        def test_xxx_fixture(fixture_path: Path) -> None:
            fixture_id = _fixture_id(fixture_path)
            skip_if_deferred(fixture_id, _DEFERRED_FIXTURES)
            # ... rest of the test ...
    """
    if fixture_id in deferrals:
        pytest.skip(f"{fixture_id}: {deferrals[fixture_id]}")
