from __future__ import annotations

import pytest


# The nested read check judges what the runners read as they ran, so the tests
# that use it go after every other test in the session, `tests/unit` included.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    last = [item for item in items if "unread_nested" in getattr(item, "fixturenames", ())]
    if last:
        moved = set(map(id, last))
        items[:] = [item for item in items if id(item) not in moved] + last
