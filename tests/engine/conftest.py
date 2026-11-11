"""Every engine test must leave nothing running: no child processes, no
multiprocessing children, no non-daemon or voiceqa threads, and the GC heap
not frozen. Later tests (the macOS `say` tests among them) run in the same
pytest process and would inherit the load or the state."""

import pytest
from tests_support import nothing_left_running


@pytest.fixture(autouse=True)
def _nothing_left_running():
    with nothing_left_running():
        yield
