"""Every engine test must leave nothing running: no child processes, no
multiprocessing children, no non-daemon or voiceqa threads, and the GC heap
not frozen. Later tests (the macOS `say` tests among them) run in the same
pytest process and would inherit the load or the state."""

import gc
import multiprocessing
import threading
import time

import pytest
from tests_support import child_processes

# Time for children and threads that are already stopping to be reaped.
GRACE_SECONDS = 5.0


def _threads() -> set[str]:
    return {
        f"{t.name} (daemon={t.daemon})"
        for t in threading.enumerate()
        if t is not threading.main_thread() and (not t.daemon or t.name.startswith("voiceqa"))
    }


def _snapshot() -> dict:
    return {
        "child processes": set(child_processes()),
        "multiprocessing children": {p.name for p in multiprocessing.active_children()},
        "threads": _threads(),
        "frozen objects": gc.get_freeze_count(),
    }


def _left(before: dict, after: dict) -> dict:
    left = {}
    for key, value in after.items():
        if isinstance(value, set):
            extra = value - before[key]
            if extra:
                left[key] = sorted(extra)
        elif value > before[key]:
            # Only a rise is ours: the last session's unfreeze also clears
            # objects something else froze earlier in the process.
            left[key] = f"{before[key]} -> {value}"
    return left


@pytest.fixture(autouse=True)
def _nothing_left_running():
    before = _snapshot()
    yield
    deadline = time.monotonic() + GRACE_SECONDS
    while True:
        left = _left(before, _snapshot())
        if not left or time.monotonic() > deadline:
            break
        time.sleep(0.1)
    assert not left, f"the test left this behind: {left}"
