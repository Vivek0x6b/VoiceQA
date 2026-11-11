"""The RTP sender keeps its slots with Silero running in the call process.

Two child processes (jitter_child.py) place calls at the same time, one
with Silero VAD and one with no VAD, and record how late each sent frame
went out against its slot. Running them side by side means a stall of the
whole machine (a busy CI runner) hits both, while anything Silero does to
its own process's event loop shows only in the Silero calls.

The sender's contract: a frame more than MAX_LATE_FRAMES late makes it skip
slots. Each such catch-up is a stall.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voiceqa.telephony.rtp.session import FRAME_SECONDS

pytestmark = [pytest.mark.model, pytest.mark.usefixtures("silero_extra")]

CHILD = Path(__file__).with_name("jitter_child.py")
CALLS = 3
CALL_SECONDS = 4.0


def stalls(dues):
    """(position in seconds from the first slot, slots skipped) per catch-up."""
    steps = np.round(np.diff(dues) / FRAME_SECONDS).astype(int)
    return [(round(dues[i] - dues[0], 2), int(s - 1)) for i, s in enumerate(steps) if s > 1]


def run_side_by_side(tmp_path):
    procs = {
        mode: subprocess.Popen(
            [sys.executable, str(CHILD), mode, str(CALLS), str(CALL_SECONDS), str(tmp_path / mode)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        for mode in ("none", "silero")
    }
    try:
        for mode, proc in procs.items():
            assert proc.stdout.readline().strip() == "ready", f"{mode} child did not start"
        for proc in procs.values():
            proc.stdin.write("go\n")
            proc.stdin.flush()
        out = {}
        for mode, proc in procs.items():
            stdout, _ = proc.communicate(timeout=120)
            assert proc.returncode == 0, f"{mode} child failed"
            out[mode] = json.loads(stdout.strip().splitlines()[-1])
        return out
    finally:
        for proc in procs.values():
            if proc.poll() is None:
                proc.kill()


def test_silero_vad_does_not_make_the_rtp_sender_late(tmp_path):
    runs = run_side_by_side(tmp_path)

    # Silero really ran on the agent's speech, and kept up.
    assert all(call["windows"] > 50 for call in runs["silero"])
    assert all(call["dropped"] == 0 for call in runs["silero"])

    stats = {}
    for mode, calls in runs.items():
        late = np.concatenate([call["lateness"] for call in calls]) * 1000
        stats[mode] = {
            "frames": len(late),
            "p50": round(float(np.percentile(late, 50)), 2),
            "p95": round(float(np.percentile(late, 95)), 2),
            "max": round(float(late.max()), 2),
            "stalls": [stalls(call["dues"]) for call in calls],
        }
    print(stats)

    silero_stalls = sum(len(s) for s in stats["silero"]["stalls"])
    none_stalls = sum(len(s) for s in stats["none"]["stalls"])
    # A stall Silero caused would come with every call (a start-up cost) or
    # recur during calls, so it shows more than once over CALLS calls. One
    # extra allows for the scheduler starving one process and not the other
    # once.
    assert silero_stalls <= none_stalls + 1, stats
    # Same lateness, within half a frame at the tail. Timer resolution on
    # Windows before 3.13 is 15.6 ms, which both share.
    half_frame_ms = FRAME_SECONDS * 1000 / 2
    assert stats["silero"]["p95"] <= stats["none"]["p95"] + half_frame_ms, stats
