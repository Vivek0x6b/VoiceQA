"""End-to-end check: run `voiceqa agent` and `voiceqa call` as real subprocesses,
through `python -m voiceqa`, and inspect what lands on disk and on stdout.

Everything else in this test suite calls into voiceqa from the same process.
This one is the only check that the console entry point actually works, that
the two commands agree on the wire, and that a full call survives going
through two separate OS processes and a real socket.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from voiceqa.audio.analysis import first_loud_frame
from voiceqa.audio.wav import read_wav
from voiceqa.telephony.media import SAMPLE_RATE

AGENT_START_TIMEOUT = 15.0
CALL_TIMEOUT = 30.0
_LISTENING_RE = re.compile(r"agent listening on ([^:\s]+):(\d+)")


def _subprocess_env() -> dict[str, str]:
    # Unbuffered, so the agent's stdout line arrives as soon as it is
    # printed instead of waiting for a full block (CI pipes stdout, which
    # is enough to make Python's stdout buffering switch to block mode).
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _start_agent() -> tuple[subprocess.Popen, int]:
    """Starts `voiceqa agent` on an OS-assigned port and returns (process, port).

    Reads stdout on a background thread, since a blocking readline() on the
    main thread has no portable way to time out on Windows.
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "voiceqa", "agent", "--host", "127.0.0.1", "--port", "0"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        env=_subprocess_env(),
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    deadline = time.monotonic() + AGENT_START_TIMEOUT
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            proc.terminate()
            raise TimeoutError("voiceqa agent did not print its listening line in time")
        try:
            line = lines.get(timeout=remaining)
        except queue.Empty:
            continue
        if line is None:
            proc.terminate()
            raise RuntimeError("voiceqa agent exited before it started listening")
        match = _LISTENING_RE.search(line)
        if match:
            return proc, int(match.group(2))


def _dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    centered = samples.astype(np.float64)
    centered -= centered.mean()
    windowed = centered * np.hanning(len(centered))
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(len(centered), d=1 / sample_rate)
    return float(freqs[int(np.argmax(spectrum))])


def test_python_dash_m_voiceqa_help_works():
    """python -m voiceqa must work with no console script on PATH (see __main__.py)."""
    result = subprocess.run(
        [sys.executable, "-m", "voiceqa", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "voiceqa" in result.stdout.lower()


def test_call_against_the_agent_subprocess_end_to_end(tmp_path):
    agent_proc, port = _start_agent()
    try:
        record_dir = tmp_path / "run"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiceqa",
                "call",
                f"sip:agent@127.0.0.1:{port}",
                "--duration",
                "3",
                "--record-dir",
                str(record_dir),
            ],
            capture_output=True,
            text=True,
            env=_subprocess_env(),
            timeout=CALL_TIMEOUT,
        )
    finally:
        agent_proc.terminate()
        agent_proc.wait(timeout=5)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "answered      yes" in result.stdout

    for name in ("caller.wav", "agent.wav", "mix.wav", "events.json"):
        assert (record_dir / name).exists()

    caller, caller_rate = read_wav(record_dir / "caller.wav")
    agent_audio, agent_rate = read_wav(record_dir / "agent.wav")
    assert caller_rate == SAMPLE_RATE
    assert agent_rate == SAMPLE_RATE
    assert len(caller) == len(agent_audio)

    caller_onset = first_loud_frame(record_dir / "caller.wav")
    agent_onset = first_loud_frame(record_dir / "agent.wav")
    assert caller_onset is not None
    assert agent_onset is not None
    onset_delay = (agent_onset - caller_onset) / SAMPLE_RATE
    assert 0 < onset_delay < 0.5

    loud_part = agent_audio[agent_onset : agent_onset + SAMPLE_RATE]
    assert _dominant_frequency(loud_part, SAMPLE_RATE) == pytest.approx(440, abs=5)

    events = json.loads((record_dir / "events.json").read_text())["events"]
    names = [e["event"] for e in events]
    assert names[0] == "invite_sent"
    assert names[-1] == "bye_sent"
    order = {name: i for i, name in enumerate(names)}
    assert order["ringing"] < order["answered"] < order["first_rtp_out"]
    assert order["answered"] < order["first_rtp_in"]

    latency_match = re.search(r"first audio\s+([\d.]+) s after answer", result.stdout)
    assert latency_match is not None, result.stdout
    assert float(latency_match.group(1)) < 1.5
