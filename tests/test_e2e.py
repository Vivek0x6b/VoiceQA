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
import yaml

from voiceqa.audio.analysis import first_loud_frame
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.models import Transcript
from voiceqa.speech.wer import wer
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


# Spoken by the OS engine on Windows (sapi) and macOS (say).
OS_TTS_TEXT = "Hello, this is a test of the local voice lab. Please repeat after me."
# The LibriVox utterance used as a WAV prompt on Linux, which has no OS engine.
PROMPT_UTTERANCE = 1
TRANSCRIBE_CALL_TIMEOUT = 180.0
# Measured with tiny.en on Windows: 0.0 on both sides for sapi and for this
# WAV prompt. The bound allows 3 wrong words in the 11-word prompt and 5 in
# the OS sentence, for other voices and CPUs, and still fails when a side
# is missing, silent or garbled.
MAX_WER = 0.35


def _speech_args(tmp_path) -> tuple[list[str], str]:
    """The voiceqa call options that make the caller speak, and the text spoken."""
    if sys.platform == "win32":
        return ["--say", OS_TTS_TEXT, "--tts", "sapi"], OS_TTS_TEXT
    if sys.platform == "darwin":
        return ["--say", OS_TTS_TEXT, "--tts", "say"], OS_TTS_TEXT
    clip = load_bench_clip(default_clip_dir())
    u = clip.utterances[PROMPT_UTTERANCE]
    speech = clip.samples[round(u.start * clip.rate) : round(u.end * clip.rate)]
    write_wav(tmp_path / "prompt.wav", speech, clip.rate)
    index = tmp_path / "prompts.yaml"
    index.write_text(yaml.safe_dump({u.text: "prompt.wav"}), encoding="utf-8")
    return ["--say", u.text, "--tts", "wav", "--prompts", str(index)], u.text


@pytest.mark.model
def test_spoken_call_to_the_echo_agent_is_transcribed_on_both_sides(test_model, tmp_path):
    say_args, text = _speech_args(tmp_path)
    env = _subprocess_env()
    # A fresh TTS cache, so the OS engine really runs.
    env["VOICEQA_CACHE_DIR"] = str(tmp_path / "cache")
    record_dir = tmp_path / "run"
    agent_proc, port = _start_agent()
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "voiceqa",
                "call",
                f"sip:agent@127.0.0.1:{port}",
                *say_args,
                "--duration",
                "8",
                "--record-dir",
                str(record_dir),
                "--transcribe",
                "--stt-model",
                test_model,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=TRANSCRIBE_CALL_TIMEOUT,
        )
    finally:
        agent_proc.kill()
        agent_proc.wait(timeout=5)

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "answered      yes" in result.stdout, output
    transcript_path = record_dir / "transcript.json"
    assert transcript_path.is_file(), output

    transcript = Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    for speaker in ("caller", "agent"):
        heard = " ".join(s.text for s in transcript.segments if s.speaker == speaker)
        assert heard, f"no {speaker} segments: {transcript.model_dump_json()}"
        assert wer(text, heard) <= MAX_WER, f"{speaker} heard {heard!r}, said {text!r}"
