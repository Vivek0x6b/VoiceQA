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
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml
from tests_support import machine_state, say_probe

from voiceqa.audio.analysis import first_loud_frame
from voiceqa.audio.dsp import resample
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.models import Transcript
from voiceqa.speech.wer import wer
from voiceqa.telephony.media import SAMPLE_RATE

# Real subprocesses with start-up and call timeouts.
pytestmark = pytest.mark.realtime

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


def _start_agent(*extra_args: str) -> tuple[subprocess.Popen, int]:
    """Starts `voiceqa agent` on an OS-assigned port and returns (process, port).

    Reads stdout on a background thread, since a blocking readline() on the
    main thread has no portable way to time out on Windows.
    """
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "voiceqa",
            "agent",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            *extra_args,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
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
        encoding="utf-8",
        errors="replace",
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
            encoding="utf-8",
            errors="replace",
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
# WAV prompt. The bound allows 3 wrong words in the 11-word prompt and 4 in
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
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=TRANSCRIBE_CALL_TIMEOUT,
        )
    finally:
        agent_proc.kill()
        agent_proc.wait(timeout=5)

    output = result.stdout + result.stderr
    if result.returncode != 0 and sys.platform == "darwin":
        # This failed on the macOS 3.11 CI job, where only the first line of
        # the failure can be read, so the details go on that line.
        output = " | ".join(output.strip().splitlines())
        output = f"{output} | {say_probe(text)}; {machine_state()}"
    assert result.returncode == 0, output
    assert "answered      yes" in result.stdout, output
    transcript_path = record_dir / "transcript.json"
    assert transcript_path.is_file(), output

    transcript = Transcript.model_validate_json(transcript_path.read_text(encoding="utf-8"))
    for speaker in ("caller", "agent"):
        heard = " ".join(s.text for s in transcript.segments if s.speaker == speaker)
        assert heard, f"no {speaker} segments: {transcript.model_dump_json()}"
        assert wer(text, heard) <= MAX_WER, f"{speaker} heard {heard!r}, said {text!r}"


# `voiceqa run` against `voiceqa agent --script`, both as subprocesses. The
# caller lines and the agent's replies are utterances of the LibriVox bench
# clip, so the run is the same on every OS and needs no TTS engine. What
# tiny.en heard for each cut on Windows, plain and through PCMU:
#   1 "Suddenly, she whirled from the window and stood before the glass."
#   2 "Her eyes were shining brilliantly, but her face had lost its color
#      within 20 seconds."
#   3 "rapidly she pulled down her hair and let it fall through its full length."
#   0 "Stella, the in slender, had mastered the art."
# Utterances 1 and 3 are the caller, 2 and 0 the agent's replies.
CALLER_UTTERANCES = (1, 3)
REPLY_UTTERANCES = (2, 0)
# The energy VAD takes the first frame as its noise floor, so speech in the
# very first frame of the call would not count as speech.
LEAD_SILENCE_S = 0.3
RUN_TIMEOUT = 180.0

SCENARIO = """\
name: librivox reading
defaults:
  timeout: 20s
  # The replies pause at commas; one turn must not end there.
  end_silence_ms: 1200
steps:
  - label: whirled
    say: {wav: caller1.wav, text: "%(caller1)s"}
    expect:
      any_of: ["shining brilliantly", "lost its color"]
  - label: hair
    say: {wav: caller2.wav, text: "%(caller2)s"}
    expect:
      %(last_expect)s
    complete: true
assert: [call_connected, agent_responded, task_completed]
"""
PASSING_EXPECT = 'fuzzy: {text: "had mastered the art", min: 0.75}'
# Nothing in utterance 0 sounds like it.
WRONG_WORD = "violin"
FAILING_EXPECT = f"contains: {WRONG_WORD}"


def _utterance_wav(clip, index: int, path) -> str:
    u = clip.utterances[index]
    speech = clip.samples[round(u.start * clip.rate) : round(u.end * clip.rate)]
    speech = resample(speech, clip.rate, SAMPLE_RATE)
    lead = np.zeros(round(LEAD_SILENCE_S * SAMPLE_RATE), dtype=np.int16)
    write_wav(path, np.concatenate([lead, speech.astype(np.int16)]), SAMPLE_RATE)
    return u.text


@pytest.fixture(scope="module")
def reading(tmp_path_factory):
    """The scenario folder (caller WAVs and the caller texts) and a scripted
    agent subprocess that replies with the other two utterances."""
    folder = tmp_path_factory.mktemp("reading")
    clip = load_bench_clip(default_clip_dir())
    texts = [
        _utterance_wav(clip, index, folder / f"caller{n}.wav")
        for n, index in enumerate(CALLER_UTTERANCES, 1)
    ]
    replies = []
    for n, index in enumerate(REPLY_UTTERANCES, 1):
        _utterance_wav(clip, index, folder / f"reply{n}.wav")
        replies.append({"wav": f"reply{n}.wav", "delay_ms": 400})
    script = folder / "replies.yaml"
    script.write_text(
        yaml.safe_dump({"replies": replies, "then": "silence"}, sort_keys=False),
        encoding="utf-8",
    )
    agent_proc, port = _start_agent("--script", str(script), "--end-silence-ms", "1000")
    try:
        yield folder, texts, port
    finally:
        agent_proc.kill()
        agent_proc.wait(timeout=5)


def _run_scenario(reading, tmp_path, model: str, last_expect: str):
    folder, texts, port = reading
    scenario = folder / f"scenario-{tmp_path.name}.yaml"
    scenario.write_text(
        SCENARIO % {"caller1": texts[0], "caller2": texts[1], "last_expect": last_expect},
        encoding="utf-8",
    )
    env = _subprocess_env()
    env["VOICEQA_DATA_DIR"] = str(tmp_path / "data")
    out = tmp_path / "runs"
    junit = tmp_path / "out.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "voiceqa",
            "run",
            str(scenario),
            "--target",
            f"sip:agent@127.0.0.1:{port}",
            "--junit",
            str(junit),
            "--stt-model",
            model,
            "--vad",
            "energy",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=RUN_TIMEOUT,
    )
    runs = subprocess.run(
        [sys.executable, "-m", "voiceqa", "runs", "list"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=30,
    )
    assert runs.returncode == 0, runs.stdout + runs.stderr
    (run_dir,) = list(out.iterdir())
    return result, run_dir, ET.parse(junit).getroot(), runs.stdout.strip().splitlines()


@pytest.mark.model
def test_a_scenario_run_against_the_scripted_agent_passes(test_model, reading, tmp_path):
    result, run_dir, junit, runs = _run_scenario(reading, tmp_path, test_model, PASSING_EXPECT)
    output = result.stdout + result.stderr

    assert result.returncode == 0, output
    data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert data["status"] == "passed", output
    assert [s["label"] for s in data["steps"]] == ["whirled", "hair"]
    assert all(s["status"] == "passed" for s in data["steps"]), output
    assert all(a["passed"] for a in data["assertions"]), output

    cases = junit.findall(".//testcase")
    names = [c.get("name") for c in cases]
    assert names == [
        "step 1 whirled",
        "step 2 hair",
        "assert call_connected",
        "assert agent_responded",
        "assert task_completed",
    ]
    assert junit.get("failures") == "0"
    assert junit.get("errors") == "0"
    assert not junit.findall(".//failure")

    transcript = Transcript.model_validate_json(
        (run_dir / "transcript.json").read_text(encoding="utf-8")
    )
    said = [s.text for s in transcript.segments if s.speaker == "caller"]
    assert said == reading[1]

    header, *rows = runs
    assert header.split()[:5] == ["id", "started", "status", "exit", "steps"]
    assert len(rows) == 1, runs
    assert rows[0].split()[2:5] == ["passed", "0", "2/2"]


@pytest.mark.model
def test_a_scenario_with_a_wrong_expectation_fails(test_model, reading, tmp_path):
    result, run_dir, junit, runs = _run_scenario(reading, tmp_path, test_model, FAILING_EXPECT)
    output = result.stdout + result.stderr

    assert result.returncode == 1, output
    assert "FAIL" in result.stdout
    assert f'contains: did not hear "{WRONG_WORD}"' in result.stdout
    data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    assert [s["status"] for s in data["steps"]] == ["passed", "failed"], output

    failures = junit.findall(".//failure")
    assert failures, ET.tostring(junit, encoding="unicode")
    assert WRONG_WORD in failures[0].get("message")
    assert int(junit.get("failures")) >= 1

    header, *rows = runs
    assert len(rows) == 1, runs
    assert rows[0].split()[2:5] == ["failed", "1", "1/2"]
