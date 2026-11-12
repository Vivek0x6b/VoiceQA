"""Helpers for scenario runs against the scripted test agent in this
process: tone WAVs, a stand-in STT that names each reply by its tone, and
the agent on a loop of its own for CLI tests."""

import asyncio
import contextlib
import threading
from collections.abc import Iterator, Mapping

import numpy as np
import yaml

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import write_wav
from voiceqa.engine.session import CallSession
from voiceqa.speech.models import Segment, Transcript
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.local.script import read_script
from voiceqa.telephony.media import SAMPLE_RATE

FRAME_S = 0.02
# One frame plus a little, for times on our own clock.
EXACT = FRAME_S + 0.005
# The scripted agent's end of turn, kept short so runs are quick.
AGENT_END_SILENCE_MS = 300
CALLER_HZ = 300


def tone(seconds, freq):
    return generate_tone(freq, seconds, SAMPLE_RATE, 0.3)


def silence(seconds):
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.int16)


def write_audio(path, samples):
    write_wav(path, samples, SAMPLE_RATE)
    return path


def dominant_hz(samples, rate=SAMPLE_RATE):
    x = samples.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / rate)[int(np.argmax(spectrum))])


class ToneStt:
    """Stands in for the STT worker: names an agent turn by its tone's
    frequency, so each scripted reply has a known transcript. A turn whose
    tone is not in `texts` comes back as "?"."""

    def __init__(self, texts: Mapping[int, str], error: str | None = None):
        self.texts = dict(texts)
        self.error = error
        self.heard: list[str] = []

    async def transcribe(self, samples, rate):
        if self.error is not None:
            raise RuntimeError(self.error)
        hz = dominant_hz(samples, rate)
        text = next((t for f, t in self.texts.items() if abs(f - hz) < 25), "?")
        self.heard.append(text)
        segment = Segment(start=0.0, end=len(samples) / rate, text=text)
        return Transcript(segments=[segment], engine="tone", model="test", language="en")


class FakeWorker(ToneStt):
    """ToneStt with the SttWorker lifecycle the CLI uses."""

    def __init__(self, texts, error=None, load_error=None):
        super().__init__(texts, error)
        self.load_error = load_error
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, *exc):
        return None

    async def wait_ready(self):
        if self.load_error is not None:
            from voiceqa.speech.worker import SttWorkerError

            raise SttWorkerError(self.load_error)


def write_replies(folder, greeting=None, replies=(), then="silence"):
    """replies.yaml for the scripted agent. `greeting` and each reply are
    (freq, seconds, delay_ms) tones, or a ready int16 array with its
    delay as (samples, delay_ms)."""
    folder.mkdir(parents=True, exist_ok=True)

    def line(name, spec):
        if isinstance(spec[0], np.ndarray):
            samples, delay = spec
        else:
            freq, seconds, delay = spec
            samples = tone(seconds, freq)
        write_audio(folder / f"{name}.wav", samples)
        return {"wav": f"{name}.wav", "delay_ms": delay}

    data = {"replies": [line(f"reply{i}", r) for i, r in enumerate(replies, 1)], "then": then}
    if greeting is not None:
        data["greeting"] = line("greeting", greeting)
    path = folder / "replies.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return read_script(path)


def agent_config(script=None, **kw):
    args = dict(host="127.0.0.1", port=0, timer_t1=0.02, end_silence_ms=AGENT_END_SILENCE_MS)
    if script is not None:
        args["script"] = script
    args.update(kw)
    return AgentConfig(**args)


def session_for(port, record_dir, stt, **kw):
    args = dict(vad="energy", end_silence_ms=AGENT_END_SILENCE_MS)
    args.update(kw)
    return CallSession("agent", "127.0.0.1", port, "1001", "", record_dir, stt=stt, **args)


@contextlib.contextmanager
def agent_in_thread(config: AgentConfig) -> Iterator[int]:
    """A LocalAgent on its own event loop in a thread, for code that runs
    asyncio.run itself (the CLI). Yields the agent's SIP port."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, name="test-agent", daemon=False)
    thread.start()
    agent = None
    try:
        agent = asyncio.run_coroutine_threadsafe(LocalAgent.start(config), loop).result(10)
        yield agent.local_addr[1]
    finally:
        if agent is not None:
            asyncio.run_coroutine_threadsafe(agent.close(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(10)
        loop.close()
