"""Local test agent: answers SIP calls and echoes the caller, plays a WAV
file, or follows a script of replies.

Used as the far end for loopback tests and for trying VoiceQA without a
real voice agent or a SIP provider.

With a script (see script.py) the agent plays the greeting on answer, then
one reply each time the VAD hears the caller's turn end. The VAD runs in a
ThreadedTurnDetector, off the event loop. A reply's first frame goes out in
the first send slot at or after the caller's turn end plus its delay_ms,
all on media_clock: the agent queues silence in front of the reply for the
slots in between, so timer jitter never moves it. A turn that ends while a
reply plays is kept, and the next reply follows at once.
"""

import asyncio
import logging
import math
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from voiceqa.audio.dsp import resample, to_mono
from voiceqa.audio.wav import read_wav
from voiceqa.speech.vad import TURN_END, ThreadedTurnDetector, TurnDetector, TurnEvent, create_vad
from voiceqa.telephony.local.script import THEN_HANGUP, Line, Script
from voiceqa.telephony.media import FRAME_SAMPLES, PTIME_MS, SAMPLE_RATE, media_clock
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip.transport import Address
from voiceqa.telephony.sip.uas import ServerCall, ServerConfig, SipServer

log = logging.getLogger(__name__)

FRAME_SECONDS = PTIME_MS / 1000
DETECTOR_CLOSE_TIMEOUT = 5.0


@dataclass(frozen=True)
class AgentConfig:
    host: str = "127.0.0.1"
    port: int = 5060
    play: str | os.PathLike | None = None
    hangup_after: float | None = None
    timer_t1: float = 0.5
    answer_delay: float = 0.0
    # A script read with read_script() and every say line synthesized.
    script: Script | None = None
    vad: str = "energy"
    end_silence_ms: float = 700


@dataclass(frozen=True)
class Played:
    """A greeting or reply the scripted agent started, for logs and tests.

    `at` is the media_clock slot of its first frame; `after` is the time
    its delay counts from: the answer for the greeting, the end of the
    caller's last speech frame for a reply."""

    call_id: str
    line: str
    kind: str
    at: float
    after: float


def load_prompt(path: str | os.PathLike) -> np.ndarray:
    """Read a 16-bit WAV as int16 mono at 8 kHz."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no such WAV file: {path}")
    samples, rate = read_wav(path)
    return resample(to_mono(samples), rate, SAMPLE_RATE)


ECHO_BUFFER_FRAMES = 50  # 1 s


class EchoBuffer:
    """Received frames waiting to be sent back. When full, the oldest are dropped."""

    def __init__(self, max_frames: int = ECHO_BUFFER_FRAMES) -> None:
        self._frames: deque[np.ndarray] = deque(maxlen=max_frames)

    def __len__(self) -> int:
        return len(self._frames)

    def push(self, frame: np.ndarray) -> None:
        self._frames.append(frame)

    def flush(self, send: Callable[[np.ndarray], None]) -> None:
        while self._frames:
            send(self._frames.popleft())


async def echo(call: ServerCall, buffer: EchoBuffer) -> None:
    while True:
        await asyncio.sleep(PTIME_MS / 1000)
        buffer.flush(call.media.send)


async def play(call: ServerCall, prompt: np.ndarray) -> None:
    call.media.send(prompt)
    await call.media.drain()


class _ScriptSink:
    """The media sink of a scripted call: feeds received frames to the VAD
    thread and remembers the slot of the last frame sent."""

    def __init__(self, detector: ThreadedTurnDetector) -> None:
        self._detector = detector
        self.last_sent: float | None = None
        self.sending = asyncio.Event()

    def sent(self, at: float, samples: np.ndarray) -> None:
        self.last_sent = at
        self.sending.set()

    def received(self, at: float, arrived: float, samples: np.ndarray) -> None:
        self._detector.feed(at, samples)


def _queue_line(call: ServerCall, sink: _ScriptSink, line: Line, target: float) -> float:
    """Queue line's audio so its first frame takes the first slot at or
    after target, and return that slot. Nothing else may be queued: then
    the next frame queued takes the slot after the last one sent. The
    sender runs on this thread, so no slot can pass while this runs.

    The slot returned (Played.at and the INFO line) assumes the sender skips
    no slots. After a stall of more than MAX_LATE_FRAMES the sender jumps
    ahead, the line goes out later than this, and the logged time reads early."""
    assert sink.last_sent is not None and line.audio is not None
    next_slot = sink.last_sent + FRAME_SECONDS
    pad = max(0, math.ceil((target - next_slot) / FRAME_SECONDS - 1e-6))
    call.media.send(np.concatenate([np.zeros(pad * FRAME_SAMPLES, dtype=np.int16), line.audio]))
    return next_slot + pad * FRAME_SECONDS


class LocalAgent:
    def __init__(self, config: AgentConfig, prompt: np.ndarray | None) -> None:
        self.config = config
        self._prompt = prompt
        self._server: SipServer | None = None
        self._vad_name = config.vad
        self.played: list[Played] = []

    @classmethod
    async def start(cls, config: AgentConfig) -> "LocalAgent":
        if config.script is not None:
            if config.play is not None:
                raise ValueError("give either a script or a WAV file to play, not both")
            for line in config.script.lines:
                if line.audio is None:
                    raise ValueError(f"{config.script.path}: {line.name}: say line not synthesized")
        prompt = load_prompt(config.play) if config.play is not None else None
        self = cls(config, prompt)
        if config.script is not None:
            # Built once here so a missing extra or model fails before
            # listening, and "auto" is resolved (and noticed) only once.
            self._vad_name = create_vad(config.vad).name
        server_config = ServerConfig(
            host=config.host,
            port=config.port,
            timer_t1=config.timer_t1,
            answer_delay=config.answer_delay,
        )
        self._server = await SipServer.open(server_config, self._open_media, self._on_call)
        return self

    @property
    def local_addr(self) -> Address:
        assert self._server is not None
        return self._server.local_addr

    @property
    def calls(self) -> list[ServerCall]:
        assert self._server is not None
        return self._server.calls

    async def close(self) -> None:
        if self._server is not None:
            await self._server.close()

    async def _open_media(self, advertise_host: str) -> RtpSession:
        return await RtpSession.open(bind_host=self.config.host, advertise_host=advertise_host)

    async def _run_script(self, call: ServerCall, script: Script) -> None:
        loop = asyncio.get_running_loop()
        ends: asyncio.Queue[float] = asyncio.Queue()

        def on_event(event: TurnEvent) -> None:
            if event.kind == TURN_END:
                loop.call_soon_threadsafe(ends.put_nowait, event.at)

        def on_error(exc: BaseException) -> None:
            log.error("call %s: the VAD failed, no more replies: %s", call.call_id, exc)

        detector = ThreadedTurnDetector(
            TurnDetector(create_vad(self._vad_name), end_silence_ms=self.config.end_silence_ms),
            on_event,
            on_error=on_error,
        )
        sink = _ScriptSink(detector)
        answered = media_clock()
        detector.start()
        try:
            call.media.record(sink)
            if script.greeting is not None:
                await self._play(call, sink, script.greeting, answered, answered)
            for reply in script.replies:
                turn_end = await ends.get()
                await self._play(call, sink, reply, turn_end, answered)
            if script.then == THEN_HANGUP:
                await call.hangup()
            else:
                await call.wait_ended()
        finally:
            try:
                await asyncio.to_thread(detector.close, DETECTOR_CLOSE_TIMEOUT)
            except TimeoutError as exc:
                log.warning("call %s: %s", call.call_id, exc)
            except Exception:
                pass  # a VAD error, already logged by on_error
            if detector.dropped_frames:
                log.warning(
                    "call %s: the VAD fell behind and dropped %d frames, turns may be missed",
                    call.call_id,
                    detector.dropped_frames,
                )

    async def _play(
        self, call: ServerCall, sink: _ScriptSink, line: Line, after: float, answered: float
    ) -> None:
        await sink.sending.wait()
        at = _queue_line(call, sink, line, after + line.delay_ms / 1000)
        self.played.append(Played(call.call_id, line.name, line.kind, at, after))
        log.info("call %s: %s (%s) at %.2f s", call.call_id, line.name, line.kind, at - answered)
        await call.media.drain()

    async def _on_call(self, call: ServerCall) -> None:
        if self.config.script is not None:
            work = asyncio.ensure_future(self._run_script(call, self.config.script))
        elif self._prompt is not None:
            work = asyncio.ensure_future(play(call, self._prompt))
        else:
            buffer = EchoBuffer()
            # call.media is the RtpSession that _open_media made.
            cast(RtpSession, call.media).listen(buffer.push)
            work = asyncio.ensure_future(echo(call, buffer))
        try:
            if self.config.hangup_after is None:
                await work
                await call.wait_ended()
            else:
                await asyncio.sleep(self.config.hangup_after)
                # We hang up ourselves, so nothing else stops the work: stop
                # it (and a script's VAD thread) before media closes.
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                await call.hangup()
        finally:
            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
