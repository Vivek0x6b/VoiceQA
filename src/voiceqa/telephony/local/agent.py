"""Local test agent: answers SIP calls and echoes the caller or plays a WAV file.

Used as the far end for loopback tests and for trying VoiceQA without a
real voice agent or a SIP provider.
"""

import asyncio
import os
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import numpy as np

from voiceqa.audio.dsp import resample, to_mono
from voiceqa.audio.wav import read_wav
from voiceqa.telephony.media import PTIME_MS, SAMPLE_RATE
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip.transport import Address
from voiceqa.telephony.sip.uas import ServerCall, ServerConfig, SipServer


@dataclass(frozen=True)
class AgentConfig:
    host: str = "127.0.0.1"
    port: int = 5060
    play: str | os.PathLike | None = None
    hangup_after: float | None = None
    timer_t1: float = 0.5


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


class LocalAgent:
    def __init__(self, config: AgentConfig, prompt: np.ndarray | None) -> None:
        self.config = config
        self._prompt = prompt
        self._server: SipServer | None = None

    @classmethod
    async def start(cls, config: AgentConfig) -> "LocalAgent":
        prompt = load_prompt(config.play) if config.play is not None else None
        self = cls(config, prompt)
        server_config = ServerConfig(
            host=config.host,
            port=config.port,
            timer_t1=config.timer_t1,
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

    async def _on_call(self, call: ServerCall) -> None:
        if self._prompt is not None:
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
                await call.hangup()
        finally:
            work.cancel()
