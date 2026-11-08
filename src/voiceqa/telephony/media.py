"""Contract between signalling (SIP) and media (RTP).

Call flow:

1. The RTP side binds a local UDP socket first and reports it as LocalMedia.
2. SIP puts LocalMedia in the SDP offer (c= host, m= port, payload types in
   preference order, a=ptime).
3. SIP parses the SDP answer into RemoteMedia: the remote RTP host, port and
   the one payload type the remote picked.
4. RTP starts sending to and receiving from RemoteMedia.

SIP never touches RTP sockets and RTP never parses SDP. Audio crossing this
boundary is int16 mono at 8 kHz, as produced by voiceqa.audio.
"""

import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np

SAMPLE_RATE = 8000
PTIME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * PTIME_MS // 1000

# The one clock for media timing and call events. Not loop.time(): on
# Windows before Python 3.13 that ticks in 15.6 ms steps.
media_clock = time.perf_counter


class PayloadType(IntEnum):
    PCMU = 0
    PCMA = 8


def _check_port(port: int) -> None:
    if not 1 <= port <= 65535:
        raise ValueError(f"RTP port must be 1-65535, got {port}")


def _check_ptime(ptime_ms: int) -> None:
    if ptime_ms != PTIME_MS:
        raise ValueError(f"only {PTIME_MS} ms ptime is supported, got {ptime_ms}")


@dataclass(frozen=True)
class LocalMedia:
    """Our side of the media stream, as offered in SDP."""

    host: str
    port: int
    payload_types: tuple[PayloadType, ...] = (PayloadType.PCMU, PayloadType.PCMA)
    ptime_ms: int = PTIME_MS

    def __post_init__(self) -> None:
        _check_port(self.port)
        _check_ptime(self.ptime_ms)
        if not self.payload_types:
            raise ValueError("at least one payload type must be offered")


@dataclass(frozen=True)
class RemoteMedia:
    """The far end of the media stream, taken from the SDP answer."""

    host: str
    port: int
    payload_type: PayloadType
    ptime_ms: int = PTIME_MS

    def __post_init__(self) -> None:
        _check_port(self.port)
        _check_ptime(self.ptime_ms)


class MediaSink(Protocol):
    """Gets every frame a MediaSession sends or receives, decoded to int16.

    Times are seconds on media_clock. `at` is where the frame belongs on that
    clock, taken from its RTP timestamp; `arrived` is when a received packet
    came in.
    """

    def sent(self, at: float, samples: np.ndarray) -> None: ...

    def received(self, at: float, arrived: float, samples: np.ndarray) -> None: ...


class MediaSession(Protocol):
    """What the RTP side provides. Kept small so it can be swapped for Rust later."""

    @property
    def local(self) -> LocalMedia:
        """The bound local socket, available before start()."""
        ...

    async def start(self, remote: RemoteMedia) -> None:
        """Begin sending and receiving. Sends encoded silence while nothing is queued."""
        ...

    def send(self, samples: np.ndarray) -> None:
        """Queue int16 8 kHz mono audio. It goes out in 20 ms packets, paced in real time."""
        ...

    async def drain(self) -> None:
        """Wait until all queued audio has been sent."""
        ...

    def record(self, sink: MediaSink) -> None:
        """Pass every frame sent and received from now on to sink. The session keeps no audio."""
        ...

    async def close(self) -> None:
        """Stop sending and close the socket. Safe to call twice."""
        ...
