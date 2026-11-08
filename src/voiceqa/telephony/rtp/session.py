"""Asyncio RTP session: one UDP socket, a paced sender task and a recorder.

Pacing: packet n is due at t0 + n * 20 ms, so timer jitter (about 15 ms on
Windows) never accumulates into drift. If the loop wakes up to 100 ms late,
the overdue packets go out back to back. If it is later than that (the loop
was blocked), the missed slots are skipped instead of burst: the RTP
timestamp jumps forward by the missed time so the far end hears a gap of the
right length, and queued audio continues where it left off.
"""

import asyncio
import contextlib
from collections import deque
from collections.abc import Callable

import numpy as np

from voiceqa.audio.g711 import decode_pcma, decode_pcmu, encode_pcma, encode_pcmu
from voiceqa.telephony.media import FRAME_SAMPLES, PTIME_MS, LocalMedia, PayloadType, RemoteMedia
from voiceqa.telephony.rtp.packet import RtpStream, parse_packet
from voiceqa.telephony.rtp.recorder import Recorder

FRAME_SECONDS = PTIME_MS / 1000
MAX_LATE_FRAMES = 5

_CODECS = {
    PayloadType.PCMU: (encode_pcmu, decode_pcmu),
    PayloadType.PCMA: (encode_pcma, decode_pcma),
}


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, on_datagram: Callable[[bytes], None]) -> None:
        self._on_datagram = on_datagram

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self._on_datagram(data)

    def error_received(self, exc: Exception) -> None:
        # ICMP port unreachable (ConnectionResetError on Windows) when the far
        # end is not listening yet or has gone. Not fatal for RTP.
        pass


class RtpSession:
    def __init__(self) -> None:
        self._transport: asyncio.DatagramTransport | None = None
        self._local: LocalMedia | None = None
        self._remote: RemoteMedia | None = None
        self._stream: RtpStream | None = None
        self._recorder: Recorder | None = None
        self._sender: asyncio.Task | None = None
        self._queue: deque[np.ndarray] = deque()
        self._drained = asyncio.Event()
        self._closed = False
        self._packets_sent = 0
        self._packets_received = 0
        self._listener: Callable[[np.ndarray], None] | None = None

    @classmethod
    async def open(
        cls,
        bind_host: str = "0.0.0.0",
        port: int = 0,
        advertise_host: str | None = None,
        payload_types: tuple[PayloadType, ...] = (PayloadType.PCMU, PayloadType.PCMA),
    ) -> "RtpSession":
        self = cls()
        loop = asyncio.get_running_loop()
        transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self._on_datagram), local_addr=(bind_host, port)
        )
        self._transport = transport
        try:
            self._local = LocalMedia(
                host=advertise_host or bind_host,
                port=transport.get_extra_info("sockname")[1],
                payload_types=tuple(payload_types),
            )
        except Exception:
            transport.close()
            raise
        return self

    @property
    def local(self) -> LocalMedia:
        return self._local

    @property
    def packets_sent(self) -> int:
        return self._packets_sent

    @property
    def packets_received(self) -> int:
        return self._packets_received

    def listen(self, callback: Callable[[np.ndarray], None]) -> None:
        """Call back with each newly received frame, decoded, in arrival order."""
        self._listener = callback

    async def start(self, remote: RemoteMedia) -> None:
        if self._closed:
            raise RuntimeError("RTP session is closed")
        if self._remote is not None:
            raise RuntimeError("RTP session already started")
        encode, decode = _CODECS[PayloadType(remote.payload_type)]
        self._remote = remote
        self._encode = encode
        self._decode = decode
        self._silence = encode(np.zeros(FRAME_SAMPLES, dtype=np.int16))
        self._stream = RtpStream(int(remote.payload_type))
        self._recorder = Recorder(decode)
        self._sender = asyncio.create_task(self._send_loop())

    def send(self, samples: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("RTP session is closed")
        x = np.asarray(samples)
        if x.ndim != 1:
            raise ValueError(f"expected mono 1-D samples, got shape {x.shape}")
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError(f"expected integer samples, got {x.dtype}")
        if len(x) == 0:
            return
        x = np.clip(x.astype(np.int64), -32768, 32767).astype(np.int16)
        x = np.concatenate([x, np.zeros(-len(x) % FRAME_SAMPLES, dtype=np.int16)])
        self._queue.extend(x.reshape(-1, FRAME_SAMPLES))
        self._drained.clear()

    async def drain(self) -> None:
        if self._closed or not self._queue:
            return
        if self._sender is None:
            raise RuntimeError("drain() before start() would wait forever")
        await self._drained.wait()

    def recording(self) -> np.ndarray:
        if self._recorder is None:
            return np.zeros(0, dtype=np.int16)
        return self._recorder.audio()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._sender is not None:
            self._sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sender
        self._transport.close()
        self._drained.set()

    async def _send_loop(self) -> None:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        slot = 0
        while True:
            delay = t0 + slot * FRAME_SECONDS - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            behind = int((loop.time() - t0) / FRAME_SECONDS) - slot
            if behind > MAX_LATE_FRAMES:
                self._stream.skip(behind * FRAME_SAMPLES)
                slot += behind
            self._send_frame()
            slot += 1

    def _send_frame(self) -> None:
        payload = self._encode(self._queue.popleft()) if self._queue else self._silence
        self._transport.sendto(self._stream.next_packet(payload), self._addr)
        self._packets_sent += 1
        if not self._queue:
            self._drained.set()

    @property
    def _addr(self) -> tuple[str, int]:
        return (self._remote.host, self._remote.port)

    def _on_datagram(self, data: bytes) -> None:
        if self._recorder is None or self._closed:
            return
        try:
            packet = parse_packet(data)
        except ValueError:
            return
        if packet.payload_type != self._remote.payload_type:
            return
        if self._recorder.add(packet):
            self._packets_received += 1
            if self._listener is not None:
                self._listener(self._decode(packet.payload))
