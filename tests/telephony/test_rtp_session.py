import asyncio
import socket
import time
import tracemalloc

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.g711 import decode_pcma, decode_pcmu, encode_pcma, encode_pcmu
from voiceqa.telephony.media import LocalMedia, PayloadType, RemoteMedia, media_clock
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.rtp.packet import RtpPacket, build_packet

CODECS = {
    PayloadType.PCMU: (encode_pcmu, decode_pcmu),
    PayloadType.PCMA: (encode_pcma, decode_pcma),
}


def _remote(session: RtpSession, pt: PayloadType = PayloadType.PCMU) -> RemoteMedia:
    return RemoteMedia(host="127.0.0.1", port=session.local.port, payload_type=pt)


async def _pair() -> tuple[RtpSession, RtpSession]:
    a = await RtpSession.open(bind_host="127.0.0.1")
    b = await RtpSession.open(bind_host="127.0.0.1")
    return a, b


async def _close(*sessions: RtpSession) -> None:
    for s in sessions:
        await s.close()


class FrameLog:
    """A MediaSink that keeps everything, for tests."""

    def __init__(self) -> None:
        self.out: list[tuple[float, np.ndarray]] = []
        self.inc: list[tuple[float, float, np.ndarray]] = []

    def sent(self, at, samples):
        self.out.append((at, samples))

    def received(self, at, arrived, samples):
        self.inc.append((at, arrived, samples))

    def heard(self) -> np.ndarray:
        """Received audio in timestamp order, without filling gaps."""
        frames = sorted(self.inc, key=lambda f: f[0])
        return np.concatenate([f[2] for f in frames]) if frames else np.zeros(0, np.int16)

    def span(self) -> float:
        times = [f[0] for f in self.inc]
        return max(times) - min(times) + 0.02


def test_open_reports_bound_port():
    async def main():
        s = await RtpSession.open(bind_host="127.0.0.1")
        t = await RtpSession.open(
            bind_host="127.0.0.1",
            advertise_host="203.0.113.7",
            payload_types=(PayloadType.PCMA,),
        )
        try:
            return s.local, t.local
        finally:
            await _close(s, t)

    s_local, t_local = asyncio.run(main())
    assert isinstance(s_local, LocalMedia)
    assert s_local.host == "127.0.0.1"
    assert s_local.port > 0
    assert s_local.payload_types == (PayloadType.PCMU, PayloadType.PCMA)
    assert t_local.host == "203.0.113.7"
    assert t_local.payload_types == (PayloadType.PCMA,)
    assert t_local.port != s_local.port


@pytest.mark.parametrize("pt", [PayloadType.PCMU, PayloadType.PCMA])
def test_exchange(pt):
    encode, decode = CODECS[pt]
    x = generate_tone(440, 0.30625)
    assert len(x) % 160 != 0

    async def main():
        a, b = await _pair()
        log_a, log_b = FrameLog(), FrameLog()
        a.record(log_a)
        b.record(log_b)
        try:
            await b.start(_remote(a, pt))
            a.send(x)
            await a.start(_remote(b, pt))
            await a.drain()
            await asyncio.sleep(0.1)
            return log_a, log_b
        finally:
            await _close(a, b)

    log_a, log_b = asyncio.run(main())
    pad = 160 - len(x) % 160
    silence = decode(encode(np.zeros(pad, dtype=np.int16)))
    for got in (log_b.heard(), np.concatenate([f[1] for f in log_a.out])):
        # The sender records what went on the wire, so both sides match.
        assert got.dtype == np.int16
        np.testing.assert_array_equal(got[: len(x)], decode(encode(x)))
        np.testing.assert_array_equal(got[len(x) : len(x) + pad], silence)


def test_silence_when_idle():
    async def main():
        a, b = await _pair()
        log = FrameLog()
        b.record(log)
        try:
            await b.start(_remote(a, PayloadType.PCMA))
            await a.start(_remote(b, PayloadType.PCMA))
            await asyncio.sleep(0.3)
            return b.packets_received, log.heard()
        finally:
            await _close(a, b)

    received, got = asyncio.run(main())
    assert received >= 10
    assert len(got) == 160 * received
    np.testing.assert_array_equal(got, decode_pcma(encode_pcma(np.zeros(len(got), np.int16))))


def test_pacing_no_drift():
    async def main():
        a, b = await _pair()
        try:
            await b.start(_remote(a))
            t0 = media_clock()
            await a.start(_remote(b))
            ts0 = a._stream.timestamp

            def slots() -> int:
                # The RTP timestamp also counts slots skipped after a stall,
                # which a slow CI runner can cause; drift is about slots.
                return ((a._stream.timestamp - ts0) & 0xFFFFFFFF) // 160

            await asyncio.sleep(1.0)
            first = (slots(), media_clock() - t0)
            await asyncio.sleep(2.0)
            second = (slots(), media_clock() - t0)
            await asyncio.sleep(0.05)
            return first, second, b.packets_received, a.packets_sent
        finally:
            await _close(a, b)

    (n1, e1), (n2, e2), received, sent = asyncio.run(main())
    assert abs(n1 - 50 * e1) <= 2
    assert abs(n2 - 50 * e2) <= 2
    assert received >= sent - 3


def test_late_sender_skips_instead_of_bursting():
    async def main():
        a, b = await _pair()
        log = FrameLog()
        b.record(log)
        try:
            await b.start(_remote(a))
            await a.start(_remote(b))
            t0 = time.perf_counter()
            await asyncio.sleep(0.2)
            s0 = time.perf_counter()
            time.sleep(0.5)  # stall the whole event loop
            stall = time.perf_counter() - s0
            await asyncio.sleep(0.3)
            sent = a.packets_sent
            elapsed = time.perf_counter() - t0
            await asyncio.sleep(0.05)
            recorded = time.perf_counter() - t0
            return sent, elapsed, stall, recorded, log.span()
        finally:
            await _close(a, b)

    # Slow CI runners overshoot sleeps, so expectations use measured time.
    sent, elapsed, stall, recorded, span = asyncio.run(main())
    # A sender that bursts its backlog sends one packet per slot of wall time.
    # Ours skips the stalled slots instead.
    assert 10 <= sent <= 50 * (elapsed - stall) + 5
    # The skipped slots still advance the RTP timestamp, so the far end
    # places the frames after the gap at the right time.
    assert abs(span - recorded) <= 0.1


def test_reorder_and_malformed_over_udp():
    def frame(value: int) -> bytes:
        return encode_pcmu(np.full(160, value, dtype=np.int16))

    def pkt(ts: int, value: int, pt: int = 0) -> bytes:
        return build_packet(
            RtpPacket(
                payload_type=pt,
                sequence=0,
                timestamp=ts & 0xFFFFFFFF,
                ssrc=5,
                payload=frame(value),
            )
        )

    async def main():
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw.bind(("127.0.0.1", 0))
        b = await RtpSession.open(bind_host="127.0.0.1")
        log = FrameLog()
        b.record(log)
        try:
            await b.start(RemoteMedia(host="127.0.0.1", port=raw.getsockname()[1], payload_type=0))
            dest = ("127.0.0.1", b.local.port)
            base = 2**32 - 160
            for data in [
                b"garbage",
                b"\x80\x00\x00",
                b"\x40" + pkt(base, 1)[1:],
                pkt(base, 7000, pt=8),
                pkt(base + 320, 3000),
                pkt(base, 1000),
                pkt(base + 160, 2000),
                pkt(base + 160, -9000),
            ]:
                raw.sendto(data, dest)
            loop = asyncio.get_running_loop()
            stop = loop.time() + 2.0
            while b.packets_received < 3 and loop.time() < stop:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            return b.packets_received, log
        finally:
            await b.close()
            raw.close()

    received, log = asyncio.run(main())
    assert received == 3
    expected = np.concatenate([decode_pcmu(frame(v)) for v in (1000, 2000, 3000)])
    np.testing.assert_array_equal(log.heard(), expected)
    times = sorted(f[0] for f in log.inc)
    assert np.diff(times) == pytest.approx([0.02, 0.02])
    # The first packet to arrive (base + 320) anchors the timeline.
    assert log.inc[0][0] == pytest.approx(log.inc[0][1])


def test_listener_gets_each_new_frame_in_arrival_order():
    def pkt(ts: int, value: int, pt: int = 0) -> bytes:
        payload = encode_pcmu(np.full(160, value, dtype=np.int16))
        return build_packet(
            RtpPacket(payload_type=pt, sequence=0, timestamp=ts, ssrc=5, payload=payload)
        )

    async def main():
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        raw.bind(("127.0.0.1", 0))
        b = await RtpSession.open(bind_host="127.0.0.1")
        heard = []
        b.listen(heard.append)
        try:
            await b.start(RemoteMedia(host="127.0.0.1", port=raw.getsockname()[1], payload_type=0))
            dest = ("127.0.0.1", b.local.port)
            for data in [
                b"garbage",
                pkt(320, 3000),
                pkt(0, 7000, pt=8),
                pkt(0, 1000),
                pkt(320, -9000),
            ]:
                raw.sendto(data, dest)
            loop = asyncio.get_running_loop()
            stop = loop.time() + 2.0
            while b.packets_received < 2 and loop.time() < stop:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            return heard
        finally:
            await b.close()
            raw.close()

    heard = asyncio.run(main())
    assert [int(f[0]) for f in heard] == [
        int(decode_pcmu(encode_pcmu(np.array([v], np.int16)))[0]) for v in (3000, 1000)
    ]
    assert all(f.dtype == np.int16 and f.shape == (160,) for f in heard)


def test_sink_gets_sent_and_received_frames_on_the_media_clock():
    x = generate_tone(440, 0.2)

    async def main():
        a, b = await _pair()
        log_a, log_b = FrameLog(), FrameLog()
        a.record(log_a)
        b.record(log_b)
        try:
            await b.start(_remote(a))
            before = media_clock()
            a.send(x)
            await a.start(_remote(b))
            await a.drain()
            await asyncio.sleep(0.1)
        finally:
            await _close(a, b)
        return before, media_clock(), log_a, log_b

    before, after, log_a, log_b = asyncio.run(main())
    sent_at = [f[0] for f in log_a.out]
    # Sent frames are timed by their slot deadline, exactly 20 ms apart.
    assert np.diff(sent_at) == pytest.approx([0.02] * (len(sent_at) - 1), abs=1e-9)
    assert before <= sent_at[0] <= sent_at[-1] <= after
    for at, arrived, _ in log_b.inc:
        assert before <= arrived <= after
        assert abs(at - arrived) < 0.05
    # A frame cannot arrive before it was sent.
    assert log_b.inc[0][1] >= sent_at[0]


def test_session_keeps_no_audio_of_its_own():
    payload = encode_pcmu(np.full(160, 500, dtype=np.int16))

    def pkt(i: int) -> bytes:
        header = dict(payload_type=0, sequence=i & 0xFFFF, timestamp=160 * i, ssrc=9)
        return build_packet(RtpPacket(**header, payload=payload))

    async def main():
        b = await RtpSession.open(bind_host="127.0.0.1")
        try:
            await b.start(RemoteMedia(host="127.0.0.1", port=9, payload_type=0))
            n = 10 * 60 * 50
            tracemalloc.start()
            try:
                for i in range(n):
                    # Straight into the receive path, ten minutes of packets.
                    b._on_datagram(pkt(i))
                    if i == n // 10:
                        early, _ = tracemalloc.get_traced_memory()
                late, _ = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            return b.packets_received, late - early
        finally:
            await b.close()

    received, growth = asyncio.run(main())
    assert received == 30000
    assert growth < 64 * 1024


def test_send_validation():
    async def main():
        s = await RtpSession.open(bind_host="127.0.0.1")
        try:
            with pytest.raises(TypeError):
                s.send(np.zeros(160, dtype=np.float32))
            with pytest.raises(ValueError):
                s.send(np.zeros((160, 2), dtype=np.int16))
            s.send(np.zeros(0, dtype=np.int16))
        finally:
            await s.close()

    asyncio.run(main())


def test_drain_rules():
    async def main():
        s = await RtpSession.open(bind_host="127.0.0.1")
        await s.drain()
        s.send(np.zeros(100, dtype=np.int16))
        with pytest.raises(RuntimeError):
            await s.drain()
        await s.close()
        await s.drain()

    asyncio.run(main())


def test_start_twice_rejected():
    async def main():
        a, b = await _pair()
        try:
            await a.start(_remote(b))
            with pytest.raises(RuntimeError):
                await a.start(_remote(b))
        finally:
            await _close(a, b)

    asyncio.run(main())


def test_close_twice():
    async def main():
        a, b = await _pair()
        await a.start(_remote(b))
        await a.close()
        await a.close()
        await b.close()
        with pytest.raises(RuntimeError):
            a.send(np.zeros(160, dtype=np.int16))
        with pytest.raises(RuntimeError):
            await a.start(_remote(b))

    asyncio.run(main())
