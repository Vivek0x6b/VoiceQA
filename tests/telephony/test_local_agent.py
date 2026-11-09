import asyncio
import json
import os
import tempfile
import time
import tracemalloc

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.telephony.local.agent import AgentConfig, EchoBuffer, LocalAgent, load_prompt
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE
from voiceqa.telephony.recording import CallRecorder
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip import SipConfig, place_call
from voiceqa.telephony.sip.dialog import via_value
from voiceqa.telephony.sip.message import Response, new_branch, parse_message
from voiceqa.telephony.sip.sdp import build_offer


def caller_config(agent):
    return SipConfig(
        server_host="127.0.0.1",
        server_port=agent.local_addr[1],
        username="1001",
        local_host="127.0.0.1",
        timer_t1=0.02,
        timer_t2=0.16,
    )


def agent_config(**kw):
    args = dict(host="127.0.0.1", port=0, timer_t1=0.02)
    args.update(kw)
    return AgentConfig(**args)


def frame_rms(audio):
    n = len(audio) // FRAME_SAMPLES
    frames = audio[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES).astype(np.float64)
    return np.sqrt((frames**2).mean(axis=1))


def loud_frames(audio, threshold=2000):
    return int((frame_rms(audio) > threshold).sum())


def dominant_hz(audio):
    loud = audio.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(loud))
    return np.fft.rfftfreq(len(loud), 1 / SAMPLE_RATE)[spectrum.argmax()]


async def record_call(agent, directory, send=None, listen=1.5, media_delay=0.0, during=None):
    """Call the agent, optionally send audio, and record both sides to directory.

    during, if given, is awaited with the call while listening.
    """
    rec = CallRecorder(directory)
    try:
        rtp = await RtpSession.open(bind_host="127.0.0.1")
        rtp.record(rec)
        try:
            call = await place_call(caller_config(agent), "agent", rtp.local, on_event=rec.event)
            try:
                await asyncio.sleep(media_delay)
                await rtp.start(call.remote_media)
                if send is not None:
                    rtp.send(send)
                if during is None:
                    await asyncio.sleep(listen)
                else:
                    await asyncio.gather(asyncio.sleep(listen), during(call, rtp))
            finally:
                await call.hangup()
        finally:
            await rtp.close()
    finally:
        rec.close()


async def call_and_record(agent, send=None, listen=1.5):
    """Call the agent, optionally send audio, and return what came back."""
    with tempfile.TemporaryDirectory() as tmp:
        await record_call(agent, tmp, send=send, listen=listen)
        return read_wav(os.path.join(tmp, "agent.wav"))[0]


def test_echo_sends_the_callers_audio_back():
    tone = generate_tone(440, 1.0)

    async def main():
        agent = await LocalAgent.start(agent_config())
        try:
            return await call_and_record(agent, send=tone, listen=1.6)
        finally:
            await agent.close()

    got = asyncio.run(main())
    # One second of tone is 50 frames; allow a frame or two at each edge.
    assert 46 <= loud_frames(got) <= 52
    assert abs(dominant_hz(got) - 440) < 5


def first_loud_sample(audio, threshold=2000):
    return int(np.flatnonzero(np.abs(audio) > threshold)[0])


def test_recording_of_an_echo_call_lines_up_on_one_clock(tmp_path):
    tone = generate_tone(440, 0.5)

    async def main():
        agent = await LocalAgent.start(agent_config())
        try:
            # Media starts well after time zero, so a recorder that did not
            # pad the start of a track would put the tone in the wrong place.
            await record_call(agent, tmp_path, send=tone, listen=1.2, media_delay=0.3)
        finally:
            await agent.close()

    asyncio.run(main())
    caller, _ = read_wav(tmp_path / "caller.wav")
    agent, _ = read_wav(tmp_path / "agent.wav")
    mix, _ = read_wav(tmp_path / "mix.wav")
    data = json.loads((tmp_path / "events.json").read_text())

    assert len(caller) == len(agent) == len(mix)
    np.testing.assert_array_equal(mix[:, 0], caller)
    np.testing.assert_array_equal(mix[:, 1], agent)
    assert abs(len(caller) / SAMPLE_RATE - data["duration"]) < 1e-3

    t = {e["event"]: e["t"] for e in data["events"]}
    assert len(t) == len(data["events"]) == 6
    assert 0 <= t["invite_sent"] <= t["ringing"] <= t["answered"]
    assert t["answered"] <= min(t["first_rtp_out"], t["first_rtp_in"])
    assert max(t["first_rtp_out"], t["first_rtp_in"]) < t["bye_sent"] <= data["duration"]

    # The tone starts where the first sent frame is placed, and the echo
    # comes back a little later on the same clock.
    said = first_loud_sample(caller) / SAMPLE_RATE
    heard = first_loud_sample(agent) / SAMPLE_RATE
    assert t["first_rtp_out"] >= t["answered"] + 0.25
    assert abs(said - t["first_rtp_out"]) < 0.025
    assert 0 < heard - said < 0.2
    assert 23 <= loud_frames(caller) <= 26
    # The agent may slip a silence frame into the echo when its queue runs
    # dry for a slot, which spreads the tone over more frames. Every frame
    # still comes back, so compare energy instead of counting frames.
    energy = np.sum(agent.astype(np.float64) ** 2) / np.sum(caller.astype(np.float64) ** 2)
    assert 0.9 < energy < 1.1


class RawSip(asyncio.DatagramProtocol):
    def __init__(self):
        self.received = []

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.received.append(parse_message(data))

    def finals(self):
        return [m for m in self.received if isinstance(m, Response) and m.status >= 200]


def test_reinvite_mid_call_does_not_interrupt_the_echo(tmp_path):
    tone = generate_tone(440, 2.0)
    seen = {}

    async def main():
        agent = await LocalAgent.start(agent_config())

        async def refresh(call, rtp):
            await asyncio.sleep(0.5)
            loop = asyncio.get_running_loop()
            transport, raw = await loop.create_datagram_endpoint(
                RawSip, local_addr=("127.0.0.1", 0)
            )
            try:
                port = transport.get_extra_info("sockname")[1]
                dialog = call._dialog
                reinvite = dialog.make_request("INVITE", via_value("127.0.0.1", port, new_branch()))
                reinvite.headers.add("Contact", f"<sip:1001@127.0.0.1:{port}>")
                reinvite.headers.add("Content-Type", "application/sdp")
                reinvite.body = build_offer(rtp.local).encode()
                transport.sendto(reinvite.to_bytes(), agent.local_addr)
                async with asyncio.timeout(3):
                    while not raw.finals():
                        await asyncio.sleep(0.005)
                seen["status"] = raw.finals()[0].status
                ack = dialog.make_request(
                    "ACK", via_value("127.0.0.1", port, new_branch()), cseq=reinvite.cseq[0]
                )
                transport.sendto(ack.to_bytes(), agent.local_addr)
            finally:
                transport.close()
            seen["up"] = not call.ended and len(agent.calls) == 1

        try:
            await record_call(agent, tmp_path, send=tone, listen=2.6, during=refresh)
        finally:
            await agent.close()

    asyncio.run(main())
    assert seen == {"status": 200, "up": True}
    caller, _ = read_wav(tmp_path / "caller.wav")
    echoed, _ = read_wav(tmp_path / "agent.wav")
    loud = np.flatnonzero(frame_rms(echoed) > 2000)
    assert loud[-1] - loud[0] >= 95
    # The echo may slip a silence frame in when its queue runs dry for a slot
    # (see test_recording_of_an_echo_call_lines_up_on_one_clock).
    gaps = (loud[-1] - loud[0] + 1) - len(loud)
    assert gaps <= 2
    energy = np.sum(echoed.astype(np.float64) ** 2) / np.sum(caller.astype(np.float64) ** 2)
    assert 0.9 < energy < 1.1


def test_play_sends_the_wav_resampled_to_8k(tmp_path):
    wav = tmp_path / "prompt.wav"
    tone = generate_tone(600, 0.5, sample_rate=16000)
    write_wav(wav, np.stack([tone, tone], axis=1), 16000)

    async def main():
        agent = await LocalAgent.start(agent_config(play=wav))
        try:
            return await call_and_record(agent, listen=1.0)
        finally:
            await agent.close()

    got = asyncio.run(main())
    assert 23 <= loud_frames(got) <= 26
    assert abs(dominant_hz(got) - 600) < 5


def test_hangup_after_ends_the_call_from_the_agent():
    async def main():
        agent = await LocalAgent.start(agent_config(hangup_after=0.3))
        try:
            rtp = await RtpSession.open(bind_host="127.0.0.1")
            call = await place_call(caller_config(agent), "agent", rtp.local)
            await rtp.start(call.remote_media)
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            async with asyncio.timeout(3):
                await call.wait_ended()
            elapsed = loop.time() - t0
            await call.hangup()
            await rtp.close()
            return elapsed, agent.calls
        finally:
            await agent.close()

    elapsed, calls = asyncio.run(main())
    assert 0.25 <= elapsed < 2
    assert calls == []


def test_load_prompt_downmixes_and_resamples(tmp_path):
    wav = tmp_path / "p.wav"
    tone = generate_tone(300, 1.0, sample_rate=16000)
    write_wav(wav, np.stack([tone, tone], axis=1), 16000)
    out = load_prompt(wav)
    assert out.dtype == np.int16
    assert out.shape == (8000,)
    assert abs(dominant_hz(out) - 300) < 2


def test_missing_prompt_fails_before_listening(tmp_path):
    with pytest.raises(FileNotFoundError):
        asyncio.run(LocalAgent.start(agent_config(play=tmp_path / "nope.wav")))


TEN_MINUTES = 10 * 60 * 1000 // 20


def frames(n):
    rng = np.random.default_rng(1)
    pool = [rng.integers(-8000, 8000, FRAME_SAMPLES, dtype=np.int16) for _ in range(64)]
    for i in range(n):
        yield pool[i % len(pool)]


def test_echo_buffer_work_per_frame_stays_flat_over_ten_minutes():
    buf = EchoBuffer()
    sent = []
    block = TEN_MINUTES // 10
    timings = []
    t = time.perf_counter()
    for i, frame in enumerate(frames(TEN_MINUTES), 1):
        buf.push(frame)
        buf.flush(sent.append)
        sent.clear()
        if i % block == 0:
            now = time.perf_counter()
            timings.append(now - t)
            t = now
    first, last = min(timings[:3]), min(timings[-3:])
    assert last < 3 * first + 0.01, timings


def test_echo_buffer_memory_stays_bounded_over_ten_minutes():
    buf = EchoBuffer()
    count = 0

    def sink(frame):
        nonlocal count
        count += 1

    tracemalloc.start()
    try:
        for i, frame in enumerate(frames(TEN_MINUTES), 1):
            buf.push(frame.copy())
            buf.flush(sink)
            if i == TEN_MINUTES // 10:
                early, _ = tracemalloc.get_traced_memory()
        late, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert count == TEN_MINUTES
    assert late - early < 64 * 1024


def test_echo_buffer_drops_the_oldest_frames_when_not_flushed():
    buf = EchoBuffer(max_frames=50)
    for frame in frames(TEN_MINUTES):
        buf.push(frame)
    assert len(buf) == 50
    out = []
    buf.flush(out.append)
    assert len(out) == 50 and len(buf) == 0
    last_50 = list(frames(TEN_MINUTES))[-50:]
    assert all(np.array_equal(a, b) for a, b in zip(out, last_50, strict=True))
