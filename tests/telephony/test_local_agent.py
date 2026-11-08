import asyncio

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import write_wav
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent, load_prompt
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip import SipConfig, place_call


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


def loud_frames(audio, threshold=2000):
    n = len(audio) // FRAME_SAMPLES
    frames = audio[: n * FRAME_SAMPLES].reshape(n, FRAME_SAMPLES).astype(np.float64)
    return int((np.sqrt((frames**2).mean(axis=1)) > threshold).sum())


def dominant_hz(audio):
    loud = audio.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(loud))
    return np.fft.rfftfreq(len(loud), 1 / SAMPLE_RATE)[spectrum.argmax()]


async def call_and_record(agent, send=None, listen=1.5):
    """Call the agent, optionally send audio, and return what came back."""
    rtp = await RtpSession.open(bind_host="127.0.0.1")
    call = await place_call(caller_config(agent), "agent", rtp.local)
    try:
        await rtp.start(call.remote_media)
        if send is not None:
            rtp.send(send)
        await asyncio.sleep(listen)
        return rtp.recording()
    finally:
        await call.hangup()
        await rtp.close()


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
