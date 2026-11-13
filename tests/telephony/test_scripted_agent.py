"""The scripted test agent (`voiceqa agent --script`): a greeting, then one
reply per caller turn, timed from the end of the caller's last speech frame.

The loopback tests place real calls from this process. The caller's audio
starts with digital silence (PCMU zeros), as a real caller's does, so these
tests also cover the energy VAD finding speech after digital silence."""

import asyncio
import logging
import time

import numpy as np
import pytest
import yaml
from test_local_agent import agent_config, caller_config, frame_rms
from tests_support import nothing_left_running

from voiceqa.audio.dsp import generate_tone, resample
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.speech import model_cache
from voiceqa.speech.interfaces import Speech
from voiceqa.speech.model_cache import SttExtraMissingError
from voiceqa.speech.tts_process import TtsError
from voiceqa.speech.vad import ThreadedTurnDetector
from voiceqa.telephony.local.agent import LocalAgent
from voiceqa.telephony.local.script import ScriptError, read_script
from voiceqa.telephony.media import SAMPLE_RATE
from voiceqa.telephony.recording import CallRecorder
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip import place_call

FRAME_S = 0.02
# Far above the one-frame tolerance, so a reply sent at the turn end (no
# delay) or at the end-of-turn decision (300 ms later) cannot pass.
DELAY_MS = 600
END_SILENCE_MS = 300
CALLER_HZ = 300


@pytest.fixture(autouse=True)
def _nothing_left_running():
    with nothing_left_running():
        yield


def tone(seconds, freq):
    return generate_tone(freq, seconds, SAMPLE_RATE, 0.3)


def silence(seconds):
    return np.zeros(round(seconds * SAMPLE_RATE), dtype=np.int16)


def write_tone(path, freq, seconds, rate=SAMPLE_RATE, stereo=False):
    samples = generate_tone(freq, seconds, rate, 0.3)
    if stereo:
        samples = np.stack([samples, samples], axis=1)
    write_wav(path, samples, rate)
    return path


def write_script(tmp_path, data, name="replies.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def tone_speech(text, voice=None):
    return Speech(samples=generate_tone(450, 0.5, 16000, 0.3), rate=16000)


# Loading a script


def test_read_script_loads_wavs_relative_to_the_yaml_file(tmp_path):
    folder = tmp_path / "prompts"
    folder.mkdir()
    write_tone(folder / "hello.wav", 500, 0.5, rate=16000, stereo=True)
    write_tone(folder / "one.wav", 700, 0.3)
    path = write_script(
        tmp_path,
        {
            "greeting": {"wav": "prompts/hello.wav"},
            "replies": [{"wav": "prompts/one.wav"}, {"say": "Table for four?", "delay_ms": 250}],
        },
    )

    script = read_script(path)

    assert script.greeting.name == "greeting"
    assert script.greeting.kind == "wav"
    assert script.greeting.audio.dtype == np.int16
    assert script.greeting.audio.shape == (4000,)  # 0.5 s, now 8 kHz mono
    assert script.greeting.delay_ms == 0
    first, second = script.replies
    assert (first.name, first.kind, len(first.audio)) == ("reply 1", "wav", 2400)
    assert (second.name, second.kind, second.text) == ("reply 2", "say", "Table for four?")
    assert second.delay_ms == 250
    assert second.audio is None
    assert script.needs_tts
    assert script.then == "hangup"


def test_say_lines_are_synthesized_and_resampled_to_8k(tmp_path):
    path = write_script(
        tmp_path, {"greeting": {"say": "Hello."}, "replies": [{"say": "Bye."}], "then": "silence"}
    )
    asked = []

    def synthesize(text):
        asked.append(text)
        return tone_speech(text)

    script = read_script(path).with_speech(synthesize)

    assert asked == ["Hello.", "Bye."]
    assert not script.needs_tts
    expected = resample(generate_tone(450, 0.5, 16000, 0.3), 16000, SAMPLE_RATE)
    np.testing.assert_array_equal(script.replies[0].audio, expected)
    assert script.greeting.audio.dtype == np.int16
    assert script.then == "silence"


@pytest.mark.parametrize(
    "error", [TtsError("engine fell over"), ValueError("no WAV prompt for text 'Bye.'")]
)
def test_a_tts_failure_names_the_line(tmp_path, error):
    path = write_script(tmp_path, {"replies": [{"say": "Hi."}, {"say": "Bye."}]})

    def synthesize(text):
        if text == "Bye.":
            raise error
        return tone_speech(text)

    with pytest.raises(ScriptError, match=r"reply 2: .*") as info:
        read_script(path).with_speech(synthesize)
    assert str(error) in str(info.value)


def test_tts_with_no_audio_is_an_error(tmp_path):
    path = write_script(tmp_path, {"replies": [{"say": "Hi."}]})
    empty = Speech(samples=np.zeros(0, dtype=np.int16), rate=8000)
    with pytest.raises(ScriptError, match="reply 1: .*no audio"):
        read_script(path).with_speech(lambda text: empty)


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"replies": [{"wav": "a.wav", "say": "Hi."}]}, "reply 1: has both wav and say"),
        ({"replies": [{"delay_ms": 100}]}, "reply 1: needs wav or say"),
        ({"replies": [{"wav": "a.wav"}, {"text": "Hi."}]}, "reply 2: unknown key 'text'"),
        ({"replies": [{"wav": "missing.wav"}]}, "reply 1: no such WAV file"),
        ({"replies": [{"wav": "a.wav"}], "then": "wait"}, "then: must be hangup or silence"),
        ({"replies": [{"wav": "a.wav", "delay_ms": -1}]}, "reply 1: delay_ms must be 0 or more"),
        ({"replies": [{"wav": "a.wav", "delay_ms": "soon"}]}, "reply 1: delay_ms must be a number"),
        ({"replies": [{"wav": "a.wav", "delay_ms": True}]}, "reply 1: delay_ms must be a number"),
        ({"replies": [{"say": "  "}]}, "reply 1: say must be text"),
        ({"replies": [{"wav": 3}]}, "reply 1: wav must be a file path"),
        ({"replies": ["a.wav"]}, "reply 1: must be a mapping"),
        ({"replies": []}, "replies: needs at least one reply"),
        ({"replies": {"wav": "a.wav"}}, "replies: must be a list"),
        ({"greeting": {"wav": "a.wav"}}, "replies: needs at least one reply"),
        ({"greeting": {"say": "Hi.", "wav": "a.wav"}, "replies": [{"wav": "a.wav"}]}, "greeting: "),
        ({"replies": [{"wav": "a.wav"}], "reply": []}, "unknown key 'reply'"),
    ],
)
def test_load_errors_name_the_line_and_the_problem(tmp_path, data, message):
    write_tone(tmp_path / "a.wav", 500, 0.2)
    path = write_script(tmp_path, data)
    with pytest.raises(ScriptError) as info:
        read_script(path)
    assert message in str(info.value)
    assert str(path) in str(info.value)


def test_a_bad_wav_file_names_the_line(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not a wav file at all")
    path = write_script(tmp_path, {"replies": [{"wav": "bad.wav"}]})
    with pytest.raises(ScriptError, match="reply 1: cannot read bad.wav"):
        read_script(path)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("replies: [unclosed", "not valid YAML"),
        ("- just\n- a list\n", "must be a mapping"),
        ("", "must be a mapping"),
    ],
)
def test_a_file_that_is_not_a_script_is_an_error(tmp_path, text, message):
    path = tmp_path / "replies.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ScriptError, match=message):
        read_script(path)


def test_a_missing_script_file_is_an_error(tmp_path):
    with pytest.raises(ScriptError, match="no such script file"):
        read_script(tmp_path / "nope.yaml")


def test_the_agent_refuses_a_script_with_unsynthesized_say_lines(tmp_path):
    script = read_script(write_script(tmp_path, {"replies": [{"say": "Hi."}]}))
    with pytest.raises(ValueError, match="reply 1"):
        asyncio.run(LocalAgent.start(agent_config(script=script)))


def test_the_agent_refuses_a_script_and_a_play_file_together(tmp_path):
    wav = write_tone(tmp_path / "a.wav", 500, 0.2)
    script = read_script(write_script(tmp_path, {"replies": [{"wav": "a.wav"}]}))
    with pytest.raises(ValueError, match="play"):
        asyncio.run(LocalAgent.start(agent_config(script=script, play=wav)))


def test_the_silero_vad_without_the_extra_fails_before_listening(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    script = read_script(write_script(tmp_path, {"replies": [{"say": "Hi."}]}))
    script = script.with_speech(tone_speech)
    with pytest.raises(SttExtraMissingError):
        asyncio.run(LocalAgent.start(agent_config(script=script, vad="silero")))


# Calls


async def scripted_call(agent, directory, send, listen, until_ended=False):
    """Call the agent, send audio, record both sides. Returns (recorder t0,
    whether the agent ended the call)."""
    rec = CallRecorder(directory)
    ended = False
    try:
        rtp = await RtpSession.open(bind_host="127.0.0.1")
        rtp.record(rec)
        try:
            call = await place_call(caller_config(agent), "agent", rtp.local, on_event=rec.event)
            try:
                await rtp.start(call.remote_media)
                rtp.send(send)
                try:
                    async with asyncio.timeout(listen):
                        await call.wait_ended()
                    ended = True
                except TimeoutError:
                    pass
            finally:
                await call.hangup()
        finally:
            await rtp.close()
    finally:
        rec.close()
    return rec.t0, ended


def loud_runs(audio, threshold=2000, max_gap=15):
    """(first frame, last frame) of each run of loud 20 ms frames. A hole
    of up to max_gap frames (300 ms) stays inside a run: a stalled sender
    skips slots, which splits a tone on the wire. Tones in these tests are
    at least 1 s apart."""
    loud = np.flatnonzero(frame_rms(audio) > threshold)
    runs = []
    for i in loud:
        if runs and i - runs[-1][1] <= max_gap:
            runs[-1][1] = i
        else:
            runs.append([i, i])
    return [tuple(r) for r in runs]


def dominant_hz(audio):
    spectrum = np.abs(np.fft.rfft(audio.astype(np.float64)))
    return np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[spectrum.argmax()]


def first_loud_time(audio, threshold=5000):
    return np.flatnonzero(np.abs(audio.astype(np.int32)) > threshold)[0] / SAMPLE_RATE


def last_loud_end(audio, threshold=5000):
    return (np.flatnonzero(np.abs(audio.astype(np.int32)) > threshold)[-1] + 1) / SAMPLE_RATE


def reply_script(tmp_path, then="silence", reply_seconds=(0.4, 0.4), greeting_delay=300):
    write_tone(tmp_path / "greeting.wav", 500, 0.4)
    replies = []
    for i, seconds in enumerate(reply_seconds, 1):
        write_tone(tmp_path / f"reply{i}.wav", 500 + 200 * i, seconds)
        replies.append({"wav": f"reply{i}.wav", "delay_ms": DELAY_MS})
    data = {
        "greeting": {"wav": "greeting.wav", "delay_ms": greeting_delay},
        "replies": replies,
        "then": then,
    }
    return read_script(write_script(tmp_path, data))


def start_scripted(script, **kw):
    return LocalAgent.start(agent_config(script=script, end_silence_ms=END_SILENCE_MS, **kw))


@pytest.mark.realtime
def test_greeting_then_replies_in_order_each_after_its_delay(tmp_path, caplog):
    script = reply_script(tmp_path)
    # Two caller turns, starting and ending in digital silence.
    caller = np.concatenate(
        [silence(1.0), tone(0.5, CALLER_HZ), silence(2.0), tone(0.5, CALLER_HZ), silence(0.5)]
    )

    async def main():
        agent = await start_scripted(script)
        try:
            t0, ended = await scripted_call(agent, tmp_path / "run", caller, listen=5.8)
            return agent.played, t0, ended
        finally:
            await agent.close()

    with caplog.at_level(logging.INFO, logger="voiceqa.telephony.local.agent"):
        played, t0, ended = asyncio.run(main())

    assert not ended
    assert [p.line for p in played] == ["greeting", "reply 1", "reply 2"]
    assert [p.kind for p in played] == ["wav"] * 3
    assert len({p.call_id for p in played}) == 1

    # By the agent's own clock: the first frame goes out in the first slot
    # at or after the turn end (or the answer, for the greeting) plus the delay.
    greeting, *replies = played
    assert 0 <= greeting.at - (greeting.after + 0.3) < FRAME_S + 1e-6
    for p in replies:
        assert 0 <= p.at - (p.after + DELAY_MS / 1000) < FRAME_S + 1e-6

    # The checks below cross from the agent's clock to the caller's
    # recording. Each receive timeline is anchored on a first packet, which
    # a loaded machine can deliver early or late by tens of ms, so they
    # allow `wire` seconds either way (0.15 s for the gap, which crosses
    # twice). The wrong answers are much further off: no delay is 0.6 s
    # early, timing from the decision 0.3 s late, timing from the tone's
    # start 0.5 s early.
    wire = 0.1
    caller_wav, _ = read_wav(tmp_path / "run" / "caller.wav")
    agent_wav, _ = read_wav(tmp_path / "run" / "agent.wav")
    # The turn end the agent timed from is the end of the caller's tone.
    bursts = [(a * FRAME_S, (b + 1) * FRAME_S) for a, b in loud_runs(caller_wav)]
    assert len(bursts) == 2
    for p, (_, end) in zip(replies, bursts, strict=True):
        assert p.after - t0 == pytest.approx(end, abs=wire)

    # On the wire: three tones in agent.wav, in script order, each where
    # the agent said it put it.
    runs = loud_runs(agent_wav)
    assert len(runs) == 3
    for (a, b), freq, p in zip(runs, (500, 700, 900), played, strict=True):
        piece = agent_wav[a * 160 : (b + 1) * 160]
        assert abs(dominant_hz(piece) - freq) < 10
        assert first_loud_time(piece) + a * FRAME_S == pytest.approx(p.at - t0, abs=wire)
    # Caller's tone end to the reply's first sound, as the caller heard it.
    # Measured 0.60 to 0.64 s idle, as low as 0.58 s under load.
    for (a, _), (_, end) in zip(runs[1:], bursts, strict=True):
        gap = first_loud_time(agent_wav[a * 160 :]) + a * FRAME_S - end
        assert abs(gap - DELAY_MS / 1000) <= 0.15

    lines = [r.getMessage() for r in caplog.records if r.name == "voiceqa.telephony.local.agent"]
    played_lines = [m for m in lines if " at " in m and ("reply" in m or "greeting" in m)]
    assert len(played_lines) == 3
    assert "reply 2 (wav) at" in played_lines[2]


@pytest.mark.realtime
def test_a_turn_that_ends_during_a_reply_is_answered_after_it(tmp_path):
    # Reply 1 lasts 2 s. The caller's second turn ends while it plays, so
    # reply 2 must follow reply 1 at once instead of being dropped.
    script = reply_script(tmp_path, reply_seconds=(2.0, 0.4), greeting_delay=0)
    caller = np.concatenate(
        [silence(0.8), tone(0.4, CALLER_HZ), silence(1.2), tone(0.4, CALLER_HZ), silence(0.2)]
    )

    async def main():
        agent = await start_scripted(script)
        try:
            await scripted_call(agent, tmp_path / "run", caller, listen=5.0)
            return agent.played
        finally:
            await agent.close()

    played = asyncio.run(main())

    assert [p.line for p in played] == ["greeting", "reply 1", "reply 2"]
    first, second = played[1:]
    first_end = first.at + 2.0
    # The second turn ended (and was decided) well before reply 1 finished.
    assert second.after + END_SILENCE_MS / 1000 < first_end
    assert second.at == pytest.approx(first_end, abs=FRAME_S + 1e-6)


@pytest.mark.realtime
def test_then_hangup_ends_the_call_once_the_last_reply_has_played(tmp_path):
    script = reply_script(tmp_path, then="hangup", reply_seconds=(0.6,), greeting_delay=0)
    caller = np.concatenate([silence(0.8), tone(0.4, CALLER_HZ), silence(0.2)])

    async def main():
        agent = await start_scripted(script)
        try:
            t0, ended = await scripted_call(agent, tmp_path / "run", caller, listen=6.0)
            return agent.played, t0, ended
        finally:
            await agent.close()

    played, t0, ended = asyncio.run(main())

    assert ended
    assert [p.line for p in played] == ["greeting", "reply 1"]
    agent_wav, _ = read_wav(tmp_path / "run" / "agent.wav")
    # The whole reply arrived before the BYE.
    reply = agent_wav[round((played[1].at - t0 - 0.1) * SAMPLE_RATE) :]
    assert last_loud_end(reply) - first_loud_time(reply) == pytest.approx(0.6, abs=FRAME_S)


@pytest.mark.realtime
def test_then_silence_keeps_the_call_up_and_ignores_later_turns(tmp_path):
    script = reply_script(tmp_path, then="silence", reply_seconds=(0.4,), greeting_delay=0)
    caller = np.concatenate(
        [silence(0.8), tone(0.4, CALLER_HZ), silence(1.5), tone(0.4, CALLER_HZ), silence(0.2)]
    )

    async def main():
        agent = await start_scripted(script)
        try:
            _, ended = await scripted_call(agent, tmp_path / "run", caller, listen=5.0)
            return agent.played, ended
        finally:
            await agent.close()

    played, ended = asyncio.run(main())

    assert not ended
    assert [p.line for p in played] == ["greeting", "reply 1"]
    agent_wav, _ = read_wav(tmp_path / "run" / "agent.wav")
    assert len(loud_runs(agent_wav)) == 2


@pytest.mark.realtime
def test_each_call_starts_the_script_from_the_top(tmp_path):
    script = reply_script(tmp_path, then="silence", greeting_delay=0)
    caller = np.concatenate([silence(0.8), tone(0.4, CALLER_HZ), silence(0.2)])

    async def main():
        agent = await start_scripted(script)
        try:
            await scripted_call(agent, tmp_path / "one", caller, listen=2.5)
            await scripted_call(agent, tmp_path / "two", caller, listen=2.5)
            return agent.played
        finally:
            await agent.close()

    played = asyncio.run(main())

    calls = list(dict.fromkeys(p.call_id for p in played))
    assert len(calls) == 2
    for call_id in calls:
        assert [p.line for p in played if p.call_id == call_id] == ["greeting", "reply 1"]
    for name in ("one", "two"):
        agent_wav, _ = read_wav(tmp_path / name / "agent.wav")
        runs = loud_runs(agent_wav)
        freqs = [dominant_hz(agent_wav[a * 160 : (b + 1) * 160]) for a, b in runs]
        assert [round(f, -2) for f in freqs] == [500, 700]


@pytest.mark.realtime
def test_a_call_is_closed_only_once_its_vad_thread_has_stopped(tmp_path, monkeypatch):
    # A slow VAD close (300 ms) shows whether the agent's call counts as
    # closed before the script has stopped its VAD thread.
    real_close = ThreadedTurnDetector.close
    stopped = []

    def slow_close(self, *args, **kwargs):
        time.sleep(0.3)
        real_close(self, *args, **kwargs)
        stopped.append(True)

    monkeypatch.setattr(ThreadedTurnDetector, "close", slow_close)
    script = reply_script(tmp_path, then="silence", greeting_delay=0)

    async def main():
        agent = await start_scripted(script)
        rtp = await RtpSession.open(bind_host="127.0.0.1")
        try:
            call = await place_call(caller_config(agent), "agent", rtp.local)
            await rtp.start(call.remote_media)
            await asyncio.sleep(0.5)
            (served,) = agent.calls
            await call.hangup()
            # Returns once the agent has closed its side of the call.
            await served.hangup()
            return list(stopped)
        finally:
            await rtp.close()
            await agent.close()

    assert asyncio.run(main()) == [True]


@pytest.mark.realtime
def test_hangup_after_stops_the_vad_thread_before_the_bye(tmp_path, monkeypatch):
    # With --hangup-after the agent hangs up itself, so nothing else cancels
    # the script. Its VAD thread must be stopped (slowly, 300 ms here)
    # before the BYE goes out, or a late turn_end would send on closed media.
    real_close = ThreadedTurnDetector.close
    stopped = []

    def slow_close(self, *args, **kwargs):
        time.sleep(0.3)
        real_close(self, *args, **kwargs)
        stopped.append(True)

    monkeypatch.setattr(ThreadedTurnDetector, "close", slow_close)
    script = reply_script(tmp_path, then="silence", greeting_delay=0)

    async def main():
        agent = await start_scripted(script, hangup_after=0.5)
        rtp = await RtpSession.open(bind_host="127.0.0.1")
        try:
            call = await place_call(caller_config(agent), "agent", rtp.local)
            await rtp.start(call.remote_media)
            async with asyncio.timeout(5):
                await call.wait_ended()
            return list(stopped)
        finally:
            await rtp.close()
            await agent.close()

    assert asyncio.run(main()) == [True]


@pytest.mark.realtime
def test_frames_the_vad_dropped_are_logged_as_a_warning(tmp_path, monkeypatch, caplog):
    # A VAD that falls behind loses frames, and turns can go missing with
    # them; the log says so once the call ends.
    monkeypatch.setattr(ThreadedTurnDetector, "dropped_frames", property(lambda self: 7))
    script = reply_script(tmp_path, then="silence", greeting_delay=0)

    async def main():
        agent = await start_scripted(script, hangup_after=0.3)
        try:
            await scripted_call(agent, tmp_path / "run", silence(1.0), listen=2.0)
        finally:
            await agent.close()

    with caplog.at_level(logging.WARNING, logger="voiceqa.telephony.local.agent"):
        asyncio.run(main())

    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "dropped 7 frames" in warnings[0]


@pytest.mark.realtime
def test_hangup_after_still_ends_a_scripted_call(tmp_path):
    script = reply_script(tmp_path, then="silence", greeting_delay=0)

    async def main():
        agent = await start_scripted(script, hangup_after=0.5)
        try:
            _, ended = await scripted_call(agent, tmp_path / "run", silence(3.0), listen=3.0)
            return agent.played, ended
        finally:
            await agent.close()

    played, ended = asyncio.run(main())

    assert ended
    assert [p.line for p in played] == ["greeting"]
