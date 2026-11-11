import asyncio
import gc
import json
import threading
import weakref

import numpy as np
import pytest
from test_call import start_rejecting_uas

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.engine import session as session_module
from voiceqa.engine.session import (
    AGENT_SPEECH_START,
    AGENT_TURN_END,
    CALLER_SAY_END,
    CALLER_SAY_START,
    OVERLAP,
    VAD_ERROR,
    CallEnded,
    CallNotAnswered,
    CallSession,
    TurnDetectionError,
    _Say,
)
from voiceqa.engine.turns import TurnTap
from voiceqa.speech import model_cache
from voiceqa.speech.models import Segment, Transcript
from voiceqa.speech.vad import (
    ENERGY_FALLBACK_NOTICE,
    SPEECH_START,
    EnergyVad,
    TurnDetector,
    TurnEvent,
)
from voiceqa.speech.worker import SttWorker
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.media import SAMPLE_RATE, media_clock

FRAME_S = 0.02
# One frame plus a little: energy turns on tones that start and end on
# frame boundaries come out exact, and RTP timestamps keep relative times.
EXACT = FRAME_S + 0.005
SCRIPTED = Transcript(engine="fake", model="echo", language="en")


def noise(seconds, seed=0):
    # Low noise for the energy VAD's floor; digital silence never sets it.
    rng = np.random.default_rng(seed)
    return rng.integers(-30, 31, round(seconds * SAMPLE_RATE)).astype(np.int16)


def tone(seconds, freq=1000):
    return generate_tone(freq, seconds, SAMPLE_RATE, 0.3)


def clip(*parts):
    return np.concatenate(parts)


class EchoStt:
    """In-process stand-in for an SttWorker running FakeStt(echo=True)."""

    def __init__(self):
        self.jobs = []

    async def transcribe(self, samples, rate):
        self.jobs.append(len(samples))
        text = f"{len(samples)} samples at {rate} Hz"
        segment = Segment(start=0.0, end=len(samples) / rate, text=text)
        return SCRIPTED.model_copy(update={"segments": [segment]})


async def start_agent(tmp_path, prompt=None, **kw):
    args = dict(host="127.0.0.1", port=0, timer_t1=0.02)
    if prompt is not None:
        path = tmp_path / "prompt.wav"
        write_wav(path, prompt, SAMPLE_RATE)
        args["play"] = path
    args.update(kw)
    return await LocalAgent.start(AgentConfig(**args))


def session_for(agent, record_dir, **kw):
    args = dict(stt=EchoStt(), vad="energy")
    args.update(kw)
    return CallSession(
        target="agent",
        server_host="127.0.0.1",
        server_port=agent.local_addr[1],
        username="1001",
        password="",
        record_dir=record_dir,
        **args,
    )


def events_of(record_dir):
    return json.loads((record_dir / "events.json").read_text())["events"]


# Two agent turns from the WAV playback agent: 600 ms and 400 ms of tone
# with 1 s of noise between, more than the 700 ms end of turn.
TWO_TURNS = clip(noise(0.4), tone(0.6), noise(1.0), tone(0.4), noise(1.0))


def test_session_finds_agent_turns_and_transcribes_each_in_the_worker(tmp_path):
    record_dir = tmp_path / "run"

    async def main():
        agent = await start_agent(tmp_path, TWO_TURNS)
        try:
            async with SttWorker("fake", transcript=SCRIPTED.model_dump(), echo=True) as worker:
                session = session_for(agent, record_dir, stt=worker)
                async with session:
                    first = await session.next_agent_turn(timeout=10)
                    second = await session.next_agent_turn(timeout=10)
                return session, first, second
        finally:
            await agent.close()

    session, first, second = asyncio.run(main())

    assert (first.turn, second.turn) == ("agent-1", "agent-2")
    assert first.end - first.start == pytest.approx(0.6, abs=EXACT)
    assert second.start - first.end == pytest.approx(1.0, abs=EXACT)
    assert second.end - second.start == pytest.approx(0.4, abs=EXACT)
    assert len(first.samples) == round((first.end - first.start) * SAMPLE_RATE)
    # The turn's audio is the tone, not silence or noise.
    assert np.abs(first.samples.astype(np.int32)).max() > 5000
    # Each job is the turn plus 200 ms either side, so the echo tells them apart.
    n1 = round((first.end - first.start + 0.4) * SAMPLE_RATE)
    n2 = round((second.end - second.start + 0.4) * SAMPLE_RATE)
    assert first.text == f"{n1} samples at 8000 Hz"
    assert second.text == f"{n2} samples at 8000 Hz"
    assert first.stt_error is None

    events = events_of(record_dir)
    turn_events = [(e["event"], e["turn"]) for e in events if "turn" in e]
    assert turn_events == [
        (AGENT_SPEECH_START, "agent-1"),
        (AGENT_TURN_END, "agent-1"),
        (AGENT_SPEECH_START, "agent-2"),
        (AGENT_TURN_END, "agent-2"),
    ]
    t0 = session._rec.t0
    starts = [e["t"] for e in events if e["event"] == AGENT_SPEECH_START]
    assert starts == [
        pytest.approx(first.start - t0, abs=1e-5),
        pytest.approx(second.start - t0, abs=1e-5),
    ]
    stopped = [e for e in events if e["event"] == "vad_stopped"]
    assert stopped == [
        {"t": stopped[0]["t"], "event": "vad_stopped", "vad": "energy", "dropped_frames": 0}
    ]

    data = json.loads((record_dir / "transcript.json").read_text())
    agent_segments = [s for s in data["segments"] if s["speaker"] == "agent"]
    assert [s["turn"] for s in agent_segments] == ["agent-1", "agent-2"]
    assert agent_segments[0]["text"] == f"{n1} samples at 8000 Hz"
    # Placed where the job's audio began: 200 ms before the turn.
    assert agent_segments[0]["start"] == pytest.approx(first.start - 0.2 - t0, abs=1e-6)
    assert data["engine"] == "fake"
    assert session.result.answered


def test_say_reports_slot_times_and_lands_in_events_and_transcript(tmp_path):
    record_dir = tmp_path / "run"
    audio = clip(noise(0.2), tone(0.5))

    async def main():
        agent = await start_agent(tmp_path)  # echo agent
        try:
            session = session_for(agent, record_dir)
            async with session:
                before = media_clock()
                said = await session.say(audio, text="hello there")
                after = media_clock()
            return session, said, before, after
        finally:
            await agent.close()

    session, said, before, after = asyncio.run(main())

    assert said.turn == "caller-1"
    frames = len(audio) // 160
    assert said.last_frame - said.first_frame == pytest.approx((frames - 1) * FRAME_S, abs=1e-6)
    assert said.end == pytest.approx(said.last_frame + FRAME_S)
    # Slots of this call, not times from zero or from another clock.
    assert before - 0.1 < said.first_frame < said.end < after + 0.1

    t0 = session._rec.t0
    events = {e["event"]: e for e in events_of(record_dir) if e.get("turn") == "caller-1"}
    assert events[CALLER_SAY_START]["t"] == pytest.approx(said.first_frame - t0, abs=1e-5)
    assert events[CALLER_SAY_END]["t"] == pytest.approx(said.end - t0, abs=1e-5)

    data = json.loads((record_dir / "transcript.json").read_text())
    caller = [s for s in data["segments"] if s["speaker"] == "caller"]
    assert caller == [
        {
            "start": pytest.approx(said.first_frame - t0, abs=1e-6),
            "end": pytest.approx(said.end - t0, abs=1e-6),
            "text": "hello there",
            "words": [],
            "speaker": "caller",
            "turn": "caller-1",
        }
    ]


class ReplyAgent(LocalAgent):
    """Waits for the caller's turn to end (energy VAD, 300 ms end of turn)
    and starts its reply tone exactly DELAY seconds after the end of the
    caller's last speech frame, by the agent's own frame times."""

    DELAY = 1.5
    LEAD = 0.3
    REPLY = clip(noise(LEAD, seed=1), tone(0.6, 700), noise(0.3, seed=2))

    async def _on_call(self, call):
        ends = asyncio.Queue()
        detector = TurnDetector(EnergyVad(), end_silence_ms=300)

        class Sink:
            def sent(self, at, samples):
                pass

            def received(self, at, arrived, samples):
                for event in detector.feed(at, samples):
                    if event.kind == "turn_end":
                        ends.put_nowait(event.at)

        call.media.record(Sink())
        while not call.ended:
            end = await ends.get()
            # The reply's tone begins after LEAD seconds of noise.
            await asyncio.sleep(max(0.0, end + self.DELAY - self.LEAD - media_clock()))
            call.media.send(self.REPLY)


def test_gap_from_say_end_to_agent_speech_is_the_agents_delay(tmp_path):
    delay = ReplyAgent.DELAY

    async def main():
        agent = await ReplyAgent.start(AgentConfig(host="127.0.0.1", port=0, timer_t1=0.02))
        try:
            session = session_for(agent, tmp_path / "run")
            async with session:
                said = await session.say(clip(noise(0.3), tone(1.0)), text="question")
                turn = await session.next_agent_turn(timeout=10)
            return said, turn
        finally:
            await agent.close()

    said, turn = asyncio.run(main())

    # Agent sends on its next 20 ms slot, and both receive timelines are
    # anchored on a loopback arrival: a few tens of ms at most, far below
    # the 1.5 s a bug measuring from the wrong end would be off by.
    assert turn.start - said.end == pytest.approx(delay, abs=0.15)
    assert turn.end - turn.start == pytest.approx(0.6, abs=EXACT)


def test_overlap_is_recorded_when_the_agent_starts_during_a_say(tmp_path):
    record_dir = tmp_path / "run"
    agent_prompt = clip(noise(1.0), tone(1.0), noise(1.0))

    async def main():
        agent = await start_agent(tmp_path, agent_prompt)
        try:
            session = session_for(agent, record_dir)
            async with session:
                await session.say(clip(noise(0.2), tone(2.8)))
                await session.next_agent_turn(timeout=10)
        finally:
            await agent.close()

    asyncio.run(main())

    overlaps = [e for e in events_of(record_dir) if e["event"] == OVERLAP]
    assert len(overlaps) == 1
    assert overlaps[0]["agent_turn"] == "agent-1"
    assert overlaps[0]["caller_turn"] == "caller-1"
    assert overlaps[0]["started_by"] == "agent"
    start = next(e["t"] for e in events_of(record_dir) if e["event"] == AGENT_SPEECH_START)
    assert overlaps[0]["t"] == pytest.approx(start, abs=1e-6)


def test_overlap_is_recorded_when_we_start_while_the_agent_speaks(tmp_path):
    record_dir = tmp_path / "run"
    agent_prompt = clip(noise(0.3), tone(3.0), noise(1.0))

    async def main():
        agent = await start_agent(tmp_path, agent_prompt)
        try:
            session = session_for(agent, record_dir)
            async with session:
                await asyncio.sleep(1.5)
                said = await session.say(tone(0.4))
                await session.next_agent_turn(timeout=10)
            return session, said
        finally:
            await agent.close()

    session, said = asyncio.run(main())

    overlaps = [e for e in events_of(record_dir) if e["event"] == OVERLAP]
    assert [(e["started_by"], e["caller_turn"]) for e in overlaps] == [("caller", "caller-1")]
    assert overlaps[0]["t"] == pytest.approx(said.first_frame - session._rec.t0, abs=1e-5)


class Events:
    def __init__(self):
        self.events = []

    def event(self, name, at=None, **detail):
        self.events.append((name, at, detail))

    def sent(self, at, samples):
        pass

    def received(self, at, arrived, samples):
        pass


def test_overlap_counts_a_say_that_began_before_the_late_speech_start():
    # speech_start reaches the loop after min_speech_ms, naming an earlier
    # time; a say that started in between overlaps and was second to start.
    session = CallSession("agent", "127.0.0.1", 5060, "1001", "", "unused", vad="energy")
    session._rec = Events()
    session._tap = TurnTap(session._rec)
    session._says = [_Say("caller-1", None, first=5000.05)]
    session._on_turn_event(TurnEvent(SPEECH_START, 5000.0))
    overlaps = [(at, d) for name, at, d in session._rec.events if name == OVERLAP]
    assert overlaps == [
        (5000.05, {"agent_turn": "agent-1", "caller_turn": "caller-1", "started_by": "caller"})
    ]


def test_far_end_hangup_ends_the_open_turn_at_its_last_speech_frame(tmp_path):
    record_dir = tmp_path / "run"
    # The agent hangs up 1.5 s after answering, in the middle of its tone.
    agent_prompt = clip(noise(0.3), tone(3.0))

    async def main():
        agent = await start_agent(tmp_path, agent_prompt, hangup_after=1.5)
        try:
            session = session_for(agent, record_dir)
            async with session:
                turn = await session.next_agent_turn(timeout=10)
                with pytest.raises(CallEnded):
                    await session.next_agent_turn(timeout=10)
                with pytest.raises(CallEnded):
                    await session.say(tone(0.2))
            return session, turn
        finally:
            await agent.close()

    session, turn = asyncio.run(main())

    # Without an end-of-stream turn_end there would be no turn at all. The
    # turn ends where the last tone sample in agent.wav is.
    t0 = session._rec.t0
    agent_wav, _ = read_wav(record_dir / "agent.wav")
    loud = np.flatnonzero(np.abs(agent_wav.astype(np.int32)) > 5000)
    assert turn.start - t0 == pytest.approx(loud[0] / SAMPLE_RATE, abs=EXACT)
    assert turn.end - t0 == pytest.approx((loud[-1] + 1) / SAMPLE_RATE, abs=EXACT)
    # The tone was cut by the BYE, well before its 3 s.
    assert turn.end - turn.start < 2.5
    ends = [e for e in events_of(record_dir) if e["event"] == AGENT_TURN_END]
    assert len(ends) == 1
    assert ends[0]["t"] == pytest.approx(turn.end - t0, abs=1e-5)


def test_hangup_during_a_say_ends_it_at_the_last_frame_sent(tmp_path):
    record_dir = tmp_path / "run"

    async def main():
        agent = await start_agent(tmp_path, hangup_after=1.0)  # echo agent
        try:
            session = session_for(agent, record_dir)
            async with session:
                with pytest.raises(CallEnded):
                    await session.say(tone(4.0), text="a long question")
        finally:
            await agent.close()

    asyncio.run(main())

    events = [e for e in events_of(record_dir) if e.get("turn") == "caller-1"]
    assert [e["event"] for e in events] == [CALLER_SAY_START, CALLER_SAY_END]
    assert events[1]["interrupted"] is True
    # caller.wav holds what went out; the say ends where its tone does.
    caller_wav, _ = read_wav(record_dir / "caller.wav")
    loud = np.flatnonzero(np.abs(caller_wav.astype(np.int32)) > 5000)
    assert events[1]["t"] == pytest.approx((loud[-1] + 1) / SAMPLE_RATE, abs=EXACT)
    assert events[1]["t"] - events[0]["t"] < 3.0
    data = json.loads((record_dir / "transcript.json").read_text())
    caller = [s for s in data["segments"] if s["speaker"] == "caller"]
    assert caller[0]["end"] == pytest.approx(events[1]["t"], abs=1e-5)


class ThreadNotingVad(EnergyVad):
    def __init__(self):
        super().__init__()
        self.threads = set()

    def process(self, frame):
        self.threads.add(threading.get_ident())
        return super().process(frame)


def test_the_vad_runs_off_the_event_loop_thread(tmp_path):
    vad = ThreadNotingVad()

    async def main():
        agent = await start_agent(tmp_path, TWO_TURNS)
        try:
            async with session_for(agent, tmp_path / "run", vad=vad) as session:
                await session.next_agent_turn(timeout=10)
            return threading.get_ident()
        finally:
            await agent.close()

    loop_thread = asyncio.run(main())

    assert vad.threads and loop_thread not in vad.threads


class BrokenVad:
    name = "broken"

    def process(self, frame):
        raise RuntimeError("model broke")

    def reset(self):
        pass


def test_a_failing_vad_fails_next_agent_turn_at_once(tmp_path):
    record_dir = tmp_path / "run"

    async def main():
        agent = await start_agent(tmp_path, TWO_TURNS)
        try:
            session = session_for(agent, record_dir, vad=BrokenVad())
            async with session:
                # The agent never hangs up, so without the early error this
                # would be a TimeoutError after the full minute.
                with pytest.raises(TurnDetectionError, match="model broke"):
                    await session.next_agent_turn(timeout=60)
            return session
        finally:
            await agent.close()

    session = asyncio.run(main())

    assert str(session.vad_error) == "model broke"
    errors = [e for e in events_of(record_dir) if e["event"] == VAD_ERROR]
    assert [e["error"] for e in errors] == ["model broke"]


def test_next_agent_turn_times_out_when_the_agent_says_nothing(tmp_path):
    async def main():
        agent = await start_agent(tmp_path, noise(3.0))
        try:
            async with session_for(agent, tmp_path / "run") as session:
                with pytest.raises(TimeoutError):
                    await session.next_agent_turn(timeout=0.5)
        finally:
            await agent.close()

    asyncio.run(main())


class Cycle:
    pass


def test_heap_is_collected_and_frozen_before_dialing_and_released_after(tmp_path, monkeypatch):
    # Python 3.12 arrives here with a few hundred objects already frozen by
    # something outside voiceqa; the session's unfreeze releases those too.
    before = gc.get_freeze_count()
    seen = {}
    real_place_call = session_module.place_call

    async def place_call(*args, **kwargs):
        seen["frozen at dial"] = gc.get_freeze_count()
        seen["cycle alive at dial"] = ref() is not None
        return await real_place_call(*args, **kwargs)

    monkeypatch.setattr(session_module, "place_call", place_call)
    # Garbage left by setup (a reference cycle) must be collected, not frozen.
    cycle = Cycle()
    cycle.me = cycle
    ref = weakref.ref(cycle)
    del cycle

    async def main():
        agent = await start_agent(tmp_path, noise(2.0))
        try:
            async with session_for(agent, tmp_path / "run", freeze_heap=True) as session:
                seen["frozen in call"] = gc.get_freeze_count()
                with pytest.raises(TimeoutError):
                    await session.next_agent_turn(timeout=0.3)
        finally:
            await agent.close()

    asyncio.run(main())

    assert seen["frozen at dial"] > before + 1000
    assert seen["frozen in call"] > before + 1000
    assert seen["cycle alive at dial"] is False
    assert gc.get_freeze_count() == 0


def test_library_use_leaves_the_gc_heap_alone(tmp_path, monkeypatch):
    """freeze_heap is off by default: a program that embeds CallSession
    keeps its own GC state, and an earlier freeze of its own survives."""
    seen = {}
    real_place_call = session_module.place_call

    async def place_call(*args, **kwargs):
        seen["frozen at dial"] = gc.get_freeze_count()
        return await real_place_call(*args, **kwargs)

    monkeypatch.setattr(session_module, "place_call", place_call)
    gc.freeze()
    own = gc.get_freeze_count()
    assert own > 0
    try:

        async def main():
            agent = await start_agent(tmp_path, noise(2.0))
            try:
                async with session_for(agent, tmp_path / "run") as session:
                    with pytest.raises(TimeoutError):
                        await session.next_agent_turn(timeout=0.3)
            finally:
                await agent.close()

        asyncio.run(main())
        assert seen["frozen at dial"] == own
        assert gc.get_freeze_count() == own
    finally:
        gc.unfreeze()


def test_auto_vad_fallback_notice_is_exposed_before_dialing(monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    session = CallSession("agent", "127.0.0.1", 5060, "1001", "", "unused", vad="auto")
    assert session.vad_name == "energy"
    assert session.vad_notice == ENERGY_FALLBACK_NOTICE
    assert "\n" not in session.vad_notice

    named = CallSession("agent", "127.0.0.1", 5060, "1001", "", "unused", vad="energy")
    assert named.vad_name == "energy"
    assert named.vad_notice is None


def test_unanswered_call_raises_and_still_records(tmp_path):
    record_dir = tmp_path / "run"

    async def main():
        uas = await start_rejecting_uas(status=486, reason="Busy Here")
        try:
            port = uas.transport.get_extra_info("sockname")[1]
            session = CallSession("agent", "127.0.0.1", port, "1001", "", record_dir, vad="energy")
            with pytest.raises(CallNotAnswered) as info:
                await session.start()
            return session, info.value
        finally:
            uas.transport.close()

    session, error = asyncio.run(main())

    assert (error.status, error.reason) == (486, "Busy Here")
    assert session.result.answered is False
    assert "call_rejected" in [e["event"] for e in events_of(record_dir)]
    assert not (record_dir / "transcript.json").exists()
