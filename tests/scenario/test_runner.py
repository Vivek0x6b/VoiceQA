"""The scenario runner against the scripted test agent, in this process.

Each agent line is a tone; ToneStt turns a tone into a known transcript,
so no speech model is needed. The caller's lines are WAV files."""

import asyncio
import gc
import json
import textwrap

import numpy as np
import pytest
from scenario_support import (
    CALLER_HZ,
    EXACT,
    ToneStt,
    agent_config,
    session_for,
    silence,
    tone,
    write_audio,
    write_replies,
)
from tests_support import nothing_left_running, start_rejecting_uas

from voiceqa.scenario import load_scenario
from voiceqa.scenario.runner import SayTtsError, SayWavError, load_say_audio, run_scenario
from voiceqa.speech.fakes import FakeTts
from voiceqa.telephony.local.agent import LocalAgent


@pytest.fixture(autouse=True)
def _nothing_left_running():
    with nothing_left_running():
        yield


GREETING = "Hi, how can I help?"
SORRY = "Sorry, what was that?"
WHAT_TIME = "For what time?"
BOOKED = "You are booked."
TEXTS = {500: GREETING, 700: SORRY, 900: WHAT_TIME, 1100: BOOKED}


def write_scenario(folder, text):
    folder.mkdir(parents=True, exist_ok=True)
    # Starts with silence: a say in the call's first frame would set the
    # scripted agent's energy VAD floor (a known limit of that VAD).
    write_audio(folder / "caller.wav", np.concatenate([silence(0.2), tone(0.5, CALLER_HZ)]))
    path = folder / "scenario.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return load_scenario(path)


def run(tmp_path, scenario, script=None, stt=None, agent=None, **session_kw):
    stt = stt or ToneStt(TEXTS)

    async def main():
        local = await LocalAgent.start(agent or agent_config(script))
        try:
            session = session_for(
                local.local_addr[1],
                tmp_path / "run",
                stt,
                end_silence_ms=scenario.end_silence_ms,
                **session_kw,
            )
            return await run_scenario(scenario, session, load_say_audio(scenario))
        finally:
            await local.close()

    return asyncio.run(main())


BOOKING = """
    name: booking
    defaults: {timeout: 5s, end_silence_ms: 300}
    steps:
      - label: greet
        expect: {any_of: ["how can I help", "how may I help"]}
      - label: ask
        say: {wav: caller.wav, text: "A table for two, please."}
        expect: {contains: what time}
        retries: 1
        on_fail: goto ask-again
      - label: ask-again
        say: {wav: caller.wav, text: "A table for two."}
        expect: {contains: what time}
        on_match: goto confirm
      - label: never
        say: {wav: caller.wav}
        expect: {contains: never heard}
      - label: confirm
        say: {wav: caller.wav, text: "At seven."}
        expect: {contains: booked}
        complete: true
    assert: [call_connected, agent_responded, task_completed]
    """


def booking_agent(tmp_path, replies):
    return write_replies(tmp_path / "agent", greeting=(500, 0.4, 200), replies=replies)


@pytest.mark.realtime
def test_a_run_follows_retries_on_fail_and_gotos_and_passes(tmp_path):
    scenario = write_scenario(tmp_path / "s", BOOKING)
    # "ask" hears "sorry" twice (one retry), so on_fail sends it to
    # "ask-again", whose on_match skips "never".
    script = booking_agent(
        tmp_path,
        [(700, 0.4, 200), (700, 0.4, 200), (900, 0.4, 200), (1100, 0.4, 200)],
    )

    result = run(tmp_path, scenario, script)

    assert result.status == "passed", result.model_dump_json(indent=2)
    assert result.exit_code == 0
    assert result.path == ["greet", "ask", "ask-again", "confirm"]
    greet, ask, again, confirm = result.steps
    assert (greet.status, greet.heard, greet.said, greet.attempts) == ("passed", GREETING, None, 1)
    assert greet.latency_ms is None
    assert ask.status == "rerouted"
    assert ask.attempts == 2
    assert [t.heard for t in ask.tries] == [SORRY, SORRY]
    assert [t.said for t in ask.tries] == ["A table for two, please."] * 2
    assert ask.next == "ask-again"
    assert 'did not hear "what time"' in ask.reason
    assert (again.status, again.heard, again.next) == ("passed", WHAT_TIME, "confirm")
    assert (confirm.status, confirm.heard, confirm.next) == ("passed", BOOKED, "end")
    assert all(s.latency_ms is not None and s.latency_ms > 0 for s in (ask, again, confirm))
    assert [(a.name, a.passed) for a in result.assertions] == [
        ("call_connected", True),
        ("agent_responded", True),
        ("task_completed", True),
    ]
    assert result.call.answered
    assert result.steps_passed == 3 and result.steps_total == 4

    # R2: the caller side of transcript.json is the said text.
    data = json.loads((tmp_path / "run" / "transcript.json").read_text())
    caller = [s["text"] for s in data["segments"] if s["speaker"] == "caller"]
    assert caller == ["A table for two, please."] * 2 + ["A table for two.", "At seven."]


@pytest.mark.realtime
def test_a_failing_expect_stops_the_run_with_the_reason(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: fails
        defaults: {timeout: 5s, end_silence_ms: 300}
        steps:
          - label: greet
            expect: {contains: help}
          - label: ask
            say: {wav: caller.wav, text: "A table for two."}
            expect: {contains: booked}
          - label: later
            say: {wav: caller.wav}
        assert: [call_connected]
        """,
    )
    script = booking_agent(tmp_path, [(700, 0.4, 200)])

    result = run(tmp_path, scenario, script)

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.path == ["greet", "ask"]
    ask = result.steps[1]
    assert ask.status == "failed"
    assert ask.next is None
    assert ask.heard == SORRY
    assert ask.reason == 'contains: did not hear "booked"'
    assert result.stopped == "step 2 (ask) failed"


@pytest.mark.realtime
def test_a_goto_loop_stops_at_max_steps_as_a_failure(tmp_path, monkeypatch):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: loop
        max_steps: 4
        steps:
          - label: again
            say: {wav: caller.wav}
            on_match: goto again
        """,
    )
    freezes = []
    monkeypatch.setattr(gc, "freeze", lambda: freezes.append(1))

    result = run(tmp_path, scenario, agent=agent_config())  # echo agent

    assert result.path == ["again"] * 4
    assert all(s.status == "passed" for s in result.steps)
    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.stopped == "stopped after max_steps (4) steps"
    # G2: only the CLI freezes the heap, never the runner.
    assert freezes == []


@pytest.mark.realtime
def test_a_failed_call_assertion_alone_fails_the_run(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: incomplete
        steps:
          - {label: hello, say: {wav: caller.wav}, on_match: goto end}
          - {label: done, say: {wav: caller.wav}, complete: true}
        assert: [call_connected, task_completed]
        """,
    )

    result = run(tmp_path, scenario, agent=agent_config())  # echo agent

    assert result.path == ["hello"]
    assert result.steps[0].next == "end"
    assert result.stopped is None
    assert [(a.name, a.passed) for a in result.assertions] == [
        ("call_connected", True),
        ("task_completed", False),
    ]
    assert result.status == "failed"


@pytest.mark.realtime
def test_an_unanswered_call_fails_call_connected(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: busy
        steps:
          - {say: {wav: caller.wav}}
        assert: [call_connected, agent_responded]
        """,
    )

    async def main():
        uas = await start_rejecting_uas(status=486, reason="Busy Here")
        try:
            port = uas.transport.get_extra_info("sockname")[1]
            session = session_for(port, tmp_path / "run", ToneStt(TEXTS))
            return await run_scenario(scenario, session, load_say_audio(scenario))
        finally:
            uas.transport.close()

    result = asyncio.run(main())

    assert result.status == "failed"
    assert result.exit_code == 1
    assert result.steps == []
    assert (result.call.answered, result.call.status, result.call.reason) == (
        False,
        486,
        "Busy Here",
    )
    assert [(a.name, a.passed, a.reason) for a in result.assertions] == [
        ("call_connected", False, "call not answered: 486 Busy Here"),
        ("agent_responded", False, "no agent turn was heard"),
    ]
    assert result.stopped == "call not answered: 486 Busy Here"


@pytest.mark.realtime
def test_latency_counts_from_the_last_voiced_frame_not_the_silence_tail(tmp_path):
    # R8: the caller's WAV ends in 300 ms of silence. The agent replies
    # 1000 ms after the end of the caller's tone, far above the agent's
    # 300 ms end of turn, so the gap is the delay.
    folder = tmp_path / "s"
    folder.mkdir()
    write_audio(
        folder / "tail.wav", np.concatenate([silence(0.2), tone(0.6, CALLER_HZ), silence(0.3)])
    )
    path = folder / "scenario.yaml"
    path.write_text(
        textwrap.dedent(
            """
            name: latency
            defaults: {timeout: 5s, end_silence_ms: 300}
            steps:
              - label: ask
                say: {wav: tail.wav, text: question}
                expect: {contains: booked, max_latency_ms: 1300}
            """
        ),
        encoding="utf-8",
    )
    scenario = load_scenario(path)
    script = write_replies(tmp_path / "agent", replies=[(1100, 0.6, 1000)])

    result = run(tmp_path, scenario, script)

    (step,) = result.steps
    (attempt,) = step.tries
    # On our own clock the tail is left out to the frame.
    assert attempt.said_end - attempt.said_voiced_end == pytest.approx(0.3, abs=1e-6)
    # Across the two clocks: 0.15 s either way (a loaded runner moves it
    # both ways); counting from the end of the WAV would be 0.3 s short.
    assert step.latency_ms == pytest.approx(1000, abs=150)
    assert step.status == "passed"


@pytest.mark.realtime
def test_a_step_end_silence_ms_keeps_a_paused_reply_in_one_turn(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: pauses
        defaults: {timeout: 6s, end_silence_ms: 300}
        steps:
          - label: ask
            say: {wav: caller.wav}
            end_silence_ms: 1200
            expect: {contains: booked}
        """,
    )
    paused = np.concatenate([tone(0.5, 1100), silence(0.9), tone(0.5, 1100)])
    script = write_replies(tmp_path / "agent", replies=[(paused, 200)])

    result = run(tmp_path, scenario, script)

    turn = result.steps[0].tries[0].agent_turn
    assert turn is not None, result.model_dump_json(indent=2)
    assert turn.end - turn.start == pytest.approx(1.9, abs=EXACT)


@pytest.mark.realtime
def test_an_stt_failure_is_a_could_not_run_error(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: stt
        defaults: {timeout: 5s, end_silence_ms: 300}
        steps:
          - {label: greet, expect: {contains: help}}
          - {label: next, say: {wav: caller.wav}}
        """,
    )
    script = booking_agent(tmp_path, [(700, 0.4, 200)])

    result = run(tmp_path, scenario, script, stt=ToneStt(TEXTS, error="worker died"))

    assert result.status == "error"
    assert result.exit_code == 3
    assert result.path == ["greet"]
    assert result.steps[0].status == "error"
    assert "worker died" in result.error


@pytest.mark.realtime
def test_no_agent_turn_within_the_timeout_fails_the_step(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: quiet
        defaults: {timeout: 0.6s, end_silence_ms: 300}
        steps:
          - {label: greet, expect: {contains: help}}
        """,
    )

    result = run(tmp_path, scenario, agent=agent_config())  # echo, says nothing

    (step,) = result.steps
    assert step.status == "failed"
    assert step.heard is None
    assert step.reason == "no agent turn within 0.6 s"
    assert result.exit_code == 1


def test_load_say_audio_reads_every_wav_and_speaks_every_text(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: audio
        steps:
          - {say: {wav: caller.wav}}
          - {say: "Hello there."}
          - {expect: {contains: x}}
        """,
    )
    asked = []
    tts = FakeTts(rate=16000)

    def synthesize(text):
        asked.append(text)
        return tts.synthesize(text)

    audio = load_say_audio(scenario, synthesize)

    assert asked == ["Hello there."]
    assert sorted(audio) == [1, 2]
    assert len(audio[1]) == 5600  # caller.wav: 0.7 s at 8 kHz
    assert audio[2].dtype.name == "int16"
    # 12 characters at 50 ms each, resampled to 8 kHz.
    assert len(audio[2]) == round(0.6 * 8000)


def test_load_say_audio_names_the_step_when_a_wav_or_the_tts_fails(tmp_path):
    scenario = write_scenario(
        tmp_path / "s",
        """
        name: audio
        steps:
          - {label: hello, say: "Hello there."}
          - {label: bad, say: {wav: caller.wav}}
        """,
    )
    (tmp_path / "s" / "caller.wav").write_bytes(b"not a wav file at all")

    def broken(text):
        raise RuntimeError("engine fell over")

    with pytest.raises(SayTtsError, match=r"step 1 \(hello\): TTS failed .*engine fell over"):
        load_say_audio(scenario, broken)
    with pytest.raises(SayTtsError, match=r"step 1 \(hello\): no TTS engine"):
        load_say_audio(scenario)

    wav_only = load_scenario(tmp_path / "s" / "scenario.yaml")
    with pytest.raises(SayWavError, match=r"step 2 \(bad\): cannot read .*caller.wav"):
        load_say_audio(wav_only, lambda text: FakeTts().synthesize(text))
