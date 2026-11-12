"""Walking a scenario's steps over one call.

For each step: say its line (if any), wait for the agent's next turn up to
the step's timeout, check the expect, then retry, or go to on_match /
on_fail / the next step. A step with only an expect listens first (a
greeting). A step without an expect does not wait for the agent.

- A failed expect is retried (say again, listen again) `retries` times.
  Still failing, it goes to on_fail; without on_fail the run stops as
  failed. A failure routed by on_fail shows as "rerouted" and does not
  fail the run.
- on_match goes to its label, or to the next step when not set. "end"
  stops the walk.
- Running more than max_steps steps (a goto loop) stops the run as failed.
- Response latency is from the end of the last voiced frame of what we
  said (Said.voiced_end) to the start of the agent's turn (VAD onset), on
  media_clock. A turn that ended before our latest say started is an
  answer to something earlier and is skipped.
- A VAD or STT failure, or a call that cannot be placed, stops the run
  with status "error" (exit code 3).

The runner never touches GC state; only the CLI asks the session to
freeze the heap.
"""

import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from voiceqa.audio.dsp import resample
from voiceqa.engine.session import (
    AgentTurn,
    CallEnded,
    CallNotAnswered,
    CallSession,
    Said,
    TurnDetectionError,
)
from voiceqa.scenario.match import match_expect
from voiceqa.scenario.result import (
    ERROR,
    FAILED,
    PASSED,
    REROUTED,
    AgentTurnTimes,
    AssertionResult,
    Attempt,
    CallInfo,
    ScenarioInfo,
    ScenarioResult,
    StepResult,
)
from voiceqa.scenario.schema import END, Scenario, Step
from voiceqa.speech.interfaces import Speech
from voiceqa.telephony.local.agent import load_prompt
from voiceqa.telephony.media import SAMPLE_RATE


class SayWavError(ValueError):
    """A step's WAV cannot be read: bad input (exit code 2)."""


class SayTtsError(RuntimeError):
    """The TTS engine failed on a step's line: could not run (exit code 3)."""


def needs_tts(scenario: Scenario) -> bool:
    return any(s.say_text is not None and s.say_wav is None for s in scenario.steps)


def load_say_audio(
    scenario: Scenario, synthesize: Callable[[str], Speech] | None = None
) -> dict[int, np.ndarray]:
    """Every step's audio, int16 8 kHz mono, by step index, made before
    dialing so a TTS problem never costs a call. `synthesize` speaks a text
    line; it is only needed when the scenario has text lines."""
    audio: dict[int, np.ndarray] = {}
    for step in scenario.steps:
        where = f"step {step.index} ({step.label})"
        if step.say_wav is not None:
            try:
                samples = load_prompt(step.say_wav)
            except (OSError, EOFError, ValueError) as exc:
                raise SayWavError(f"{where}: cannot read {step.say_wav}: {exc}") from None
        elif step.say_text is not None:
            if synthesize is None:
                raise SayTtsError(f"{where}: no TTS engine for {step.say_text!r}")
            try:
                speech = synthesize(step.say_text)
            except Exception as exc:
                raise SayTtsError(f"{where}: TTS failed for {step.say_text!r}: {exc}") from None
            samples = resample(speech.samples, speech.rate, SAMPLE_RATE)
        else:
            continue
        if len(samples) == 0:
            raise SayTtsError(f"{where}: no audio for {step.caller_text!r}")
        audio[step.index] = samples
    return audio


class _CouldNotRun(Exception):
    pass


class _Walk:
    def __init__(
        self, scenario: Scenario, session: CallSession, audio: Mapping[int, np.ndarray]
    ) -> None:
        self.scenario = scenario
        self.session = session
        self.audio = audio
        self.steps: list[StepResult] = []
        self.path: list[str] = []
        self.completed = False
        self.stopped: str | None = None
        self.error: str | None = None
        self.call_ended = False
        self.last_said: Said | None = None

    def rel(self, t: float | None) -> float | None:
        return None if t is None else t - self.session.t0

    async def run(self) -> None:
        step: Step | None = self.scenario.steps[0]
        while step is not None:
            if len(self.steps) >= self.scenario.max_steps:
                self.stopped = f"stopped after max_steps ({self.scenario.max_steps}) steps"
                return
            self.path.append(step.label)
            try:
                result = await self.run_step(step)
            except _CouldNotRun:
                return
            self.steps.append(result)
            if result.status == FAILED:
                self.stopped = self.stopped or f"step {step.index} ({step.label}) failed"
                return
            if result.next == END or result.next is None:
                return
            step = self.scenario.step(result.next)

    async def run_step(self, step: Step) -> StepResult:
        self.session.set_end_silence_ms(step.end_silence_ms)
        started = time.monotonic()
        tries: list[Attempt] = []
        while True:
            try:
                attempt = await self.attempt(step)
            except _CouldNotRun as exc:
                tries.append(exc.args[0])
                self.steps.append(self.finish(step, tries, ERROR, None, started))
                raise
            tries.append(attempt)
            if attempt.passed:
                if step.complete:
                    self.completed = True
                return self.finish(step, tries, PASSED, self.next_after(step), started)
            if self.call_ended:
                self.stopped = "the call ended"
                return self.finish(step, tries, FAILED, None, started)
            if len(tries) <= step.retries:
                continue
            if step.on_fail is None:
                return self.finish(step, tries, FAILED, None, started)
            return self.finish(step, tries, REROUTED, step.on_fail, started)

    def next_after(self, step: Step) -> str:
        if step.on_match is not None:
            return step.on_match
        if step.index < len(self.scenario.steps):
            return self.scenario.steps[step.index].label
        return END

    def finish(
        self, step: Step, tries: list[Attempt], status: str, next_label: str | None, started: float
    ) -> StepResult:
        last = tries[-1]
        reason = ""
        if not last.passed:
            reason = last.error or "; ".join(
                f"{m.matcher}: {m.reason}" for m in last.matchers if not m.passed
            )
        return StepResult(
            index=step.index,
            label=step.label,
            status=status,
            said=last.said,
            heard=last.heard,
            latency_ms=last.latency_ms,
            matchers=last.matchers,
            reason=reason,
            attempts=len(tries),
            next=next_label,
            duration=time.monotonic() - started,
            tries=tries,
        )

    async def attempt(self, step: Step) -> Attempt:
        said_fields: dict = {}
        if step.index in self.audio:
            try:
                said = await self.session.say(self.audio[step.index], text=step.caller_text)
            except CallEnded:
                self.call_ended = True
                return Attempt(said=step.caller_text, passed=False, error="the call ended")
            self.last_said = said
            said_fields = dict(
                said=step.caller_text,
                said_at=self.rel(said.first_frame),
                said_end=self.rel(said.end),
                said_voiced_end=self.rel(said.voiced_end),
            )
        if step.expect is None:
            return Attempt(passed=True, **said_fields)
        since = self.last_said.first_frame if self.last_said is not None else None
        try:
            turn = await self.session.next_agent_turn(step.timeout, since=since)
        except TimeoutError:
            return Attempt(
                passed=False, error=f"no agent turn within {step.timeout:g} s", **said_fields
            )
        except CallEnded:
            self.call_ended = True
            return Attempt(
                passed=False, error="the call ended before the agent answered", **said_fields
            )
        except TurnDetectionError as exc:
            self.error = str(exc)
            raise _CouldNotRun(Attempt(passed=False, error=str(exc), **said_fields)) from None
        times = AgentTurnTimes(turn=turn.turn, start=self.rel(turn.start), end=self.rel(turn.end))
        if turn.stt_error is not None:
            self.error = f"no transcript for {turn.turn}: {turn.stt_error}"
            raise _CouldNotRun(
                Attempt(passed=False, error=self.error, agent_turn=times, **said_fields)
            )
        latency_ms = self.latency_ms(turn)
        match = match_expect(step.expect, turn.text, latency_ms)
        return Attempt(
            agent_turn=times,
            heard=turn.text,
            latency_ms=latency_ms,
            matchers=list(match.results),
            passed=match.passed,
            **said_fields,
        )

    def latency_ms(self, turn: AgentTurn) -> float | None:
        if self.last_said is None:
            return None
        return (turn.start - self.last_said.voiced_end) * 1000


def _assertions(scenario: Scenario, session: CallSession, walk: _Walk) -> list[AssertionResult]:
    results = []
    for name in scenario.assertions:
        if name == "call_connected":
            if session.answered:
                passed, reason = True, "the call was answered"
            else:
                passed, reason = False, _not_answered(session)
        elif name == "agent_responded":
            count = session.agent_turn_count if session.answered else 0
            passed = count > 0
            reason = f"{count} agent turn(s) heard" if passed else "no agent turn was heard"
        else:  # task_completed
            passed = walk.completed
            reason = "a complete step passed" if passed else "no step with complete: true passed"
        results.append(AssertionResult(name=name, passed=passed, reason=reason))
    return results


def _not_answered(session: CallSession) -> str:
    detail = f"{session.status} {session.reason}" if session.status else (session.reason or "")
    return f"call not answered: {detail}".strip()


async def run_scenario(
    scenario: Scenario, session: CallSession, audio: Mapping[int, np.ndarray]
) -> ScenarioResult:
    """Place the call (the session must not be started), walk the steps,
    hang up and check the call assertions. `audio` is load_say_audio()'s
    output. Never raises for a failed step, an unanswered call or a VAD/STT
    failure; those are in the result."""
    started = datetime.now(UTC)
    clock = time.monotonic()
    walk = _Walk(scenario, session, audio)
    error = stopped = None
    try:
        await session.start()
    except CallNotAnswered:
        stopped = _not_answered(session)
    except OSError as exc:
        error = f"could not place the call: {exc}"
    else:
        try:
            await walk.run()
        finally:
            try:
                await session.close()
            except Exception as exc:
                error = f"hangup failed: {exc}"
        stopped = walk.stopped
        error = walk.error or error

    assertions = _assertions(scenario, session, walk)
    if error is not None:
        status = ERROR
    elif (
        # A failed step, an ended call and max_steps all stop the walk.
        not session.answered or walk.stopped is not None or not all(a.passed for a in assertions)
    ):
        status = FAILED
    else:
        status = PASSED
    result = session.result
    return ScenarioResult(
        scenario=ScenarioInfo(name=scenario.name, path=str(scenario.path), sha256=scenario.sha256),
        variables=scenario.variables,
        started=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration=time.monotonic() - clock,
        status=status,
        error=error,
        stopped=stopped,
        run_dir=str(Path(session.record_dir)),
        call=CallInfo(
            answered=session.answered,
            status=session.status,
            reason=session.reason,
            duration=result.duration if result is not None else None,
        ),
        path=walk.path,
        steps=walk.steps,
        assertions=assertions,
    )
