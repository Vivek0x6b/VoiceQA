"""The result of one scenario run, as written to result.json.

Times inside a run (`said_at`, `agent_turn`, ...) are seconds since the
recording's zero, the same zero as events.json, transcript.json and the
WAVs. Latencies are in ms. Bump RESULT_VERSION when a field changes
meaning or goes away; adding a field does not need it.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

from voiceqa.scenario.match import MatcherResult

RESULT_VERSION = 1

PASSED = "passed"
FAILED = "failed"
# A failed expect whose on_fail sent the walk to another step. It does
# not fail the run.
REROUTED = "rerouted"
# Could not run: the VAD or STT failed, or the call could not be placed.
ERROR = "error"

EXIT_CODES = {PASSED: 0, FAILED: 1, ERROR: 3}


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True)


class AgentTurnTimes(_Model):
    turn: str
    start: float
    end: float


class Attempt(_Model):
    """One say-and-listen of a step. `heard` is None when no agent turn
    came; `error` says why the attempt failed without matching (timeout,
    call ended, no transcript)."""

    said: str | None = None
    said_at: float | None = None
    said_end: float | None = None
    said_voiced_end: float | None = None
    agent_turn: AgentTurnTimes | None = None
    heard: str | None = None
    latency_ms: float | None = None
    matchers: list[MatcherResult] = []
    passed: bool
    error: str | None = None


class StepResult(_Model):
    """A step as executed. `said`, `heard`, `latency_ms` and `matchers` are
    from the last attempt, `tries` has every attempt. `next` is the label
    the walk went to, "end", or None when the run stopped at this step."""

    index: int
    label: str
    status: Literal["passed", "failed", "rerouted", "error"]
    said: str | None = None
    heard: str | None = None
    latency_ms: float | None = None
    matchers: list[MatcherResult] = []
    reason: str = ""
    attempts: int
    next: str | None = None
    duration: float
    tries: list[Attempt] = []


class AssertionResult(_Model):
    name: str
    passed: bool
    reason: str


class CallInfo(_Model):
    answered: bool
    status: int | None = None
    reason: str | None = None
    duration: float | None = None


class ScenarioInfo(_Model):
    name: str
    path: str
    sha256: str


class ScenarioResult(_Model):
    """`started` is UTC in ISO 8601, `duration` the whole run in seconds.
    `stopped` says why the walk ended early, `error` what could not run."""

    version: int = RESULT_VERSION
    scenario: ScenarioInfo
    variables: dict[str, str] = {}
    started: str
    duration: float
    status: Literal["passed", "failed", "error"]
    error: str | None = None
    stopped: str | None = None
    run_dir: str | None = None
    call: CallInfo
    path: list[str] = []
    steps: list[StepResult] = []
    assertions: list[AssertionResult] = []

    @computed_field
    @property
    def exit_code(self) -> int:
        return EXIT_CODES[self.status]

    @computed_field
    @property
    def steps_passed(self) -> int:
        return sum(s.status == PASSED for s in self.steps)

    @computed_field
    @property
    def steps_total(self) -> int:
        return len(self.steps)
