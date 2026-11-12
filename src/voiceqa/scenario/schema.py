"""Scenario file schema and the loaded scenario.

The `...File` models mirror the YAML exactly and refuse unknown keys. The
loader (`voiceqa.scenario.load`) checks the parts that need the whole file
(labels, gotos, variables, regexes, WAV paths) and builds the loaded
`Scenario`, where variables are filled in, WAV paths are absolute, durations
are seconds and each step carries its own timeout and retries.
"""

import math
import re
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Discriminator,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    Tag,
)

END = "end"
ASSERTIONS = ("call_connected", "agent_responded", "task_completed")
DEFAULT_TIMEOUT = 10.0
DEFAULT_END_SILENCE_MS = 700.0
DEFAULT_RETRIES = 0
DEFAULT_MAX_STEPS = 50
DEFAULT_FUZZY_MIN = 0.8

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(s|ms)?")
_GOTO = re.compile(r"goto\s+(\S+)")


def parse_duration(value: Any) -> float:
    """Seconds from `10s`, `500ms`, `1.5s` or a bare number of seconds."""
    hint = "use seconds like 10, 1.5s or 500ms"
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        raise ValueError(f"bad duration {value!r}, {hint}")
    if isinstance(value, str):
        match = _DURATION.fullmatch(value)
        if match is None:
            raise ValueError(f"bad duration {value!r}, {hint}")
        seconds = float(match.group(1)) / (1000 if match.group(2) == "ms" else 1)
    else:
        if not math.isfinite(value):
            raise ValueError(f"bad duration {value!r}, {hint}")
        seconds = float(value)
    if seconds <= 0:
        raise ValueError(f"duration must be above 0, got {value!r}")
    return seconds


def parse_goto(value: str) -> str:
    """The target label of `goto <label>`."""
    match = _GOTO.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"must be 'goto <label>', got {value!r}")
    return match.group(1)


def _non_negative(value: int) -> int:
    if value < 0:
        raise ValueError(f"must be 0 or more, got {value}")
    return value


def _positive(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"must be above 0, got {value}")
    return value


def _at_least_one(value: int) -> int:
    if value < 1:
        raise ValueError(f"must be 1 or more, got {value}")
    return value


def _fuzzy_min(value: float) -> float:
    if not 0 < value <= 1:
        raise ValueError(f"must be above 0 and at most 1, got {value}")
    return value


def _one_or_more(value: Any) -> Any:
    return [value] if isinstance(value, str) else value


def _a_list(value: Any) -> Any:
    if isinstance(value, str):
        raise ValueError("must be a list of phrases")
    return value


def _not_empty_list(value: list[str]) -> list[str]:
    if not value:
        raise ValueError("needs at least one phrase")
    return value


def _variable_value(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"must be text or a number, got {value!r}")
    return str(value)


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty")
    return value


def _one_word(value: str) -> str:
    if not value or any(c.isspace() for c in value):
        raise ValueError(f"must be one word with no spaces, got {value!r}")
    return value


Duration = Annotated[float, BeforeValidator(parse_duration)]
Goto = Annotated[StrictStr, AfterValidator(parse_goto)]
Number = StrictInt | StrictFloat
Phrases = Annotated[list[StrictStr], BeforeValidator(_one_or_more)]


class _File(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FuzzyFile(_File):
    text: StrictStr
    min: Annotated[Number, AfterValidator(_fuzzy_min)] = DEFAULT_FUZZY_MIN


class ExpectFile(_File):
    contains: Phrases = []
    any_of: (
        Annotated[list[StrictStr], BeforeValidator(_a_list), AfterValidator(_not_empty_list)] | None
    ) = None
    not_contains: Phrases = []
    regex: StrictStr | None = None
    fuzzy: FuzzyFile | None = None
    max_latency_ms: Annotated[Number, AfterValidator(_positive)] | None = None
    timeout: Duration | None = None


class SayWavFile(_File):
    wav: Annotated[StrictStr, AfterValidator(_not_blank)]
    # What the WAV says, for the transcript and the results.
    text: Annotated[StrictStr, AfterValidator(_not_blank)] | None = None


# Tags of the say union. They show up in pydantic error locations and the
# loader drops them from messages.
SAY_TAGS = ("say-text", "say-wav")


def _say_kind(value: Any) -> str | None:
    if isinstance(value, str):
        return SAY_TAGS[0]
    if isinstance(value, dict | SayWavFile):
        return SAY_TAGS[1]
    return None


Say = Annotated[
    Annotated[Annotated[StrictStr, AfterValidator(_not_blank)], Tag(SAY_TAGS[0])]
    | Annotated[SayWavFile, Tag(SAY_TAGS[1])],
    Discriminator(
        _say_kind,
        custom_error_type="say_kind",
        custom_error_message="must be text or {wav: path}",
    ),
]


class StepFile(_File):
    label: Annotated[StrictStr, AfterValidator(_one_word)] | None = None
    say: Say | None = None
    expect: ExpectFile | None = None
    retries: Annotated[StrictInt, AfterValidator(_non_negative)] | None = None
    end_silence_ms: Annotated[Number, AfterValidator(_positive)] | None = None
    complete: StrictBool = False
    on_match: Goto | None = None
    on_fail: Goto | None = None


class DefaultsFile(_File):
    timeout: Duration = DEFAULT_TIMEOUT
    end_silence_ms: Annotated[Number, AfterValidator(_positive)] = DEFAULT_END_SILENCE_MS
    retries: Annotated[StrictInt, AfterValidator(_non_negative)] = DEFAULT_RETRIES


class ScenarioFile(_File):
    name: Annotated[StrictStr, AfterValidator(_not_blank)]
    variables: dict[StrictStr, Annotated[str, BeforeValidator(_variable_value)]] = {}
    defaults: DefaultsFile = DefaultsFile()
    max_steps: Annotated[StrictInt, AfterValidator(_at_least_one)] = DEFAULT_MAX_STEPS
    steps: Annotated[list[StepFile], Field(min_length=1)]
    assertions: list[StrictStr] = Field(default=[], alias="assert")


class _Loaded(BaseModel):
    model_config = ConfigDict(frozen=True)


class Fuzzy(_Loaded):
    text: str
    min: float


class Expect(_Loaded):
    """What the agent's turn must hold. Text is after variable substitution.
    Every matcher that is set must pass."""

    contains: tuple[str, ...] = ()
    any_of: tuple[str, ...] = ()
    not_contains: tuple[str, ...] = ()
    regex: str | None = None
    fuzzy: Fuzzy | None = None
    max_latency_ms: float | None = None


class Step(_Loaded):
    """One step. `index` counts from 1. `label` is the label from the file,
    or `step-<index>` when it has none.

    `say_wav` is the absolute path of a WAV to play. `say_text` is the line
    with variables filled in: the text to speak when there is no WAV, else
    the WAV's optional `text:`. `caller_text` is what goes in the transcript
    and the results. `timeout` (seconds) is the step's wait for the agent's
    turn, `retries` how many more times to say the line and listen again
    after a failed expect, `end_silence_ms` the silence that ends the
    agent's turn for this step. `on_match` and `on_fail` are a target label,
    `end`, or None: on_match None goes on to the next step, on_fail None
    stops the run as failed."""

    index: int
    label: str
    say_text: str | None = None
    say_wav: Path | None = None
    expect: Expect | None = None
    timeout: float
    retries: int
    end_silence_ms: float = DEFAULT_END_SILENCE_MS
    complete: bool = False
    on_match: str | None = None
    on_fail: str | None = None

    @property
    def caller_text(self) -> str | None:
        """The said text, or the WAV's file name in brackets when a WAV has
        no `text:`. None for a step that says nothing."""
        if self.say_text is not None:
            return self.say_text
        if self.say_wav is not None:
            return f"[{self.say_wav.name}]"
        return None


class Scenario(_Loaded):
    """A checked scenario. `sha256` is the hash of the file's bytes.
    `variables` holds the file's variables with the overrides applied."""

    name: str
    path: Path
    sha256: str
    variables: dict[str, str]
    end_silence_ms: float
    max_steps: int
    steps: tuple[Step, ...]
    assertions: tuple[str, ...] = ()

    def step(self, label: str) -> Step:
        for step in self.steps:
            if step.label == label:
                return step
        raise KeyError(label)
