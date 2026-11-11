"""Reading and checking a scenario file.

Every problem names where it is: `step <n> (<label>)` for a step (n counts
from 1), the key path for the rest. Problems are collected rather than
stopping at the first one, except that the checks needing the whole file
only run once the file has the right shape.
"""

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from voiceqa.scenario.schema import (
    ASSERTIONS,
    END,
    SAY_TAGS,
    Expect,
    ExpectFile,
    Fuzzy,
    SayWavFile,
    Scenario,
    ScenarioFile,
    Step,
    StepFile,
)
from voiceqa.scenario.variables import NAME, UnknownVariableError, substitute
from voiceqa.speech.wer import normalize_words


class ScenarioError(ValueError):
    """An invalid scenario. `problems` lists each one without the path."""

    def __init__(self, path: Path, problems: Sequence[str]):
        self.path = path
        self.problems = list(problems)
        lines = "\n".join("  " + p for p in self.problems)
        super().__init__(f"{path}: invalid scenario:\n{lines}")


def check_scenario(path: str | Path, overrides: Mapping[str, str] | None = None) -> list[str]:
    """Every problem with the scenario, or an empty list when it is valid."""
    try:
        load_scenario(path, overrides)
    except ScenarioError as exc:
        return exc.problems
    return []


def load_scenario(path: str | Path, overrides: Mapping[str, str] | None = None) -> Scenario:
    """Read and check a scenario. `overrides` are variables that win over
    the file's own (`voiceqa run --var`). Raises ScenarioError."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ScenarioError(path, [f"no such scenario file: {path}"]) from None
    except OSError as exc:
        raise ScenarioError(path, [f"cannot read it: {exc}"]) from None
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ScenarioError(path, [f"cannot read it: {exc}"]) from None
    except yaml.YAMLError as exc:
        raise ScenarioError(path, [f"not valid YAML: {exc}"]) from None
    if not isinstance(data, dict):
        raise ScenarioError(path, ["scenario: must be a mapping with name and steps"])
    try:
        file = ScenarioFile.model_validate(data)
    except ValidationError as exc:
        raise ScenarioError(path, [_describe(e, data) for e in exc.errors()]) from None

    problems: list[str] = []
    scenario = _build(file, path, hashlib.sha256(raw).hexdigest(), overrides or {}, problems)
    if problems:
        raise ScenarioError(path, problems)
    return scenario


def _step_name(index: int, label: Any) -> str:
    return f"step {index} ({label})" if isinstance(label, str) and label else f"step {index}"


def _describe(error: Mapping[str, Any], data: dict) -> str:
    loc = [part for part in error["loc"] if part not in SAY_TAGS]
    message = error["msg"].removeprefix("Value error, ")
    if error["type"] == "extra_forbidden":
        message = f"unknown key {loc.pop()!r}"
    elif error["type"] == "missing":
        message = f"missing key {loc.pop()!r}"
    elif error["type"] == "too_short" and loc == ["steps"]:
        message = "needs at least one step"

    where = "scenario"
    if len(loc) >= 2 and loc[0] == "steps" and isinstance(loc[1], int):
        steps = data.get("steps")
        item = steps[loc[1]] if isinstance(steps, list) and loc[1] < len(steps) else None
        label = item.get("label") if isinstance(item, dict) else None
        where = _step_name(loc[1] + 1, label)
        loc = loc[2:]
    field = ""
    for part in loc:
        field += f"[{part}]" if isinstance(part, int) else ("." if field else "") + str(part)
    if where == "scenario" and field:
        return f"{field}: {message}"
    return f"{where}: {field}: {message}" if field else f"{where}: {message}"


def _bad_name(name: str) -> bool:
    return NAME.fullmatch(name) is None


def _build(
    file: ScenarioFile,
    path: Path,
    sha256: str,
    overrides: Mapping[str, str],
    problems: list[str],
) -> Scenario:
    hint = "use letters, digits and underscores"
    for name in file.variables:
        if _bad_name(name):
            problems.append(f"variables: bad name {name!r}, {hint}")
    for name in overrides:
        if _bad_name(name):
            problems.append(f"variable override: bad name {name!r}, {hint}")
    variables = {**file.variables, **overrides}
    folder = path.resolve().parent

    labels = [s.label or f"step-{i}" for i, s in enumerate(file.steps, 1)]
    first_index: dict[str, int] = {}
    for i, label in enumerate(labels, 1):
        where = _step_name(i, file.steps[i - 1].label)
        if label == END:
            problems.append(f"{where}: label {END!r} is reserved")
        elif label in first_index:
            problems.append(f"{where}: duplicate label, also step {first_index[label]}")
        else:
            first_index[label] = i

    steps = []
    for i, (item, label) in enumerate(zip(file.steps, labels, strict=True), 1):
        where = _step_name(i, item.label)

        def report(field: str, problem: str, where: str = where) -> None:
            problems.append(f"{where}: {field}: {problem}" if field else f"{where}: {problem}")

        steps.append(_build_step(item, i, label, file, variables, folder, first_index, report))

    for name in file.assertions:
        if name not in ASSERTIONS:
            expected = f"{', '.join(ASSERTIONS[:-1])} or {ASSERTIONS[-1]}"
            problems.append(f"assert: unknown assertion {name!r}, expected {expected}")
    if "task_completed" in file.assertions and not any(s.complete for s in file.steps):
        problems.append("assert: task_completed needs a step with complete: true")

    return Scenario(
        name=file.name,
        path=path.resolve(),
        sha256=sha256,
        variables=variables,
        end_silence_ms=file.defaults.end_silence_ms,
        max_steps=file.max_steps,
        steps=tuple(steps),
        assertions=tuple(file.assertions),
    )


def _build_step(
    item: StepFile,
    index: int,
    label: str,
    file: ScenarioFile,
    variables: Mapping[str, str],
    folder: Path,
    labels: Mapping[str, int],
    report: Callable[..., None],
) -> Step:
    def fill(field: str, text: str, escape: Callable[[str], str] | None = None) -> str | None:
        try:
            return substitute(text, variables, escape)
        except UnknownVariableError as exc:
            report(field, str(exc))
            return None

    if item.say is None and item.expect is None:
        report("", "needs say or expect")
    if item.expect is None:
        if item.on_fail is not None:
            report("", "on_fail needs an expect")
        if item.retries:
            report("", "retries needs an expect")
    for field, target in (("on_match", item.on_match), ("on_fail", item.on_fail)):
        if target is not None and target != END and target not in labels:
            report(field, f"no step labelled {target!r}")

    say_text = say_wav = None
    if isinstance(item.say, SayWavFile):
        say_wav = (folder / item.say.wav).resolve()
        if not say_wav.is_file():
            report("say.wav", f"no such WAV file: {say_wav}")
    elif item.say is not None:
        say_text = fill("say", item.say) or item.say

    expect = None
    if item.expect is not None:
        expect = _build_expect(item.expect, fill, report)
    timeout = file.defaults.timeout
    if item.expect is not None and item.expect.timeout is not None:
        timeout = item.expect.timeout
    retries = item.retries if item.retries is not None else file.defaults.retries
    return Step(
        index=index,
        label=label,
        say_text=say_text,
        say_wav=say_wav,
        expect=expect,
        timeout=timeout,
        retries=retries if expect is not None else 0,
        complete=item.complete,
        on_match=item.on_match,
        on_fail=item.on_fail,
    )


def _build_expect(
    raw: ExpectFile,
    fill: Callable[..., str],
    report: Callable[..., None],
) -> Expect:
    # A text with an unknown variable is reported by fill and kept as
    # written, so the loaded model stays buildable while problems collect.
    def words(field: str, texts: list[str]) -> tuple[str, ...]:
        result = []
        for text in texts:
            filled = fill(field, text)
            if filled is not None and not normalize_words(filled):
                report(field, f"{filled!r} has no words after normalising")
            result.append(text if filled is None else filled)
        return tuple(result)

    regex = None
    if raw.regex is not None:
        regex = fill("expect.regex", raw.regex, re.escape)
        if regex is None:
            regex = raw.regex
        else:
            try:
                re.compile(regex)
            except re.error as exc:
                report("expect.regex", f"bad regex {regex!r}: {exc}")
    fuzzy = None
    if raw.fuzzy is not None:
        (text,) = words("expect.fuzzy.text", [raw.fuzzy.text])
        fuzzy = Fuzzy(text=text, min=raw.fuzzy.min)
    return Expect(
        contains=words("expect.contains", raw.contains),
        any_of=words("expect.any_of", raw.any_of or []),
        not_contains=words("expect.not_contains", raw.not_contains),
        regex=regex,
        fuzzy=fuzzy,
        max_latency_ms=raw.max_latency_ms,
    )
