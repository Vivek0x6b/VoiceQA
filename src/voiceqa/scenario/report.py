"""Writing a scenario result out: result.json, the console summary and
JUnit XML (xml.etree only).

format_summary works from a ScenarioResult alone, so the same text can be
shown later from a stored result.json."""

import xml.etree.ElementTree as ET
from collections.abc import Sequence
from pathlib import Path

from voiceqa.scenario.result import ERROR, FAILED, PASSED, REROUTED, ScenarioResult, StepResult

RESULT_FILE = "result.json"

_TAGS = {PASSED: "PASS", FAILED: "FAIL", REROUTED: "ROUTED", ERROR: "ERROR"}
_VERDICTS = {PASSED: "PASSED", FAILED: "FAILED", ERROR: "COULD NOT RUN"}


def write_result(run_dir: str | Path, result: ScenarioResult) -> Path:
    path = Path(run_dir) / RESULT_FILE
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _step_detail(step: StepResult) -> str:
    parts = []
    if step.status != PASSED and step.reason:
        parts.append(step.reason)
    if step.heard is not None:
        parts.append(f'heard "{step.heard}"')
    if step.latency_ms is not None:
        parts.append(f"latency {step.latency_ms:.0f} ms")
    if step.attempts > 1:
        parts.append(f"{step.attempts} attempts")
    if step.status == REROUTED and step.next is not None:
        parts.append(f"goto {step.next}")
    return "; ".join(parts)


def format_summary(result: ScenarioResult) -> str:
    """One line per executed step, then the call assertions, why the run
    stopped or could not run, the verdict and the run folder."""
    lines = [f"{'scenario':<10}{result.scenario.name} ({result.scenario.path})"]
    for step in result.steps:
        tag = _TAGS[step.status]
        lines.append(f"{tag:<7}{step.index:>2}  {step.label:<16}  {_step_detail(step)}".rstrip())
    for a in result.assertions:
        lines.append(f"{'assert':<10}{a.name:<18}{'pass' if a.passed else 'FAIL':<6}{a.reason}")
    if result.stopped:
        lines.append(f"{'stopped':<10}{result.stopped}")
    if result.error:
        lines.append(f"{'error':<10}{result.error}")
    verdict = _VERDICTS[result.status]
    lines.append(
        f"{'result':<10}{verdict}, {result.steps_passed} of {result.steps_total} steps passed"
    )
    if result.run_dir:
        lines.append(f"{'run':<10}{result.run_dir}")
    return "\n".join(lines)


def _failure_text(step: StepResult) -> str:
    lines = [f"expected: {m.matcher} {m.expected}" for m in step.matchers if not m.passed]
    if step.heard is not None:
        lines.append(f'heard: "{step.heard}"')
    if step.said is not None:
        lines.append(f"said: {step.said}")
    if step.latency_ms is not None:
        lines.append(f"latency: {step.latency_ms:.0f} ms")
    lines.append(f"attempts: {step.attempts}")
    return "\n".join(lines)


def _suite(result: ScenarioResult) -> ET.Element:
    suite = ET.Element("testsuite", name=result.scenario.name, timestamp=result.started)
    props = ET.SubElement(suite, "properties")
    for name, value in (
        ("scenario_path", result.scenario.path),
        ("sha256", result.scenario.sha256),
        ("status", result.status),
        ("run_dir", result.run_dir or ""),
    ):
        ET.SubElement(props, "property", name=name, value=value)
    failures = errors = 0

    def case(name: str, time: float = 0.0) -> ET.Element:
        return ET.SubElement(
            suite, "testcase", classname=result.scenario.name, name=name, time=f"{time:.3f}"
        )

    for step in result.steps:
        el = case(f"step {step.index} {step.label}", step.duration)
        if step.status == FAILED:
            failures += 1
            failure = ET.SubElement(el, "failure", message=step.reason, type="expect")
            failure.text = _failure_text(step)
        elif step.status == ERROR:
            errors += 1
            ET.SubElement(el, "error", message=step.reason, type="error")
        elif step.status == REROUTED:
            out = ET.SubElement(el, "system-out")
            out.text = f"failed, rerouted by on_fail: {step.reason}; goto {step.next}"
    for a in result.assertions:
        el = case(f"assert {a.name}")
        if not a.passed:
            failures += 1
            ET.SubElement(el, "failure", message=a.reason, type="assertion")
    if result.status != PASSED and failures + errors == 0:
        # The run failed with nothing above showing it (say, an unanswered
        # call without call_connected asserted, or max_steps reached).
        el = case("run", result.duration)
        message = result.error or result.stopped or result.status
        if result.status == ERROR:
            errors += 1
            ET.SubElement(el, "error", message=message, type="error")
        else:
            failures += 1
            ET.SubElement(el, "failure", message=message, type="run")
    suite.set("tests", str(len(suite.findall("testcase"))))
    suite.set("failures", str(failures))
    suite.set("errors", str(errors))
    suite.set("skipped", "0")
    suite.set("time", f"{result.duration:.3f}")
    return suite


def write_junit(path: str | Path, results: Sequence[ScenarioResult]) -> Path:
    """JUnit XML with one testsuite per scenario run: a testcase per
    executed step and per call assertion."""
    root = ET.Element("testsuites", name="voiceqa")
    totals = {"tests": 0, "failures": 0, "errors": 0}
    duration = 0.0
    for result in results:
        suite = _suite(result)
        root.append(suite)
        for key in totals:
            totals[key] += int(suite.get(key))
        duration += result.duration
    for key, value in totals.items():
        root.set(key, str(value))
    root.set("time", f"{duration:.3f}")
    tree = ET.ElementTree(root)
    ET.indent(tree)
    path = Path(path)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path
