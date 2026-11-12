"""result.json, the console summary and JUnit XML, from built results."""

import json
import xml.etree.ElementTree as ET

from voiceqa.scenario.match import MatcherResult
from voiceqa.scenario.report import format_summary, write_junit, write_result
from voiceqa.scenario.result import (
    AgentTurnTimes,
    AssertionResult,
    Attempt,
    CallInfo,
    ScenarioInfo,
    ScenarioResult,
    StepResult,
)

CONTAINS_FAIL = MatcherResult(
    matcher="contains", passed=False, expected="booked", reason='did not hear "booked"'
)
ANY_OF_PASS = MatcherResult(
    matcher="any_of", passed=True, expected='"how can I help"', reason='heard "how can i help"'
)


def result(status="failed", steps=None, assertions=None, **kw):
    if steps is None:
        steps = [
            StepResult(
                index=1,
                label="greet",
                status="passed",
                heard="Hi, how can I help?",
                matchers=[ANY_OF_PASS],
                attempts=1,
                next="ask",
                duration=1.5,
                tries=[
                    Attempt(
                        heard="Hi, how can I help?",
                        agent_turn=AgentTurnTimes(turn="agent-1", start=0.3, end=1.2),
                        matchers=[ANY_OF_PASS],
                        passed=True,
                    )
                ],
            ),
            StepResult(
                index=2,
                label="ask",
                status="failed",
                said="A table for two.",
                heard="Sorry, what was that?",
                latency_ms=812.4,
                matchers=[CONTAINS_FAIL],
                reason='contains: did not hear "booked"',
                attempts=2,
                duration=4.25,
                tries=[
                    Attempt(
                        said="A table for two.", passed=False, error="no agent turn within 5 s"
                    ),
                    Attempt(
                        said="A table for two.",
                        said_at=2.0,
                        said_end=3.0,
                        said_voiced_end=2.7,
                        heard="Sorry, what was that?",
                        latency_ms=812.4,
                        matchers=[CONTAINS_FAIL],
                        passed=False,
                    ),
                ],
            ),
        ]
    if assertions is None:
        assertions = [
            AssertionResult(name="call_connected", passed=True, reason="the call was answered"),
            AssertionResult(
                name="task_completed", passed=False, reason="no step with complete: true passed"
            ),
        ]
    args = dict(
        scenario=ScenarioInfo(name="booking", path="/x/scenario.yaml", sha256="ab" * 32),
        variables={"name": "Alex"},
        started="2026-11-11T10:15:30Z",
        duration=7.5,
        status=status,
        stopped="step 2 (ask) failed",
        run_dir="runs/20261111-101530Z-booking",
        call=CallInfo(answered=True, status=200, reason="OK", duration=6.9),
        path=["greet", "ask"],
        steps=steps,
        assertions=assertions,
    )
    args.update(kw)
    return ScenarioResult(**args)


def test_result_json_has_every_field(tmp_path):
    path = write_result(tmp_path, result())

    assert path == tmp_path / "result.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == [
        "version",
        "scenario",
        "variables",
        "started",
        "duration",
        "status",
        "error",
        "stopped",
        "run_dir",
        "call",
        "path",
        "steps",
        "assertions",
        "exit_code",
        "steps_passed",
        "steps_total",
    ]
    assert data["version"] == 1
    assert data["scenario"] == {"name": "booking", "path": "/x/scenario.yaml", "sha256": "ab" * 32}
    assert data["call"] == {"answered": True, "status": 200, "reason": "OK", "duration": 6.9}
    assert (data["exit_code"], data["steps_passed"], data["steps_total"]) == (1, 1, 2)
    step = data["steps"][1]
    assert list(step) == [
        "index",
        "label",
        "status",
        "said",
        "heard",
        "latency_ms",
        "matchers",
        "reason",
        "attempts",
        "next",
        "duration",
        "tries",
    ]
    assert step["matchers"] == [
        {
            "matcher": "contains",
            "passed": False,
            "expected": "booked",
            "reason": 'did not hear "booked"',
            "score": None,
        }
    ]
    assert list(step["tries"][1]) == [
        "said",
        "said_at",
        "said_end",
        "said_voiced_end",
        "agent_turn",
        "heard",
        "latency_ms",
        "matchers",
        "passed",
        "error",
    ]
    assert data["steps"][0]["tries"][0]["agent_turn"] == {
        "turn": "agent-1",
        "start": 0.3,
        "end": 1.2,
    }
    assert data["assertions"][1] == {
        "name": "task_completed",
        "passed": False,
        "reason": "no step with complete: true passed",
    }


def test_result_json_reads_back_into_the_same_result(tmp_path):
    original = result()
    path = write_result(tmp_path, original)
    assert ScenarioResult.model_validate_json(path.read_text(encoding="utf-8")) == original


def test_summary_has_a_line_per_step_then_the_assertions_and_the_verdict():
    lines = format_summary(result()).splitlines()

    assert lines == [
        "scenario  booking (/x/scenario.yaml)",
        'PASS    1  greet             heard "Hi, how can I help?"',
        'FAIL    2  ask               contains: did not hear "booked"; heard "Sorry, what was '
        'that?"; latency 812 ms; 2 attempts',
        "assert    call_connected    pass  the call was answered",
        "assert    task_completed    FAIL  no step with complete: true passed",
        "stopped   step 2 (ask) failed",
        "result    FAILED, 1 of 2 steps passed",
        "run       runs/20261111-101530Z-booking",
    ]


def test_summary_of_a_rerouted_step_and_of_an_error():
    rerouted = StepResult(
        index=3,
        label="ask",
        status="rerouted",
        heard="Sorry?",
        reason='contains: did not hear "time"',
        attempts=1,
        next="ask-again",
        duration=1.0,
    )
    broken = StepResult(
        index=4,
        label="ask-again",
        status="error",
        reason="no transcript for agent-3: worker died",
        attempts=1,
        duration=1.0,
    )
    text = format_summary(
        result(
            status="error",
            steps=[rerouted, broken],
            assertions=[],
            stopped=None,
            error="no transcript for agent-3: worker died",
        )
    )
    lines = text.splitlines()
    assert lines[1] == (
        'ROUTED  3  ask               contains: did not hear "time"; heard "Sorry?"; goto ask-again'
    )
    assert lines[2] == "ERROR   4  ask-again         no transcript for agent-3: worker died"
    assert lines[3] == "error     no transcript for agent-3: worker died"
    assert lines[4] == "result    COULD NOT RUN, 0 of 2 steps passed"


def test_summary_of_an_unanswered_call():
    text = format_summary(
        result(
            steps=[],
            assertions=[],
            path=[],
            stopped="call not answered: 486 Busy Here",
            call=CallInfo(answered=False, status=486, reason="Busy Here"),
        )
    )
    assert text.splitlines()[1:3] == [
        "stopped   call not answered: 486 Busy Here",
        "result    FAILED, 0 of 0 steps passed",
    ]


def test_junit_has_a_testcase_per_step_and_assertion_with_failures(tmp_path):
    path = tmp_path / "out" / "junit.xml"
    path.parent.mkdir()
    write_junit(path, [result()])

    root = ET.parse(path).getroot()
    assert root.tag == "testsuites"
    (suite,) = root.findall("testsuite")
    assert suite.get("name") == "booking"
    assert (suite.get("tests"), suite.get("failures"), suite.get("errors")) == ("4", "2", "0")
    assert suite.get("timestamp") == "2026-11-11T10:15:30Z"
    cases = suite.findall("testcase")
    assert [c.get("name") for c in cases] == [
        "step 1 greet",
        "step 2 ask",
        "assert call_connected",
        "assert task_completed",
    ]
    assert cases[0].find("failure") is None
    failure = cases[1].find("failure")
    assert failure.get("message") == 'contains: did not hear "booked"'
    assert failure.text.splitlines() == [
        "expected: contains booked",
        'heard: "Sorry, what was that?"',
        "said: A table for two.",
        "latency: 812 ms",
        "attempts: 2",
    ]
    assert cases[1].get("time") == "4.250"
    assert cases[3].find("failure").get("message") == "no step with complete: true passed"
    props = {p.get("name"): p.get("value") for p in suite.find("properties")}
    assert props["sha256"] == "ab" * 32
    assert props["run_dir"] == "runs/20261111-101530Z-booking"


def test_junit_marks_a_rerouted_step_as_passed_and_an_error_as_error(tmp_path):
    rerouted = StepResult(
        index=1, label="a", status="rerouted", reason="x", attempts=1, next="b", duration=1.0
    )
    broken = StepResult(
        index=2, label="b", status="error", reason="worker died", attempts=1, duration=1.0
    )
    path = tmp_path / "junit.xml"
    write_junit(path, [result(status="error", steps=[rerouted, broken], assertions=[])])

    suite = ET.parse(path).getroot().find("testsuite")
    a, b = suite.findall("testcase")
    assert a.find("failure") is None and a.find("error") is None
    assert "x" in a.find("system-out").text
    assert b.find("error").get("message") == "worker died"
    assert (suite.get("failures"), suite.get("errors")) == ("0", "1")


def test_junit_reports_a_run_failure_that_no_testcase_shows(tmp_path):
    # Not answered, and call_connected was not asserted.
    path = tmp_path / "junit.xml"
    write_junit(
        path,
        [
            result(
                steps=[],
                assertions=[],
                path=[],
                stopped="call not answered: 486 Busy Here",
                call=CallInfo(answered=False, status=486, reason="Busy Here"),
            )
        ],
    )
    suite = ET.parse(path).getroot().find("testsuite")
    (case,) = suite.findall("testcase")
    assert case.get("name") == "run"
    assert case.find("failure").get("message") == "call not answered: 486 Busy Here"
    assert suite.get("failures") == "1"
