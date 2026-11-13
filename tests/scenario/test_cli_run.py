"""`voiceqa run`: options, --check, pre-dial checks, exit codes, result.json
and JUnit, with the scripted agent on a loop of its own in this process
and a stand-in STT worker (scenario_support.FakeWorker)."""

import json
import re
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scenario_support import (
    CALLER_HZ,
    FakeWorker,
    agent_config,
    agent_in_thread,
    silence,
    tone,
    write_audio,
    write_replies,
)
from tests_support import nothing_left_running
from typer.testing import CliRunner

from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.engine import session as session_module
from voiceqa.scenario import history
from voiceqa.speech import model_cache

runner = CliRunner()

GREETING = "Hi, how can I help?"
BOOKED = "You are booked."
TEXTS = {500: GREETING, 1100: BOOKED}


@pytest.fixture(autouse=True)
def _nothing_left_running():
    with nothing_left_running():
        yield


@pytest.fixture(autouse=True)
def cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICEQA_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def worker(monkeypatch):
    """Replaces the faster-whisper worker (and its pre-dial checks)."""
    made = []

    def make(model, models_dir):
        made.append((model, models_dir))
        fake = FakeWorker(TEXTS)
        made.append(fake)
        return fake

    monkeypatch.setattr(cli, "_stt_worker", make)
    return made


def write_scenario(folder, text):
    folder.mkdir(parents=True, exist_ok=True)
    write_audio(folder / "caller.wav", np.concatenate([silence(0.2), tone(0.5, CALLER_HZ)]))
    path = folder / "scenario.yaml"
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


PASSING = """
    name: Booking, quick
    defaults: {timeout: 5s, end_silence_ms: 300}
    steps:
      - label: greet
        expect: {contains: help}
      - label: book
        say: {wav: caller.wav, text: "A table for two."}
        expect: {contains: booked}
        complete: true
    assert: [call_connected, agent_responded, task_completed]
    """

FAILING = PASSING.replace("contains: booked", "contains: cancelled")


def booking_script(tmp_path):
    return write_replies(tmp_path / "agent", greeting=(500, 0.4, 200), replies=[(1100, 0.4, 200)])


def invoke(args):
    return runner.invoke(app, ["run", *map(str, args)])


def only_run_dir(out):
    (run_dir,) = list(out.iterdir())
    return run_dir


def test_check_prints_ok_for_a_valid_scenario(tmp_path):
    path = write_scenario(tmp_path / "s", PASSING)
    result = invoke([path, "--check"])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output


def test_check_lists_the_problems_and_exits_2(tmp_path):
    path = write_scenario(
        tmp_path / "s", "name: t\nsteps:\n  - {label: a, say: hi, on_match: goto nope}\n"
    )
    result = invoke([path, "--check"])
    assert result.exit_code == 2
    assert "step 1 (a): on_match: no step labelled 'nope'" in result.output


def test_an_invalid_scenario_exits_2_without_check(tmp_path):
    path = write_scenario(tmp_path / "s", "name: t\nsteps: []\n")
    result = invoke([path, "--target", "sip:agent@127.0.0.1:5999"])
    assert result.exit_code == 2
    assert "needs at least one step" in result.output


def test_var_overrides_reach_the_scenario_and_bad_ones_exit_2(tmp_path):
    path = write_scenario(tmp_path / "s", "name: t\nsteps:\n  - {say: 'hi {who}'}\n")
    assert invoke([path, "--check"]).exit_code == 2
    assert invoke([path, "--check", "--var", "who=Alex"]).exit_code == 0
    bad = invoke([path, "--check", "--var", "who"])
    assert bad.exit_code == 2
    assert "--var" in bad.output


def test_run_without_a_target_exits_2(tmp_path):
    path = write_scenario(tmp_path / "s", PASSING)
    result = invoke([path])
    assert result.exit_code == 2
    assert "--target" in result.output


def test_a_missing_stt_extra_exits_3_before_dialing(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"
    result = invoke([path, "--target", "sip:agent@127.0.0.1:5999", "--out", out, "--vad", "energy"])
    assert result.exit_code == 3
    assert "[stt] extra" in result.output
    assert not out.exists()


def test_a_tts_failure_exits_3_before_dialing(tmp_path, worker):
    path = write_scenario(
        tmp_path / "s",
        "name: t\nsteps:\n  - {say: Hello there.}\n",
    )
    index = tmp_path / "prompts.yaml"
    index.write_text("Something else.: other.wav\n", encoding="utf-8")
    out = tmp_path / "runs"
    result = invoke(
        [path, "--target", "sip:agent@127.0.0.1:5999", "--out", out, "--vad", "energy"]
        + ["--tts", "wav", "--prompts", index]
    )
    assert result.exit_code == 3
    assert "step 1 (step-1): TTS failed for 'Hello there.'" in result.output
    assert not out.exists()


def test_an_stt_model_that_fails_to_load_exits_3_before_dialing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "_stt_worker", lambda model, models_dir: FakeWorker(TEXTS, load_error="bad model")
    )
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"
    result = invoke([path, "--target", "sip:agent@127.0.0.1:5999", "--out", out, "--vad", "energy"])
    assert result.exit_code == 3
    assert "bad model" in result.output
    assert not out.exists()


@pytest.mark.realtime
def test_a_passing_run_exits_0_and_writes_result_json_and_junit(tmp_path, worker, monkeypatch):
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"
    junit = tmp_path / "junit.xml"
    seen = {}
    real = cli.CallSession

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(cli, "CallSession", spy)

    with agent_in_thread(agent_config(booking_script(tmp_path))) as port:
        result = invoke(
            [path, "--target", f"sip:agent@127.0.0.1:{port}", "--out", out]
            + ["--junit", junit, "--vad", "energy", "--stt-model", "tiny.en"]
        )

    assert result.exit_code == 0, result.output
    # G2: the CLI, which owns its process, asks for the heap freeze.
    assert seen["freeze_heap"] is True
    assert worker[0] == ("tiny.en", None)
    run_dir = only_run_dir(out)
    assert run_dir.name.endswith("Z-booking-quick")
    data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert data["status"] == "passed"
    assert data["exit_code"] == 0
    assert data["path"] == ["greet", "book"]
    assert data["steps"][1]["said"] == "A table for two."
    assert data["steps"][1]["heard"] == BOOKED
    assert data["run_dir"] == str(run_dir)
    for name in ("caller.wav", "agent.wav", "mix.wav", "events.json", "transcript.json"):
        assert (run_dir / name).is_file()
    lines = result.output.splitlines()
    assert any(line.startswith("PASS    2  book") for line in lines)
    assert "result    PASSED, 2 of 2 steps passed" in lines
    suite = ET.parse(junit).getroot().find("testsuite")
    assert suite.get("name") == "Booking, quick"
    assert (suite.get("tests"), suite.get("failures")) == ("5", "0")
    (row,) = history.list_runs()
    assert (row.status, row.exit_code, row.steps_passed, row.steps_total) == ("passed", 0, 2, 2)
    assert row.run_dir == str(run_dir.resolve())


@pytest.mark.realtime
def test_a_failing_run_exits_1_with_the_failure_in_console_and_junit(tmp_path, worker):
    path = write_scenario(tmp_path / "s", FAILING)
    out = tmp_path / "runs"
    junit = tmp_path / "junit.xml"

    with agent_in_thread(agent_config(booking_script(tmp_path))) as port:
        result = invoke(
            [path, "--target", f"sip:agent@127.0.0.1:{port}", "--out", out]
            + ["--junit", junit, "--vad", "energy"]
        )

    assert result.exit_code == 1, result.output
    assert 'FAIL    2  book              contains: did not hear "cancelled"' in result.output
    assert "result    FAILED, 1 of 2 steps passed" in result.output
    data = json.loads((only_run_dir(out) / "result.json").read_text(encoding="utf-8"))
    assert data["status"] == "failed"
    failure = ET.parse(junit).getroot().find("testsuite/testcase[@name='step 2 book']/failure")
    assert failure.get("message") == 'contains: did not hear "cancelled"'
    assert f'heard: "{BOOKED}"' in failure.text


def test_an_unanswered_call_exits_1_and_still_writes_result_json(tmp_path, worker, monkeypatch):
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"

    class NotAnswered(session_module.CallSession):
        async def _dial(self):
            self.status, self.reason = 486, "Busy Here"

    monkeypatch.setattr(cli, "CallSession", NotAnswered)
    result = invoke([path, "--target", "sip:agent@127.0.0.1:5999", "--out", out, "--vad", "energy"])

    assert result.exit_code == 1, result.output
    assert "call not answered: 486 Busy Here" in result.output
    data = json.loads((only_run_dir(out) / "result.json").read_text(encoding="utf-8"))
    assert data["call"]["answered"] is False
    assert data["assertions"][0] == {
        "name": "call_connected",
        "passed": False,
        "reason": "call not answered: 486 Busy Here",
    }
    (row,) = history.list_runs()
    assert (row.scenario_name, row.status, row.exit_code) == ("Booking, quick", "failed", 1)


@pytest.mark.realtime
def test_a_could_not_run_error_during_the_call_exits_3(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "_stt_worker", lambda model, models_dir: FakeWorker(TEXTS, error="worker died")
    )
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"

    with agent_in_thread(agent_config(booking_script(tmp_path))) as port:
        result = invoke(
            [path, "--target", f"sip:agent@127.0.0.1:{port}", "--out", out, "--vad", "energy"]
        )

    assert result.exit_code == 3, result.output
    assert "COULD NOT RUN" in result.output
    data = json.loads((only_run_dir(out) / "result.json").read_text(encoding="utf-8"))
    assert data["status"] == "error"


def test_the_vad_fallback_notice_is_printed_before_dialing(tmp_path, worker, monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    path = write_scenario(tmp_path / "s", PASSING)

    class NotAnswered(session_module.CallSession):
        async def _dial(self):
            self.reason = "nobody home"

    monkeypatch.setattr(cli, "CallSession", NotAnswered)
    result = invoke([path, "--target", "sip:agent@127.0.0.1:5999", "--out", tmp_path / "runs"])

    assert "using the energy VAD" in result.output
    assert result.output.index("using the energy VAD") < result.output.index("result")


def test_a_junit_path_in_a_missing_folder_exits_2_before_dialing(tmp_path, worker):
    path = write_scenario(tmp_path / "s", PASSING)
    out = tmp_path / "runs"
    result = invoke(
        [path, "--target", "sip:agent@127.0.0.1:5999", "--out", out]
        + ["--junit", tmp_path / "nope" / "junit.xml", "--vad", "energy"]
    )
    assert result.exit_code == 2
    assert "--junit" in result.output
    assert not out.exists()


def test_run_help_lists_its_options():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    # Rich styles the help when it thinks colour is wanted (on CI, for one),
    # and puts a style change between the two dashes of an option.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in (
        "--target",
        "--server",
        "--user",
        "--tts",
        "--voice",
        "--prompts",
        "--stt-model",
        "--models-dir",
        "--vad",
        "--var",
        "--out",
        "--junit",
        "--check",
    ):
        assert option in plain
