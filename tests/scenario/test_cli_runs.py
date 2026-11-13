"""The history row `voiceqa run` writes, and `voiceqa runs list` / `runs
show`. The call itself is replaced by a canned result here (the real
call path is covered in test_cli_run.py), so nothing goes over sockets."""

import shutil
import sqlite3
import textwrap
from contextlib import closing

import pytest
from scenario_support import FakeWorker, silence, write_audio
from typer.testing import CliRunner

from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.scenario import history
from voiceqa.scenario.report import format_summary
from voiceqa.scenario.result import (
    AssertionResult,
    CallInfo,
    ScenarioInfo,
    ScenarioResult,
    StepResult,
)
from voiceqa.speech import model_cache
from voiceqa.speech.worker import SttWorkerError

runner = CliRunner()

SCENARIO = """
    name: Booking
    steps:
      - label: book
        say: {wav: caller.wav}
        expect: {contains: booked}
"""


@pytest.fixture
def scenario(tmp_path):
    folder = tmp_path / "s"
    folder.mkdir()
    write_audio(folder / "caller.wav", silence(0.2))
    path = folder / "scenario.yaml"
    path.write_text(textwrap.dedent(SCENARIO), encoding="utf-8")
    return path


def sql(statement, params=()):
    with closing(sqlite3.connect(history.db_path())) as conn:
        conn.execute(statement, params)
        conn.commit()


def canned(status="passed", started="2026-11-12T10:00:00Z", run_dir=None, name="Booking"):
    step = "passed" if status == "passed" else "failed"
    return ScenarioResult(
        scenario=ScenarioInfo(name=name, path="/s/scenario.yaml", sha256="cd" * 32),
        started=started,
        duration=3.25,
        status=status,
        error="the STT worker stopped" if status == "error" else None,
        stopped=None if status == "passed" else "step 1 (book) failed",
        run_dir=run_dir,
        call=CallInfo(answered=True, status=200, reason="OK", duration=3.0),
        path=["book"],
        steps=[
            StepResult(
                index=1,
                label="book",
                status=step,
                heard="You are booked." if step == "passed" else "Sorry?",
                reason="" if step == "passed" else 'contains: did not hear "booked"',
                attempts=1,
                duration=1.5,
            )
        ],
        assertions=[AssertionResult(name="call_connected", passed=True, reason="answered")],
    )


def fake_call(monkeypatch, status="passed", raises=None):
    """Replace the call with one that makes the run folder and returns a
    canned result (or raises once the folder exists)."""
    monkeypatch.setattr(cli, "_stt_worker", lambda model, models_dir: FakeWorker({}))

    async def run(loaded, session, audio, worker):
        session.record_dir.mkdir(parents=True)
        if raises is not None:
            raise raises
        return canned(status, run_dir=str(session.record_dir))

    monkeypatch.setattr(cli, "_run_with_worker", run)


def run(scenario, out, *extra):
    args = ["run", str(scenario), "--target", "sip:agent@127.0.0.1:5999", "--out", str(out)]
    return runner.invoke(app, [*args, "--vad", "energy", *map(str, extra)])


def runs(*args):
    return runner.invoke(app, ["runs", *map(str, args)])


# The row voiceqa run writes


@pytest.mark.parametrize("status, code", [("passed", 0), ("failed", 1), ("error", 3)])
def test_run_writes_a_history_row_with_its_exit_code(monkeypatch, scenario, tmp_path, status, code):
    fake_call(monkeypatch, status)
    result = run(scenario, tmp_path / "runs")

    assert result.exit_code == code, result.output
    (row,) = history.list_runs()
    assert (row.status, row.exit_code, row.scenario_name) == (status, code, "Booking")
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert row.run_dir == str(run_dir.resolve())
    assert history.load_result(row) == canned(status, run_dir=str(run_dir))


def test_a_junit_write_failure_is_stored_with_exit_code_3(monkeypatch, scenario, tmp_path):
    fake_call(monkeypatch, "passed")
    junit = tmp_path / "junit.xml"
    junit.mkdir()  # a folder in the way, so the write fails after the run
    result = run(scenario, tmp_path / "runs", "--junit", junit)

    assert result.exit_code == 3, result.output
    (row,) = history.list_runs()
    assert (row.status, row.exit_code) == ("passed", 3)


@pytest.mark.parametrize(
    "raised, message",
    [(SttWorkerError("worker died"), "worker died"), (KeyboardInterrupt(), "run interrupted")],
)
def test_a_run_that_could_not_finish_still_gets_a_row(
    monkeypatch, scenario, tmp_path, raised, message
):
    fake_call(monkeypatch, raises=raised)
    result = run(scenario, tmp_path / "runs")

    assert result.exit_code == 3, result.output
    (row,) = history.list_runs()
    assert (row.status, row.exit_code, row.steps_total) == ("error", 3, 0)
    stored = history.load_result(row)
    assert stored.error == message
    assert stored.scenario.name == "Booking"
    assert stored.call.answered is False
    (run_dir,) = (tmp_path / "runs").iterdir()
    assert stored.run_dir == str(run_dir)
    assert stored.started.endswith("Z")


def test_no_row_when_the_run_fails_before_dialing(monkeypatch, scenario, tmp_path):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    result = run(scenario, tmp_path / "runs")
    assert result.exit_code == 3, result.output
    assert not history.db_path().exists()


def test_check_writes_no_row(scenario):
    result = runner.invoke(app, ["run", str(scenario), "--check"])
    assert result.exit_code == 0, result.output
    assert not history.db_path().exists()


@pytest.mark.parametrize("status, code", [("passed", 0), ("failed", 1), ("error", 3)])
def test_a_db_that_cannot_be_written_warns_and_keeps_the_exit_code(
    monkeypatch, scenario, tmp_path, status, code
):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv("VOICEQA_DATA_DIR", str(blocker))
    fake_call(monkeypatch, status)
    result = run(scenario, tmp_path / "runs")

    assert result.exit_code == code, result.output
    assert "warning: run history not saved:" in result.stderr
    assert "result" in result.stdout


def test_a_newer_db_schema_warns_with_the_version_and_keeps_the_exit_code(
    monkeypatch, scenario, tmp_path
):
    history.record_run(canned())
    sql("PRAGMA user_version = 9")
    fake_call(monkeypatch, "failed")
    result = run(scenario, tmp_path / "runs")

    assert result.exit_code == 1, result.output
    assert "warning: run history not saved:" in result.stderr
    assert "schema version 9" in result.stderr


# voiceqa runs list


def test_runs_list_prints_newest_first(scenario):
    history.record_run(canned("passed", "2026-11-12T10:00:00Z"))
    history.record_run(canned("failed", "2026-11-12T12:00:00Z", name="Cancel"))
    history.record_run(canned("error", "2026-11-12T11:00:00Z"), exit_code=3)

    result = runs("list")
    assert result.exit_code == 0, result.output
    assert result.stdout.splitlines() == [
        "id  started               status  exit  steps  duration  scenario",
        " 2  2026-11-12T12:00:00Z  failed     1    0/1     3.2 s  Cancel",
        " 3  2026-11-12T11:00:00Z  error      3    0/1     3.2 s  Booking",
        " 1  2026-11-12T10:00:00Z  passed     0    1/1     3.2 s  Booking",
    ]


def test_runs_list_filters_and_limits():
    history.record_run(canned("passed", "2026-11-12T10:00:00Z"))
    history.record_run(canned("failed", "2026-11-12T11:00:00Z"))
    history.record_run(canned("failed", "2026-11-12T12:00:00Z", name="Cancel"))

    def ids(*args):
        result = runs("list", *args)
        assert result.exit_code == 0, result.output
        return [int(line.split()[0]) for line in result.stdout.splitlines()[1:]]

    assert ids() == [3, 2, 1]
    assert ids("--limit", 1) == [3]
    assert ids("--scenario", "booking") == [2, 1]
    assert ids("--status", "failed") == [3, 2]
    assert ids("--scenario", "Booking", "--status", "failed") == [2]


def test_runs_list_with_no_runs_says_where_it_looked():
    result = runs("list")
    assert result.exit_code == 0
    assert result.stdout.strip() == f"no runs recorded in {history.db_path()}"


def test_runs_list_refuses_bad_options():
    assert runs("list", "--status", "great").exit_code == 2
    assert runs("list", "--limit", 0).exit_code == 2


# voiceqa runs show


def test_runs_show_prints_the_summary_after_the_run_folder_is_gone(monkeypatch, scenario, tmp_path):
    fake_call(monkeypatch, "failed")
    console = run(scenario, tmp_path / "runs")
    assert console.exit_code == 1
    (row,) = history.list_runs()
    shutil.rmtree(tmp_path / "runs")

    result = runs("show", row.id)
    assert result.exit_code == 0, result.output
    assert result.stdout.rstrip("\n") == format_summary(history.load_result(row))
    # The same text the run printed at the end.
    assert result.stdout in console.stdout


def test_runs_show_of_an_unknown_id_exits_2():
    history.record_run(canned())
    result = runs("show", 42)
    assert result.exit_code == 2
    assert f"no run with id 42 in {history.db_path()}" in result.stderr


def test_runs_commands_report_a_newer_schema_and_exit_1():
    history.record_run(canned())
    sql("PRAGMA user_version = 9")
    for args in (("list",), ("show", 1)):
        result = runs(*args)
        assert result.exit_code == 1, result.output
        assert "schema version 9" in result.stderr


def test_runs_show_of_a_row_that_does_not_parse_exits_1():
    run_id = history.record_run(canned())
    sql("UPDATE runs SET result_json = '{}' WHERE id = ?", (run_id,))
    result = runs("show", run_id)
    assert result.exit_code == 1
    assert f"run {run_id}: stored result does not read back" in result.stderr


def test_runs_help_lists_the_commands():
    result = runner.invoke(app, ["runs", "--help"])
    assert result.exit_code == 0
    assert "list" in result.stdout and "show" in result.stdout
