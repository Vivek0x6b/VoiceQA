"""Run history: data dir per platform, the runs table, list order and
filters, the schema version and concurrent writers. Every test has its own
VOICEQA_DATA_DIR (tests/conftest.py)."""

import gc
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from contextlib import closing
from pathlib import Path

import pytest

from voiceqa.scenario import history
from voiceqa.scenario.result import CallInfo, ScenarioInfo, ScenarioResult, StepResult


def make_result(
    name="booking", status="passed", started="2026-11-12T10:00:00Z", steps=(), run_dir="runs/x"
):
    return ScenarioResult(
        scenario=ScenarioInfo(name=name, path=f"/s/{name}.yaml", sha256="ab" * 32),
        started=started,
        duration=4.5,
        status=status,
        run_dir=run_dir,
        call=CallInfo(answered=True, status=200, reason="OK", duration=4.0),
        steps=[
            StepResult(index=i + 1, label=f"s{i + 1}", status=s, attempts=1, duration=1.0)
            for i, s in enumerate(steps)
        ],
    )


# Data dir


@pytest.fixture
def no_data_env(monkeypatch, tmp_path):
    monkeypatch.delenv("VOICEQA_DATA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    return tmp_path


def test_data_dir_env_wins_on_every_platform(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICEQA_DATA_DIR", str(tmp_path / "mine"))
    for platform in ("win32", "darwin", "linux"):
        monkeypatch.setattr(history.sys, "platform", platform)
        assert history.data_dir() == tmp_path / "mine"
    assert history.db_path() == tmp_path / "mine" / "runs.sqlite3"


def test_data_dir_on_windows_is_under_localappdata(monkeypatch, no_data_env):
    monkeypatch.setattr(history.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(no_data_env / "local"))
    assert history.data_dir() == no_data_env / "local" / "voiceqa"
    monkeypatch.delenv("LOCALAPPDATA")
    assert history.data_dir() == no_data_env / "home" / "AppData" / "Local" / "voiceqa"


def test_data_dir_on_macos_is_application_support(monkeypatch, no_data_env):
    monkeypatch.setattr(history.sys, "platform", "darwin")
    monkeypatch.setenv("XDG_DATA_HOME", str(no_data_env / "xdg"))
    assert history.data_dir() == no_data_env / "home" / "Library" / "Application Support" / (
        "voiceqa"
    )


def test_data_dir_elsewhere_is_xdg_data_home_or_local_share(monkeypatch, no_data_env):
    monkeypatch.setattr(history.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(no_data_env / "xdg"))
    assert history.data_dir() == no_data_env / "xdg" / "voiceqa"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert history.data_dir() == no_data_env / "home" / ".local" / "share" / "voiceqa"


# The runs table


def test_a_recorded_run_reads_back_with_every_column(tmp_path):
    result = make_result(steps=("passed", "failed"), status="failed")
    run_id = history.record_run(result)

    row = history.get_run(run_id)
    assert row.id == run_id
    assert (row.scenario_name, row.scenario_path, row.scenario_hash) == (
        "booking",
        "/s/booking.yaml",
        "ab" * 32,
    )
    assert (row.started, row.duration, row.status) == ("2026-11-12T10:00:00Z", 4.5, "failed")
    assert (row.exit_code, row.steps_passed, row.steps_total) == (1, 1, 2)
    assert row.run_dir == str(Path("runs/x").resolve())
    assert history.load_result(row) == result
    assert history.db_path().is_file()


def test_record_run_stores_the_exit_code_it_is_given():
    # A passed run whose JUnit file could not be written exits 3.
    run_id = history.record_run(make_result(), exit_code=3)
    assert history.get_run(run_id).exit_code == 3


def test_a_result_without_a_run_folder_has_no_run_dir():
    run_id = history.record_run(make_result(run_dir=None))
    assert history.get_run(run_id).run_dir is None


def test_get_run_of_an_unknown_id_is_none():
    history.record_run(make_result())
    assert history.get_run(999) is None


def test_list_is_newest_first_by_start_time_then_by_id():
    ids = {}
    for key, started in (("a", "10:00"), ("b", "12:00"), ("c", "11:00"), ("d", "12:00")):
        ids[key] = history.record_run(make_result(name=key, started=f"2026-11-12T{started}:00Z"))
    rows = history.list_runs()
    assert [r.scenario_name for r in rows] == ["d", "b", "c", "a"]
    assert [r.id for r in history.list_runs(limit=2)] == [ids["d"], ids["b"]]


def test_list_filters_by_scenario_name_and_status():
    history.record_run(make_result(name="Booking", status="passed"))
    history.record_run(make_result(name="Booking", status="failed", started="2026-11-12T11:00:00Z"))
    history.record_run(make_result(name="cancel", status="failed"))
    history.record_run(make_result(name="cancel", status="error"))

    assert [r.status for r in history.list_runs(scenario="booking")] == ["failed", "passed"]
    assert {r.scenario_name for r in history.list_runs(status="failed")} == {"Booking", "cancel"}
    assert len(history.list_runs(status="failed")) == 2
    (row,) = history.list_runs(scenario="cancel", status="error")
    assert (row.scenario_name, row.status) == ("cancel", "error")
    assert history.list_runs(scenario="nope") == []


def test_list_without_a_db_file_is_empty_and_creates_nothing():
    assert history.list_runs() == []
    assert history.get_run(1) is None
    assert not history.db_path().exists()


def test_history_never_touches_the_gc(monkeypatch):
    def refuse(*args):
        raise AssertionError("history touched the GC")

    for name in ("freeze", "unfreeze", "disable", "enable"):
        monkeypatch.setattr(gc, name, refuse)
    run_id = history.record_run(make_result())
    history.list_runs()
    history.get_run(run_id)


# Schema version


def test_a_new_db_has_schema_version_1():
    history.record_run(make_result())
    assert query("PRAGMA user_version")[0][0] == history.SCHEMA_VERSION == 1


def query(statement):
    with closing(sqlite3.connect(history.db_path())) as conn:
        rows = conn.execute(statement).fetchall()
        conn.commit()
    return rows


def _set_version(version):
    query(f"PRAGMA user_version = {version}")


def test_a_newer_schema_version_is_an_error_that_names_it():
    history.record_run(make_result())
    _set_version(7)
    for call in (
        lambda: history.record_run(make_result()),
        lambda: history.list_runs(),
        lambda: history.get_run(1),
    ):
        with pytest.raises(history.HistoryError, match=r"schema version 7.*knows up to 1"):
            call()


def test_an_older_schema_is_migrated_step_by_step(monkeypatch):
    first = history.record_run(make_result())

    def to_v2(conn):
        conn.execute("ALTER TABLE runs ADD COLUMN note TEXT")

    monkeypatch.setattr(history, "SCHEMA_VERSION", 2)
    monkeypatch.setitem(history.MIGRATIONS, 2, to_v2)
    second = history.record_run(make_result())

    assert query("PRAGMA user_version")[0][0] == 2
    assert "note" in [r[1] for r in query("PRAGMA table_info(runs)")]
    assert [r.id for r in history.list_runs()] == [second, first]


# Concurrency


def test_connections_wait_for_a_lock_instead_of_failing():
    history.record_run(make_result())
    conn = history.connect(history.db_path())
    try:
        busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert busy == history.BUSY_TIMEOUT_MS
    assert busy >= 2000


@pytest.mark.realtime
def test_a_write_waits_for_another_writer_to_finish():
    history.record_run(make_result())
    holder = sqlite3.connect(history.db_path(), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO runs (scenario_name, scenario_path, scenario_hash, started, duration,"
        " status, exit_code, steps_passed, steps_total, run_dir, result_json)"
        " VALUES ('holder', '', '', '2026-11-12T09:00:00Z', 0, 'passed', 0, 0, 0, NULL, '{}')"
    )
    done = {}

    def write():
        try:
            done["id"] = history.record_run(make_result(name="waiter"))
        except Exception as exc:
            done["error"] = exc

    thread = threading.Thread(target=write)
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive(), done
    holder.execute("COMMIT")
    holder.close()
    thread.join(10)

    assert "error" not in done, done
    names = [r.scenario_name for r in sorted(history.list_runs(), key=lambda r: r.id)]
    # The waiter's row went in after the holder's, so it really waited.
    assert names == ["booking", "holder", "waiter"]


_WRITER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from voiceqa.scenario import history
    from voiceqa.scenario.result import CallInfo, ScenarioInfo, ScenarioResult

    name, go, count = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
    result = ScenarioResult(
        scenario=ScenarioInfo(name=name, path="/s.yaml", sha256="00"),
        started="2026-11-12T10:00:00Z", duration=1.0, status="passed",
        call=CallInfo(answered=True),
    )
    while not go.exists():
        time.sleep(0.01)
    for _ in range(count):
        history.record_run(result)
    """
)


@pytest.mark.realtime
def test_two_processes_writing_at_once_both_land(tmp_path):
    go = tmp_path / "go"
    count = 25
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _WRITER, name, str(go), str(count)], stderr=subprocess.PIPE
        )
        for name in ("one", "two")
    ]
    try:
        go.write_text("", encoding="utf-8")
        errors = [p.communicate(timeout=60)[1] for p in procs]
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()
    assert [p.returncode for p in procs] == [0, 0], errors

    rows = history.list_runs(limit=1000)
    assert sum(r.scenario_name == "one" for r in rows) == count
    assert sum(r.scenario_name == "two" for r in rows) == count
