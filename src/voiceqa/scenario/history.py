"""Local history of scenario runs, in SQLite (stdlib sqlite3).

The database is runs.sqlite3 in data_dir(): VOICEQA_DATA_DIR if set, else
%LOCALAPPDATA%/voiceqa on Windows, ~/Library/Application Support/voiceqa
on macOS, and $XDG_DATA_HOME/voiceqa or ~/.local/share/voiceqa elsewhere.

Each row keeps the whole result.json text, so a run can be shown again
after its run folder is gone. The schema version is in PRAGMA
user_version. A file with a newer version is refused; an older one is
brought up to date by MIGRATIONS, one version at a time.

Writers use SQLite's default locking. A connection waits up to
BUSY_TIMEOUT_MS for another writer instead of failing, so two runs that
finish at once both land.
"""

import os
import sqlite3
import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from voiceqa.scenario.result import ScenarioResult

DATA_DIR_ENV = "VOICEQA_DATA_DIR"
DB_FILE = "runs.sqlite3"
SCHEMA_VERSION = 1
BUSY_TIMEOUT_MS = 5000

_COLUMNS = (
    "id",
    "scenario_name",
    "scenario_path",
    "scenario_hash",
    "started",
    "duration",
    "status",
    "exit_code",
    "steps_passed",
    "steps_total",
    "run_dir",
    "result_json",
)


class HistoryError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunRow:
    id: int
    scenario_name: str
    scenario_path: str
    scenario_hash: str
    started: str
    duration: float
    status: str
    exit_code: int
    steps_passed: int
    steps_total: int
    run_dir: str | None
    result_json: str


def data_dir() -> Path:
    if os.environ.get(DATA_DIR_ENV):
        return Path(os.environ[DATA_DIR_ENV]).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "voiceqa"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "voiceqa"
    xdg = os.environ.get("XDG_DATA_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share") / "voiceqa"


def db_path() -> Path:
    return data_dir() / DB_FILE


def _create_v1(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_name TEXT NOT NULL,
            scenario_path TEXT NOT NULL,
            scenario_hash TEXT NOT NULL,
            started TEXT NOT NULL,
            duration REAL NOT NULL,
            status TEXT NOT NULL,
            exit_code INTEGER NOT NULL,
            steps_passed INTEGER NOT NULL,
            steps_total INTEGER NOT NULL,
            run_dir TEXT,
            result_json TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX runs_started ON runs (started, id)")


# MIGRATIONS[n] takes a database at version n - 1 to version n. Version 0
# is an empty file.
MIGRATIONS: dict[int, Callable[[sqlite3.Connection], None]] = {1: _create_v1}


def connect(path: Path) -> sqlite3.Connection:
    """A connection in autocommit mode (transactions are explicit) that
    waits BUSY_TIMEOUT_MS for a lock."""
    return sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)


def _version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _upgrade(conn: sqlite3.Connection, path: Path) -> None:
    if _version(conn) == SCHEMA_VERSION:
        return
    # Read the version again under the write lock: another process may
    # have created or migrated the file in the meantime.
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = _version(conn)
        if version > SCHEMA_VERSION:
            raise HistoryError(
                f"{path} has schema version {version}, this voiceqa knows up to "
                f"{SCHEMA_VERSION}; use a newer voiceqa or another VOICEQA_DATA_DIR"
            )
        for target in range(version + 1, SCHEMA_VERSION + 1):
            MIGRATIONS[target](conn)
            conn.execute(f"PRAGMA user_version = {target}")
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _open(create: bool) -> sqlite3.Connection | None:
    path = db_path()
    if not create and not path.is_file():
        return None
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(path)
    try:
        _upgrade(conn, path)
    except BaseException:
        conn.close()
        raise
    return conn


def record_run(result: ScenarioResult, exit_code: int | None = None) -> int:
    """Add a row for `result` and return its id. `exit_code` is what the
    command exited with, if it differs from the result's own code."""
    run_dir = str(Path(result.run_dir).resolve()) if result.run_dir else None
    values = (
        result.scenario.name,
        result.scenario.path,
        result.scenario.sha256,
        result.started,
        result.duration,
        result.status,
        result.exit_code if exit_code is None else exit_code,
        result.steps_passed,
        result.steps_total,
        run_dir,
        result.model_dump_json(indent=2),
    )
    with closing(_open(create=True)) as conn:
        cursor = conn.execute(
            f"INSERT INTO runs ({', '.join(_COLUMNS[1:])}) VALUES ({', '.join('?' * len(values))})",
            values,
        )
        return cursor.lastrowid


def list_runs(
    limit: int = 20, scenario: str | None = None, status: str | None = None
) -> list[RunRow]:
    """Newest first. `scenario` matches the scenario name, ignoring case."""
    where, params = [], []
    if scenario is not None:
        where.append("scenario_name = ? COLLATE NOCASE")
        params.append(scenario)
    if status is not None:
        where.append("status = ?")
        params.append(status)
    sql = f"SELECT {', '.join(_COLUMNS)} FROM runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY started DESC, id DESC LIMIT ?"
    conn = _open(create=False)
    if conn is None:
        return []
    with closing(conn):
        return [RunRow(*row) for row in conn.execute(sql, (*params, limit))]


def get_run(run_id: int) -> RunRow | None:
    conn = _open(create=False)
    if conn is None:
        return None
    with closing(conn):
        row = conn.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM runs WHERE id = ?", (run_id,)
        ).fetchone()
    return RunRow(*row) if row else None


def load_result(row: RunRow) -> ScenarioResult:
    return ScenarioResult.model_validate_json(row.result_json)
