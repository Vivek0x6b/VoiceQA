import platform
import re
import shutil

from typer.testing import CliRunner

import voiceqa
from voiceqa.cli.main import app

runner = CliRunner()


def test_version_prints_package_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert voiceqa.__version__ in result.output


def test_doctor_prints_python_version():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert platform.python_version() in result.output


def test_doctor_reports_ffmpeg_found(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/ffmpeg")
    result = runner.invoke(app, ["doctor"])
    assert "ffmpeg: found (/usr/bin/ffmpeg)" in result.output


def test_doctor_reports_ffmpeg_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    result = runner.invoke(app, ["doctor"])
    assert "ffmpeg: not found" in result.output


def test_agent_refuses_a_missing_wav_before_listening(tmp_path):
    result = runner.invoke(app, ["agent", "--port", "0", "--play", str(tmp_path / "nope.wav")])
    assert result.exit_code != 0
    assert "nope.wav" in result.output


def test_agent_help_lists_its_options():
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    # Rich styles the help when it thinks colour is wanted (on CI, for one).
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--host", "--port", "--play", "--hangup-after"):
        assert option in plain


def test_call_help_lists_its_options():
    result = runner.invoke(app, ["call", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--server", "--user", "--play", "--record-dir", "--duration", "--codec"):
        assert option in plain


def test_call_without_server_or_host_fails_with_bad_options():
    result = runner.invoke(app, ["call", "agent"])
    assert result.exit_code == 2
    assert "--server" in result.output
