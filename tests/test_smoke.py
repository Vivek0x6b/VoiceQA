import platform
import re
import shutil
from pathlib import Path

import typer
from typer.testing import CliRunner

import voiceqa
from voiceqa.cli.main import app
from voiceqa.engine.calibrate import CalibrationResult

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
    for option in ("--host", "--port", "--play", "--hangup-after", "--answer-delay"):
        assert option in plain


def test_agent_with_a_negative_answer_delay_fails_with_bad_options():
    result = runner.invoke(app, ["agent", "--port", "0", "--answer-delay", "-1"])
    assert result.exit_code == 2


def test_call_help_lists_its_options():
    result = runner.invoke(app, ["call", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--server", "--user", "--play", "--record-dir", "--duration", "--codec"):
        assert option in plain


def test_call_help_shows_the_stt_extra_name():
    result = runner.invoke(app, ["call", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    # Once in the --transcribe help and once in the command description.
    assert plain.count("[stt]") == 2


def _commands(command, path=()):
    yield path, command
    for name, sub in getattr(command, "commands", {}).items():
        yield from _commands(sub, (*path, name))


def test_every_bracketed_word_in_help_text_survives_rich_markup():
    """Rich reads "[stt]" as a style tag and drops it unless it is escaped."""
    root = typer.main.get_command(app)
    checked = 0
    for path, command in _commands(root):
        texts = [command.help or ""] + [getattr(p, "help", None) or "" for p in command.params]
        words = {w for text in texts for w in re.findall(r"\[\w+\]", text)}
        if not words:
            continue
        result = runner.invoke(app, [*path, "--help"])
        plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        for word in words:
            assert word in plain, f"{' '.join(path)} --help lost {word}"
            checked += 1
    assert checked > 0


def test_call_without_server_or_host_fails_with_bad_options():
    result = runner.invoke(app, ["call", "agent"])
    assert result.exit_code == 2
    assert "--server" in result.output


def test_call_with_a_non_numeric_server_port_fails_with_bad_options():
    result = runner.invoke(app, ["call", "agent", "--server", "127.0.0.1:abc"])
    assert result.exit_code == 2
    assert "abc" in result.output


def test_call_with_a_non_positive_duration_fails_with_bad_options():
    result = runner.invoke(app, ["call", "agent", "--server", "127.0.0.1", "--duration", "0"])
    assert result.exit_code == 2


def test_call_with_an_at_sign_in_a_bare_target_fails_with_bad_options():
    result = runner.invoke(app, ["call", "agent@127.0.0.1", "--server", "127.0.0.1"])
    assert result.exit_code == 2
    assert "@" in result.output


def test_call_interrupted_prints_the_recording_folder(tmp_path, monkeypatch):
    import voiceqa.cli.main as main

    def interrupted(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(main, "run_call", interrupted)
    result = runner.invoke(app, ["call", "sip:agent@127.0.0.1:5070", "--record-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "call interrupted" in result.output
    assert str(tmp_path) in result.output


def test_calibrate_with_a_non_positive_seconds_fails_with_bad_options():
    result = runner.invoke(app, ["calibrate", "--seconds", "-1"])
    assert result.exit_code == 2


def test_calibrate_help_lists_its_options():
    result = runner.invoke(app, ["calibrate", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--seconds", "--record-dir"):
        assert option in plain


def _fake_calibration(pulses_sent, pulses_matched, answered=True, status=200, reason="OK"):
    async def run(record_dir, seconds=10.0):
        return CalibrationResult(
            created_at="2026-01-01T00:00:00.000Z",
            samples=pulses_matched,
            median_ms=20.0 if pulses_matched else None,
            p95_ms=20.0 if pulses_matched else None,
            min_ms=20.0 if pulses_matched else None,
            max_ms=20.0 if pulses_matched else None,
            pulses_sent=pulses_sent,
            pulses_matched=pulses_matched,
            answered=answered,
            status=status,
            reason=reason,
            record_dir=Path(record_dir),
        )

    return run


def test_calibrate_exits_1_when_fewer_than_half_the_pulses_matched(monkeypatch):
    import voiceqa.cli.main as main

    monkeypatch.setattr(main, "run_calibration", _fake_calibration(pulses_sent=4, pulses_matched=1))
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 1


def test_calibrate_exits_0_when_half_or_more_pulses_matched(monkeypatch):
    import voiceqa.cli.main as main

    monkeypatch.setattr(main, "run_calibration", _fake_calibration(pulses_sent=4, pulses_matched=2))
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 0


def test_calibrate_prints_the_reason_when_the_call_was_not_answered(monkeypatch):
    import voiceqa.cli.main as main

    # status and reason in the shape run_call actually returns them (a
    # numeric status and a short reason text, not one combined string).
    fake = _fake_calibration(
        pulses_sent=0, pulses_matched=0, answered=False, status=486, reason="Busy Here"
    )
    monkeypatch.setattr(main, "run_calibration", fake)
    result = runner.invoke(app, ["calibrate"])
    assert result.exit_code == 1
    assert "486 Busy Here" in result.output
