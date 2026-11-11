"""`voiceqa agent --script`: options, loading and synthesis before listening.
The server itself is replaced, so nothing listens here; the calls are
tested in tests/telephony/test_scripted_agent.py."""

import re

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from voiceqa.audio.dsp import generate_tone, resample
from voiceqa.audio.wav import write_wav
from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.speech import model_cache
from voiceqa.telephony.media import SAMPLE_RATE

runner = CliRunner()


@pytest.fixture(autouse=True)
def cache_env(monkeypatch, tmp_path):
    """Never touch the real per-user cache from these tests."""
    monkeypatch.setenv("VOICEQA_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def served(monkeypatch):
    configs = []

    async def serve(config):
        configs.append(config)

    monkeypatch.setattr(cli, "_serve_agent", serve)
    return configs


def write_script(folder, data):
    path = folder / "replies.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def prompts(tmp_path):
    folder = tmp_path / "prompts"
    folder.mkdir()
    write_wav(folder / "hi.wav", generate_tone(500, 0.5, sample_rate=16000), 16000)
    index = folder / "index.yaml"
    index.write_text(yaml.safe_dump({"Hi there.": "hi.wav"}), encoding="utf-8")
    return index


def agent_args(script, *extra):
    return ["agent", "--port", "0", "--script", str(script), *extra]


def test_say_lines_are_synthesized_with_wav_prompts_before_listening(tmp_path, served, prompts):
    write_wav(tmp_path / "bye.wav", generate_tone(700, 0.3), SAMPLE_RATE)
    script = write_script(
        tmp_path, {"greeting": {"say": "Hi there."}, "replies": [{"wav": "bye.wav"}]}
    )

    result = runner.invoke(app, agent_args(script, "--tts", "wav", "--prompts", str(prompts)))

    assert result.exit_code == 0, result.output
    (config,) = served
    expected = resample(generate_tone(500, 0.5, sample_rate=16000), 16000, SAMPLE_RATE)
    np.testing.assert_array_equal(config.script.greeting.audio, expected)
    assert len(config.script.replies[0].audio) == 2400
    assert (config.vad, config.end_silence_ms, config.play) == ("energy", 700, None)


def test_vad_and_end_silence_reach_the_agent(tmp_path, served):
    write_wav(tmp_path / "a.wav", generate_tone(700, 0.3), SAMPLE_RATE)
    script = write_script(tmp_path, {"replies": [{"wav": "a.wav"}]})
    result = runner.invoke(app, agent_args(script, "--vad", "auto", "--end-silence-ms", "400"))
    assert result.exit_code == 0, result.output
    assert (served[0].vad, served[0].end_silence_ms) == ("auto", 400)


def test_a_script_without_say_lines_needs_no_tts(tmp_path, served, monkeypatch):
    # The default engine off Windows and macOS is wav prompts, which would
    # need --prompts; a script of WAV lines must not ask for it.
    monkeypatch.setattr(cli, "default_tts_name", lambda: "wav")
    write_wav(tmp_path / "a.wav", generate_tone(700, 0.3), SAMPLE_RATE)
    script = write_script(tmp_path, {"replies": [{"wav": "a.wav"}]})
    result = runner.invoke(app, agent_args(script))
    assert result.exit_code == 0, result.output
    assert len(served) == 1


def test_a_tts_failure_exits_1_naming_the_line(tmp_path, served, prompts):
    script = write_script(tmp_path, {"replies": [{"say": "Hi there."}, {"say": "Unknown."}]})
    result = runner.invoke(app, agent_args(script, "--tts", "wav", "--prompts", str(prompts)))
    assert result.exit_code == 1
    assert "reply 2" in result.output and "Unknown." in result.output
    assert served == []


def test_a_load_error_exits_1_before_listening(tmp_path, served):
    script = write_script(tmp_path, {"replies": [{"wav": "a.wav", "say": "Hi."}]})
    result = runner.invoke(app, agent_args(script))
    assert result.exit_code == 1
    assert "reply 1: has both wav and say" in result.output
    assert served == []


def test_script_with_play_exits_1(tmp_path, served):
    write_wav(tmp_path / "a.wav", generate_tone(700, 0.3), SAMPLE_RATE)
    script = write_script(tmp_path, {"replies": [{"wav": "a.wav"}]})
    result = runner.invoke(app, agent_args(script, "--play", str(tmp_path / "a.wav")))
    assert result.exit_code == 1
    assert "--script" in result.output and "--play" in result.output
    assert served == []


@pytest.mark.parametrize(
    "option",
    [
        ["--tts", "fake"],
        ["--voice", "x"],
        ["--prompts", "index.yaml"],
        ["--vad", "energy"],
        ["--end-silence-ms", "500"],
    ],
)
def test_script_options_without_a_script_are_bad_options(served, option):
    result = runner.invoke(app, ["agent", "--port", "0", *option])
    assert result.exit_code == 2
    assert "--script" in result.output
    assert served == []


def test_end_silence_must_be_positive(tmp_path, served):
    script = write_script(tmp_path, {"replies": [{"say": "Hi."}]})
    result = runner.invoke(app, agent_args(script, "--end-silence-ms", "0", "--tts", "fake"))
    assert result.exit_code == 2


def test_silero_without_the_extra_exits_1_before_listening(tmp_path, monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    write_wav(tmp_path / "a.wav", generate_tone(700, 0.3), SAMPLE_RATE)
    script = write_script(tmp_path, {"replies": [{"wav": "a.wav"}]})
    # The real server: it fails while starting, before it listens.
    result = runner.invoke(app, agent_args(script, "--vad", "silero"))
    assert result.exit_code == 1
    assert "[stt]" in result.output


def test_agent_help_lists_the_script_options():
    result = runner.invoke(app, ["agent", "--help"])
    assert result.exit_code == 0
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ("--script", "--tts", "--voice", "--prompts", "--vad", "--end-silence-ms"):
        assert option in plain
