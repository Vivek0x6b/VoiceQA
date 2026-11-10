import re
import sys

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import read_wav, write_wav
from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.engine.call import CallResult
from voiceqa.speech.tts_process import TtsError

runner = CliRunner()


@pytest.fixture(autouse=True)
def cache_env(monkeypatch, tmp_path):
    """Never touch the real per-user cache from these tests."""
    monkeypatch.setenv("VOICEQA_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "tts"


@pytest.fixture
def prompts(tmp_path):
    folder = tmp_path / "prompts"
    folder.mkdir()
    write_wav(folder / "hi.wav", generate_tone(500, 0.5, sample_rate=16000), 16000)
    index = folder / "index.yaml"
    index.write_text(yaml.safe_dump({"Hi there.": "hi.wav"}), encoding="utf-8")
    return index


@pytest.fixture
def fake_call(monkeypatch, tmp_path):
    state = {"calls": [], "answered": True}

    async def run_call(**kwargs):
        state["calls"].append(kwargs)
        record_dir = tmp_path / "run"
        record_dir.mkdir(exist_ok=True)
        return CallResult(
            answered=state["answered"],
            status=200,
            reason="OK",
            duration=1.0,
            frames_sent=50,
            frames_received=50,
            first_audio_latency=None,
            record_dir=record_dir,
        )

    monkeypatch.setattr(cli, "run_call", run_call)
    return state


def call_args(tmp_path, *extra):
    return ["call", "sip:agent@127.0.0.1:5070", "--record-dir", str(tmp_path / "run"), *extra]


class FailingTts:
    name = "broken"

    def voices(self):
        return []

    def synthesize(self, text, voice=None):
        raise TtsError("broken engine: it fell over")


# voiceqa say


def test_say_writes_the_engine_output_to_a_wav(tmp_path):
    out = tmp_path / "out.wav"
    result = runner.invoke(app, ["say", "hello world", "-o", str(out), "--tts", "fake"])

    assert result.exit_code == 0, result.output
    samples, rate = read_wav(out)
    assert rate == 8000
    # FakeTts makes 0.05 s per character.
    assert len(samples) == round(0.05 * len("hello world") * 8000)
    assert str(out) in result.output


def test_say_with_wav_prompts(tmp_path, prompts):
    out = tmp_path / "out.wav"
    result = runner.invoke(
        app, ["say", "Hi there.", "-o", str(out), "--tts", "wav", "--prompts", str(prompts)]
    )
    assert result.exit_code == 0, result.output
    samples, rate = read_wav(out)
    assert (rate, len(samples)) == (16000, 8000)


def test_say_wav_without_a_prompt_for_the_text(tmp_path, prompts):
    result = runner.invoke(
        app,
        ["say", "Bye.", "-o", str(tmp_path / "o.wav"), "--tts", "wav", "--prompts", str(prompts)],
    )
    assert result.exit_code == 1
    assert "no WAV prompt for text 'Bye.'" in result.stderr
    assert not (tmp_path / "o.wav").exists()


def test_say_wav_without_an_index_asks_for_one(tmp_path):
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "wav"])
    assert result.exit_code == 2
    assert "--prompts" in result.stderr


def test_say_default_engine_off_windows_and_macos_explains_the_choice(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "default_tts_name", lambda: "wav")
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav")])
    assert result.exit_code == 2
    message = " ".join(result.stderr.split())
    assert "default TTS engine here is wav prompts" in message
    assert "--prompts" in message


def test_say_uses_the_os_default_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "default_tts_name", lambda: "fake")
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav")])
    assert result.exit_code == 0, result.output
    assert re.search(r"engine\s+fake", result.output)


def test_say_unknown_engine_is_a_bad_option(tmp_path):
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "nope"])
    assert result.exit_code == 2
    assert "unknown TTS engine 'nope'" in result.stderr


def test_say_os_engine_on_the_wrong_os_is_a_bad_option(tmp_path):
    other = "say" if sys.platform == "win32" else "sapi"
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", other])
    assert result.exit_code == 2
    assert "needs" in result.stderr


def test_say_prompts_with_another_engine_is_a_bad_option(tmp_path, prompts):
    result = runner.invoke(
        app,
        ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "fake", "--prompts", str(prompts)],
    )
    assert result.exit_code == 2
    assert "--prompts" in result.stderr


def test_say_engine_failure_exits_1_with_the_reason(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "create_tts", lambda name, **kw: FailingTts())
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "x"])
    assert result.exit_code == 1
    assert "it fell over" in result.stderr


def test_say_empty_text_is_a_bad_option(tmp_path):
    result = runner.invoke(app, ["say", "  ", "-o", str(tmp_path / "o.wav"), "--tts", "fake"])
    assert result.exit_code == 2


def test_say_caches_engine_output(tmp_path, monkeypatch, cache_env):
    calls = []

    class Counting:
        name = "counting"

        def voices(self):
            return []

        def synthesize(self, text, voice=None):
            calls.append(text)
            return cli.Speech(samples=generate_tone(300, 0.2), rate=8000)

    monkeypatch.setattr(cli, "create_tts", lambda name, **kw: Counting())
    for _ in range(2):
        result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "x"])
        assert result.exit_code == 0, result.output
    assert calls == ["Hi."]
    assert len(list(cache_env.glob("*.wav"))) == 1


def test_say_does_not_cache_the_fake_engine(tmp_path, cache_env):
    result = runner.invoke(app, ["say", "Hi.", "-o", str(tmp_path / "o.wav"), "--tts", "fake"])
    assert result.exit_code == 0
    assert not cache_env.exists()


def test_say_list_voices(monkeypatch):
    class Voices:
        name = "v"

        def voices(self):
            return ["Anna", "Bob Smith"]

    monkeypatch.setattr(cli, "create_tts", lambda name, **kw: Voices())
    result = runner.invoke(app, ["say", "--list-voices", "--tts", "x"])
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == ["Anna", "Bob Smith"]


def test_say_needs_text_and_out_without_list_voices():
    result = runner.invoke(app, ["say", "--tts", "fake"])
    assert result.exit_code == 2


# voiceqa call --say


def test_call_say_sends_the_speech_at_8_khz(tmp_path, fake_call, prompts):
    result = runner.invoke(
        app, call_args(tmp_path, "--say", "Hi there.", "--tts", "wav", "--prompts", str(prompts))
    )
    assert result.exit_code == 0, result.output
    (kwargs,) = fake_call["calls"]
    audio = kwargs["audio"]
    assert audio.dtype == np.int16
    # The 0.5 s, 16 kHz prompt arrives resampled to 8 kHz.
    assert len(audio) == 4000
    assert re.search(r"tts\s+wav", result.output)


def test_call_say_with_play_is_a_bad_option(tmp_path, fake_call, prompts):
    wav = tmp_path / "x.wav"
    write_wav(wav, generate_tone(300, 0.1), 8000)
    result = runner.invoke(
        app, call_args(tmp_path, "--say", "Hi there.", "--tts", "fake", "--play", str(wav))
    )
    assert result.exit_code == 2
    assert "--say" in result.stderr and "--play" in result.stderr
    assert fake_call["calls"] == []


def test_call_tts_options_need_say(tmp_path, fake_call):
    result = runner.invoke(app, call_args(tmp_path, "--tts", "fake"))
    assert result.exit_code == 2
    assert "--say" in result.stderr
    assert fake_call["calls"] == []


def test_call_say_failure_never_dials(tmp_path, fake_call, monkeypatch):
    monkeypatch.setattr(cli, "create_tts", lambda name, **kw: FailingTts())
    result = runner.invoke(app, call_args(tmp_path, "--say", "Hello.", "--tts", "x"))
    assert result.exit_code == 1
    assert "it fell over" in result.stderr
    assert fake_call["calls"] == []


def test_call_say_missing_prompt_never_dials(tmp_path, fake_call, prompts):
    result = runner.invoke(
        app, call_args(tmp_path, "--say", "Other.", "--tts", "wav", "--prompts", str(prompts))
    )
    assert result.exit_code == 1
    assert "no WAV prompt" in result.stderr
    assert fake_call["calls"] == []


def test_call_say_default_wav_engine_suggests_play(tmp_path, fake_call, monkeypatch):
    monkeypatch.setattr(cli, "default_tts_name", lambda: "wav")
    result = runner.invoke(app, call_args(tmp_path, "--say", "Hello."))
    assert result.exit_code == 2
    message = " ".join(result.stderr.split())
    assert "--prompts" in message and "--play" in message
    assert fake_call["calls"] == []


def test_call_without_say_still_sends_the_tone(tmp_path, fake_call):
    result = runner.invoke(app, call_args(tmp_path))
    assert result.exit_code == 0, result.output
    (kwargs,) = fake_call["calls"]
    assert len(kwargs["audio"]) == 8000
