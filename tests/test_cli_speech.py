import json
import re

import pytest
from typer.testing import CliRunner

from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.speech import model_cache
from voiceqa.speech.models import Transcript

runner = CliRunner()


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def make_cached_model(cache, name, size):
    snapshot = cache / f"models--Systran--faster-whisper-{name}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x" * size)
    return snapshot


@pytest.fixture
def models_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICEQA_MODELS_DIR", str(tmp_path))
    return tmp_path


def test_doctor_reports_the_stt_extra_installed(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(cli, "_faster_whisper_version", lambda: "9.9.9")
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "stt extra:     installed (faster-whisper 9.9.9)" in result.output


def test_doctor_reports_the_stt_extra_missing(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "stt extra:     not installed (pip install voiceqa[stt])" in result.output


def test_doctor_shows_models_dir_and_cached_models(models_env):
    make_cached_model(models_env, "tiny.en", 2 * 1024 * 1024)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert f"models dir:    {models_env} (VOICEQA_MODELS_DIR)" in result.output
    assert re.search(r"stt model:\s+tiny\.en\s+2\.1 MB", result.output)


def test_doctor_says_when_no_model_is_cached(models_env):
    result = runner.invoke(app, ["doctor"])
    assert "stt models:    none cached" in result.output


def test_models_list_shows_name_size_and_path(models_env):
    make_cached_model(models_env, "base.en", 3_000_000)
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "base.en" in result.output
    assert "3.0 MB" in result.output
    assert str(models_env / "models--Systran--faster-whisper-base.en") in result.output


def test_models_list_takes_models_dir(tmp_path, models_env):
    other = tmp_path / "other"
    make_cached_model(other, "small.en", 10)
    result = runner.invoke(app, ["models", "list", "--models-dir", str(other)])
    assert "small.en" in result.output
    assert f"{other} (--models-dir)" in result.output


def test_models_list_with_nothing_cached(models_env):
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "no faster-whisper models" in result.output


def test_models_pull_downloads_into_the_models_dir(monkeypatch, models_env):
    calls = []

    def fake_pull(model, models_dir):
        calls.append((model, models_dir))
        return make_cached_model(models_dir, model, 5)

    monkeypatch.setattr(cli, "pull_stt_model", fake_pull)
    result = runner.invoke(app, ["models", "pull", "stt", "base.en"])
    assert result.exit_code == 0, result.output
    assert calls == [("base.en", models_env)]
    assert "base.en" in result.output
    assert str(models_env) in result.output


def test_models_pull_says_when_already_cached(monkeypatch, models_env):
    make_cached_model(models_env, "tiny.en", 5)

    def no_pull(*args):
        raise AssertionError("must not download a cached model")

    monkeypatch.setattr(cli, "pull_stt_model", no_pull)
    result = runner.invoke(app, ["models", "pull", "stt", "tiny.en"])
    assert result.exit_code == 0
    assert "already cached" in result.output


def test_models_pull_without_the_extra_names_it(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    result = runner.invoke(app, ["models", "pull", "stt", "base.en"])
    assert result.exit_code == 1
    assert "voiceqa[stt]" in result.output


def test_models_pull_rejects_an_unknown_model(models_env):
    result = runner.invoke(app, ["models", "pull", "stt", "enormous.en"])
    assert result.exit_code == 2
    assert "base.en" in result.output


def test_models_pull_only_knows_stt(models_env):
    result = runner.invoke(app, ["models", "pull", "tts", "base.en"])
    assert result.exit_code == 2


class FakeEngine:
    def transcribe(self, samples, rate):
        return Transcript(engine="fake", model="none", language="en")


@pytest.fixture
def fake_bench_engine(monkeypatch):
    built = []

    def factory(model, models_dir):
        built.append((model, models_dir))
        return FakeEngine()

    monkeypatch.setattr(cli, "_bench_engine", factory)
    monkeypatch.setattr(cli, "find_model", lambda model, models_dir: models_dir / model)
    return built


def test_bench_stt_json(fake_bench_engine, models_env):
    result = runner.invoke(app, ["bench", "stt", "--stt-model", "tiny.en", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["model"] == "tiny.en"
    assert data["engine"] == "faster-whisper"
    assert data["utterances"] >= 3
    for key in ("rtf", "load_seconds", "audio_seconds", "utterance_median_ms"):
        assert key in data
    assert fake_bench_engine == [("tiny.en", models_env)]


def test_bench_stt_plain_summary(fake_bench_engine, models_env):
    result = runner.invoke(app, ["bench", "stt"])
    assert result.exit_code == 0, result.output
    out = plain(result.output)
    assert "base.en" in out
    for label in ("model load", "real-time factor", "utterance median", "p95", "max"):
        assert label in out


def test_bench_stt_downloads_a_missing_model_with_a_message(monkeypatch, models_env):
    pulled = []
    monkeypatch.setattr(cli, "_bench_engine", lambda model, models_dir: FakeEngine())
    monkeypatch.setattr(cli, "find_model", lambda model, models_dir: None)
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(cli, "pull_stt_model", lambda m, d: pulled.append((m, d)))

    result = runner.invoke(app, ["bench", "stt", "--stt-model", "tiny.en", "--json"])

    assert result.exit_code == 0, result.output
    assert pulled == [("tiny.en", models_env)]
    assert "downloading faster-whisper model tiny.en (78.1 MB)" in result.stderr
    assert str(models_env) in result.stderr
    json.loads(result.stdout)


def test_bench_stt_without_the_extra_names_it(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    result = runner.invoke(app, ["bench", "stt"])
    assert result.exit_code == 1
    assert "voiceqa[stt]" in result.output
