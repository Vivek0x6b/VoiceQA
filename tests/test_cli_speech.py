import json
import re
import wave

import numpy as np
import pytest
from typer.testing import CliRunner

from voiceqa.audio.wav import write_wav
from voiceqa.cli import main as cli
from voiceqa.cli.main import app
from voiceqa.engine.call import CallResult
from voiceqa.speech import model_cache
from voiceqa.speech.models import Segment, Transcript
from voiceqa.speech.worker import SttWorker

runner = CliRunner()


def plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def make_cached_model(cache, name, size):
    snapshot = cache / f"models--Systran--faster-whisper-{name}" / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "model.bin").write_bytes(b"x" * size)
    return snapshot


@pytest.fixture(autouse=True)
def hide_stt_extra(monkeypatch):
    """Every test here runs as if faster-whisper were not installed, the way
    a contributor with only the [dev] extra runs them. Tests that need the
    extra say so by patching stt_extra_installed."""
    real_find_spec = model_cache.importlib.util.find_spec

    def find_spec(name, *args, **kwargs):
        if name == "faster_whisper" or name.startswith("faster_whisper."):
            return None
        return real_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(model_cache.importlib.util, "find_spec", find_spec)


@pytest.fixture
def stt_extra(monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)


def test_the_stt_extra_is_hidden_in_this_module():
    assert model_cache.stt_extra_installed() is False


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


def test_doctor_survives_a_faster_whisper_without_package_metadata(monkeypatch, models_env):
    from importlib import metadata

    def no_metadata(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(metadata, "version", no_metadata)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "stt extra:     installed (faster-whisper, version unknown" in result.output


def test_doctor_reports_the_stt_extra_missing(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert f"stt extra:     not installed ({model_cache.STT_EXTRA_HINT})" in result.output


def test_doctor_shows_silero_vad_with_the_stt_extra(monkeypatch, models_env):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(cli, "_faster_whisper_version", lambda: "9.9.9")
    result = runner.invoke(app, ["doctor"])
    assert "vad:           silero and energy (auto uses silero)" in result.output


def test_doctor_shows_energy_vad_without_the_stt_extra(models_env):
    result = runner.invoke(app, ["doctor"])
    assert "vad:           energy only (silero needs the [stt] extra)" in result.output


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


def test_models_pull_downloads_into_the_models_dir(monkeypatch, models_env, stt_extra):
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
    assert model_cache.STT_EXTRA_HINT in result.output


def test_models_pull_rejects_an_unknown_model(models_env):
    result = runner.invoke(app, ["models", "pull", "stt", "enormous.en"])
    assert result.exit_code == 2
    assert "base.en" in result.output


def test_models_pull_rejects_a_local_folder(tmp_path, models_env):
    result = runner.invoke(app, ["models", "pull", "stt", str(tmp_path / "my-model")])
    assert result.exit_code == 2
    assert "not a local folder" in plain(result.output)


def test_models_pull_only_knows_stt(models_env):
    result = runner.invoke(app, ["models", "pull", "tts", "base.en"])
    assert result.exit_code == 2


class FakeEngine:
    def transcribe(self, samples, rate):
        return Transcript(engine="fake", model="none", language="en")


@pytest.fixture
def fake_bench_engine(monkeypatch, stt_extra):
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
    assert model_cache.STT_EXTRA_HINT in result.output


SCRIPTED = Transcript(
    segments=[Segment(start=0.25, end=0.75, text="hello there")],
    engine="fake",
    model="scripted",
    language="en",
)


def make_run_dir(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("caller.wav", "agent.wav"):
        write_wav(run_dir / name, np.zeros(8000, dtype=np.int16), 8000)
    return run_dir


@pytest.fixture
def fake_worker(monkeypatch):
    """Replaces the faster-whisper worker with FakeStt in a real worker
    process, and records which model and models dir were asked for."""
    asked = []

    def factory(model, models_dir):
        asked.append((model, models_dir))
        return SttWorker("fake", transcript=SCRIPTED.model_dump())

    monkeypatch.setattr(cli, "_stt_worker", factory)
    return asked


def test_transcribe_writes_transcript_json(tmp_path, fake_worker, models_env):
    run_dir = make_run_dir(tmp_path / "run")

    result = runner.invoke(app, ["transcribe", str(run_dir), "--stt-model", "tiny.en"])

    assert result.exit_code == 0, result.output
    path = run_dir / "transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [(s["speaker"], s["text"]) for s in data["segments"]] == [
        ("caller", "hello there"),
        ("agent", "hello there"),
    ]
    assert str(path) in result.output
    assert "caller  hello there" in result.output
    assert fake_worker == [("tiny.en", None)]


def test_transcribe_passes_models_dir(tmp_path, fake_worker, models_env):
    run_dir = make_run_dir(tmp_path / "run")
    result = runner.invoke(app, ["transcribe", str(run_dir), "--models-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert fake_worker == [("base.en", tmp_path)]


def test_transcribe_without_the_extra_names_it(tmp_path, models_env):
    run_dir = make_run_dir(tmp_path / "run")
    result = runner.invoke(app, ["transcribe", str(run_dir)])
    assert result.exit_code == 1
    assert model_cache.STT_EXTRA_HINT in result.output
    assert not (run_dir / "transcript.json").exists()


def test_transcribe_without_the_model_names_the_pull_command(tmp_path, models_env, stt_extra):
    run_dir = make_run_dir(tmp_path / "run")
    result = runner.invoke(app, ["transcribe", str(run_dir), "--stt-model", "tiny.en"])
    assert result.exit_code == 1
    assert "voiceqa models pull stt tiny.en" in plain(result.output)
    assert not (run_dir / "transcript.json").exists()


def test_transcribe_with_a_mistyped_model_folder_names_it(tmp_path, models_env, stt_extra):
    run_dir = make_run_dir(tmp_path / "run")
    typo = str(tmp_path / "models" / "tiny-enn")
    result = runner.invoke(app, ["transcribe", str(run_dir), "--stt-model", typo])
    assert result.exit_code == 1
    output = plain(result.output)
    assert "does not exist" in output
    assert "models pull" not in output


def test_bench_stt_with_a_mistyped_model_folder_names_it(tmp_path, models_env, stt_extra):
    typo = str(tmp_path / "models" / "tiny-enn")
    result = runner.invoke(app, ["bench", "stt", "--stt-model", typo])
    assert result.exit_code == 1
    assert "does not exist" in plain(result.output)


def test_bench_stt_with_a_model_that_will_not_load_names_the_problem(
    monkeypatch, models_env, stt_extra
):
    make_cached_model(models_env, "tiny.en", 5)

    def no_tokenizer(model, models_dir):
        raise model_cache.ModelNotAvailableError("model folder has no tokenizer.json")

    monkeypatch.setattr(cli, "_bench_engine", no_tokenizer)
    result = runner.invoke(app, ["bench", "stt", "--stt-model", "tiny.en"])
    assert result.exit_code == 1
    assert "no tokenizer.json" in plain(result.output)


def test_transcribe_a_folder_without_recordings(tmp_path, fake_worker, models_env):
    result = runner.invoke(app, ["transcribe", str(tmp_path)])
    assert result.exit_code == 2
    assert "caller.wav" in result.output
    assert fake_worker == []


def test_transcribe_a_corrupt_wav_fails_cleanly(tmp_path, fake_worker, models_env):
    run_dir = make_run_dir(tmp_path / "run")
    (run_dir / "caller.wav").write_bytes(b"NOTAWAVFILE1234567890")

    result = runner.invoke(app, ["transcribe", str(run_dir)])

    assert result.exit_code == 1
    assert "caller.wav" in result.output
    assert not (run_dir / "transcript.json").exists()


def test_transcribe_a_non_16_bit_wav_names_the_file(tmp_path, fake_worker, models_env):
    run_dir = make_run_dir(tmp_path / "run")
    with wave.open(str(run_dir / "caller.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x00" * 8000)

    result = runner.invoke(app, ["transcribe", str(run_dir)])

    assert result.exit_code == 1
    assert "caller.wav" in result.output
    assert not (run_dir / "transcript.json").exists()


def test_transcribe_interrupted(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_make_transcript", interrupted)
    run_dir = make_run_dir(tmp_path / "run")
    result = runner.invoke(app, ["transcribe", str(run_dir)])
    assert result.exit_code == 1
    assert "transcription interrupted" in result.output


def test_call_transcribe_interrupted_keeps_the_call_summary(
    tmp_path, fake_call, fake_worker, monkeypatch
):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_run_transcription", interrupted)
    result = runner.invoke(app, call_args(tmp_path, "--transcribe"))
    assert result.exit_code == 1
    assert "answered      yes" in result.stdout
    assert "transcription interrupted" in result.output


def test_transcribe_rejects_an_unknown_model(tmp_path, fake_worker, models_env):
    run_dir = make_run_dir(tmp_path / "run")
    result = runner.invoke(app, ["transcribe", str(run_dir), "--stt-model", "enormous.en"])
    assert result.exit_code == 2
    assert fake_worker == []


@pytest.fixture
def fake_call(monkeypatch):
    """Stands in for run_call: writes the two tracks and reports the call
    as answered (or not, if set)."""
    state = {"answered": True, "calls": 0}

    async def run_call(**kwargs):
        state["calls"] += 1
        record_dir = make_run_dir(kwargs["record_dir"])
        answered = state["answered"]
        return CallResult(
            answered=answered,
            status=200 if answered else 486,
            reason="OK" if answered else "Busy Here",
            duration=1.0 if answered else None,
            frames_sent=50,
            frames_received=50,
            first_audio_latency=None,
            record_dir=record_dir,
        )

    monkeypatch.setattr(cli, "run_call", run_call)
    return state


def call_args(tmp_path, *extra):
    return ["call", "sip:agent@127.0.0.1:5070", "--record-dir", str(tmp_path / "run"), *extra]


def test_call_transcribe_writes_the_transcript_after_the_call(
    tmp_path, fake_call, fake_worker, models_env
):
    result = runner.invoke(app, call_args(tmp_path, "--transcribe", "--stt-model", "tiny.en"))

    assert result.exit_code == 0, result.output
    path = tmp_path / "run" / "transcript.json"
    assert path.exists()
    assert re.search(r"transcript\s+" + re.escape(str(path)), result.output)
    assert fake_worker == [("tiny.en", None)]


def test_call_without_transcribe_writes_no_transcript(tmp_path, fake_call, fake_worker):
    result = runner.invoke(app, call_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "run" / "transcript.json").exists()
    assert fake_worker == []


def test_call_transcribe_without_the_extra_never_dials(tmp_path, fake_call, models_env):
    result = runner.invoke(app, call_args(tmp_path, "--transcribe"))

    # The extra is checked before dialing, so nothing is called and the
    # error names the fix.
    assert result.exit_code == 1
    assert model_cache.STT_EXTRA_HINT in result.stderr
    assert fake_call["calls"] == 0
    assert not (tmp_path / "run" / "transcript.json").exists()


def test_call_transcribe_without_the_model_never_dials(tmp_path, fake_call, models_env, stt_extra):
    result = runner.invoke(app, call_args(tmp_path, "--transcribe", "--stt-model", "tiny.en"))
    assert result.exit_code == 1
    assert "voiceqa models pull stt tiny.en" in result.stderr
    assert fake_call["calls"] == 0


def test_call_transcribe_with_a_model_missing_tokenizer_json_never_dials(
    tmp_path, fake_call, models_env, stt_extra
):
    make_cached_model(models_env, "tiny.en", 5)
    result = runner.invoke(app, call_args(tmp_path, "--transcribe", "--stt-model", "tiny.en"))
    assert result.exit_code == 1
    assert "tokenizer.json" in result.stderr
    assert fake_call["calls"] == 0


def test_call_transcribe_skips_an_unanswered_call(tmp_path, fake_call, fake_worker, models_env):
    fake_call["answered"] = False
    result = runner.invoke(app, call_args(tmp_path, "--transcribe"))
    assert result.exit_code == 1
    assert re.search(r"transcript\s+skipped", result.output)
    # The extra/model check (through the faked worker) still runs before
    # dialing, even though the call ends up unanswered.
    assert fake_worker == [("base.en", None)]


def test_call_transcribe_checks_the_model_name_before_calling(tmp_path, fake_call, fake_worker):
    result = runner.invoke(app, call_args(tmp_path, "--transcribe", "--stt-model", "enormous.en"))
    assert result.exit_code == 2
    assert fake_call["calls"] == 0


def test_call_stt_model_without_transcribe_is_a_bad_option(tmp_path, fake_call):
    result = runner.invoke(app, call_args(tmp_path, "--stt-model", "tiny.en"))
    assert result.exit_code == 2
    assert "--transcribe" in result.stderr
    assert fake_call["calls"] == 0


def test_call_models_dir_without_transcribe_is_a_bad_option(tmp_path, fake_call):
    result = runner.invoke(app, call_args(tmp_path, "--models-dir", str(tmp_path)))
    assert result.exit_code == 2
    assert "--transcribe" in result.stderr
    assert fake_call["calls"] == 0
