from types import SimpleNamespace

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.speech import create_stt, model_cache
from voiceqa.speech.model_cache import ModelNotAvailableError

pytest.importorskip("faster_whisper", reason="needs the [stt] extra")

from voiceqa.speech import whisper  # noqa: E402


class FakeWhisperModel:
    """Stands in for faster_whisper.WhisperModel and records what it was given."""

    instances: list["FakeWhisperModel"] = []

    def __init__(self, model_size_or_path, **kwargs):
        self.path = model_size_or_path
        self.kwargs = kwargs
        self.calls = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))
        words = [
            SimpleNamespace(start=0.1, end=0.4, word=" Hello", probability=0.9),
            SimpleNamespace(start=0.45, end=0.8, word=" there.", probability=0.8),
        ]
        segments = [SimpleNamespace(start=0.1, end=0.8, text=" Hello there.", words=words)]
        info = SimpleNamespace(language="en")
        return iter(segments), info


@pytest.fixture
def fake_model(monkeypatch, tmp_path, make_cached_model):
    FakeWhisperModel.instances = []
    monkeypatch.setattr(whisper, "WhisperModel", FakeWhisperModel)
    snapshot = make_cached_model(tmp_path, "Systran/faster-whisper-tiny.en", {"model.bin": b"x"})
    return SimpleNamespace(models_dir=tmp_path, snapshot=snapshot)


def test_defaults_are_cpu_int8_with_word_timestamps_and_vad(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)
    stt.transcribe(generate_tone(440, 0.5), 8000)

    model = FakeWhisperModel.instances[0]
    assert model.path == str(fake_model.snapshot)
    assert model.kwargs["device"] == "cpu"
    assert model.kwargs["compute_type"] == "int8"
    _, options = model.calls[0]
    assert options["word_timestamps"] is True
    assert options["vad_filter"] is True


def test_default_model_is_base_en():
    assert whisper.DEFAULT_MODEL == "base.en"
    assert whisper.FasterWhisperStt.__init__.__defaults__[0] == "base.en"


def test_8k_audio_is_resampled_to_16k_float(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)
    tone = generate_tone(440, 1.0, sample_rate=8000, amplitude=0.5)

    stt.transcribe(tone, 8000)

    audio, _ = FakeWhisperModel.instances[0].calls[0]
    assert audio.dtype == np.float32
    assert len(audio) == 16000
    # A 0.5 amplitude tone stays a 0.5 amplitude tone after resampling.
    assert abs(float(np.max(np.abs(audio[1000:-1000]))) - 0.5) < 0.01
    # And it is still 440 Hz: the spectrum peaks at bin 440 of a 1 s window.
    spectrum = np.abs(np.fft.rfft(audio))
    assert int(np.argmax(spectrum)) == 440


def test_16k_audio_is_passed_without_resampling(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)
    tone = generate_tone(440, 0.25, sample_rate=16000)

    stt.transcribe(tone, 16000)

    audio, _ = FakeWhisperModel.instances[0].calls[0]
    np.testing.assert_array_equal(audio, tone.astype(np.float32) / 32768)


def test_transcript_maps_segments_and_words(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)

    transcript = stt.transcribe(generate_tone(440, 0.5), 8000)

    assert transcript.engine == "faster-whisper"
    assert transcript.model == "tiny.en"
    assert transcript.language == "en"
    [segment] = transcript.segments
    assert (segment.start, segment.end, segment.text) == (0.1, 0.8, "Hello there.")
    assert [(w.text, w.start, w.end, w.probability) for w in segment.words] == [
        ("Hello", 0.1, 0.4, 0.9),
        ("there.", 0.45, 0.8, 0.8),
    ]


def test_empty_audio_gives_an_empty_transcript(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)
    transcript = stt.transcribe(np.zeros(0, dtype=np.int16), 8000)
    assert transcript.segments == []
    assert FakeWhisperModel.instances[0].calls == []


def test_rejects_non_int16_or_stereo(fake_model):
    stt = whisper.FasterWhisperStt(model="tiny.en", models_dir=fake_model.models_dir)
    with pytest.raises(TypeError):
        stt.transcribe(np.zeros(100, dtype=np.float32), 8000)
    with pytest.raises(ValueError):
        stt.transcribe(np.zeros((100, 2), dtype=np.int16), 8000)


def test_missing_model_is_an_error_naming_the_pull_command(fake_model, tmp_path):
    empty = tmp_path / "empty"
    with pytest.raises(ModelNotAvailableError, match="voiceqa models pull stt base.en"):
        whisper.FasterWhisperStt(model="base.en", models_dir=empty)
    assert FakeWhisperModel.instances == []


def test_the_backend_never_downloads(fake_model, monkeypatch, tmp_path):
    def no_network(*args, **kwargs):
        raise AssertionError("the backend must not download")

    monkeypatch.setattr(model_cache, "_download", no_network)
    with pytest.raises(ModelNotAvailableError):
        whisper.FasterWhisperStt(model="base.en", models_dir=tmp_path / "empty")
    # No way to ask it to, either: downloads go through voiceqa models pull,
    # which says what it fetches and where.
    with pytest.raises(TypeError):
        whisper.FasterWhisperStt(model="base.en", models_dir=tmp_path / "empty", download=True)


def test_models_dir_env_is_used_when_no_dir_is_given(fake_model, monkeypatch):
    monkeypatch.setenv("VOICEQA_MODELS_DIR", str(fake_model.models_dir))
    whisper.FasterWhisperStt(model="tiny.en")
    assert FakeWhisperModel.instances[0].path == str(fake_model.snapshot)


def test_registry_builds_it_from_plain_kwargs(fake_model):
    stt = create_stt("faster-whisper", model="tiny.en", models_dir=str(fake_model.models_dir))
    assert isinstance(stt, whisper.FasterWhisperStt)
    assert stt.name == "faster-whisper"
    assert stt.model == "tiny.en"
