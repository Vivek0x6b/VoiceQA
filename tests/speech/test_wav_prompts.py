import numpy as np
import pytest
import yaml

from voiceqa.audio.wav import write_wav
from voiceqa.speech import WavPromptTts


def _write_tone(path, rate=16000, channels=1):
    samples = np.full(rate // 10, 1000, dtype=np.int16)
    if channels > 1:
        samples = np.column_stack([samples] * channels)
    write_wav(path, samples, rate)


def test_wav_prompt_tts_loads_from_yaml_index(tmp_path):
    _write_tone(tmp_path / "hello.wav", rate=16000)
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({"hello": "hello.wav"}), encoding="utf-8")

    tts = WavPromptTts(index_path)
    speech = tts.synthesize("hello")
    assert speech.rate == 16000
    assert speech.samples.dtype == np.int16
    assert speech.samples.ndim == 1


def test_wav_prompt_tts_matches_text_after_strip(tmp_path):
    _write_tone(tmp_path / "hello.wav")
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({"hello": "hello.wav"}), encoding="utf-8")

    tts = WavPromptTts(index_path)
    tts.synthesize("  hello  ")  # does not raise


def test_wav_prompt_tts_is_case_sensitive(tmp_path):
    _write_tone(tmp_path / "hello.wav")
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({"hello": "hello.wav"}), encoding="utf-8")

    tts = WavPromptTts(index_path)
    with pytest.raises(ValueError, match="Hello"):
        tts.synthesize("Hello")


def test_wav_prompt_tts_unknown_text_names_it(tmp_path):
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({"hello": "hello.wav"}), encoding="utf-8")

    tts = WavPromptTts(index_path)
    with pytest.raises(ValueError, match="goodbye"):
        tts.synthesize("goodbye")


@pytest.mark.parametrize("content", [["hello.wav", "bye.wav"], "hello.wav"])
def test_wav_prompt_tts_index_that_is_not_a_mapping_names_the_file(tmp_path, content):
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a YAML mapping") as caught:
        WavPromptTts(index_path)
    assert str(index_path) in str(caught.value)


def test_wav_prompt_tts_downmixes_stereo(tmp_path):
    _write_tone(tmp_path / "stereo.wav", rate=8000, channels=2)
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({"stereo": "stereo.wav"}), encoding="utf-8")

    tts = WavPromptTts(index_path)
    speech = tts.synthesize("stereo")
    assert speech.samples.ndim == 1


def test_wav_prompt_tts_from_mapping_without_a_yaml_file(tmp_path):
    _write_tone(tmp_path / "hi.wav")
    tts = WavPromptTts({"hi": tmp_path / "hi.wav"})
    speech = tts.synthesize("hi")
    assert speech.samples.dtype == np.int16


def test_wav_prompt_tts_name_and_voices(tmp_path):
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    tts = WavPromptTts(index_path)
    assert tts.name == "wav"
    assert tts.voices() == []
