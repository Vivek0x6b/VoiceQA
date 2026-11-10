import pytest
import yaml

from voiceqa.speech import FakeStt, FakeTts, WavPromptTts
from voiceqa.speech.registry import create_stt, create_tts


def test_create_tts_fake():
    tts = create_tts("fake")
    assert isinstance(tts, FakeTts)


def test_create_tts_wav(tmp_path):
    index_path = tmp_path / "index.yaml"
    index_path.write_text(yaml.safe_dump({}), encoding="utf-8")
    tts = create_tts("wav", index=index_path)
    assert isinstance(tts, WavPromptTts)


def test_create_stt_fake():
    transcript = {"engine": "fake", "model": "scripted", "language": "en"}
    stt = create_stt("fake", transcript=transcript)
    assert isinstance(stt, FakeStt)


@pytest.mark.parametrize("name", ["sapi", "say"])
def test_create_tts_reserved_names_not_implemented(name):
    with pytest.raises(NotImplementedError):
        create_tts(name)


def test_create_stt_reserved_name_not_implemented():
    with pytest.raises(NotImplementedError):
        create_stt("faster-whisper")


def test_create_tts_plugin_prefix_parses_but_is_not_implemented():
    with pytest.raises(NotImplementedError, match="plugin"):
        create_tts("plugin:kokoro")


def test_create_tts_unknown_name_lists_known_names():
    with pytest.raises(ValueError, match="fake") as exc_info:
        create_tts("not-a-real-engine")
    assert "wav" in str(exc_info.value)


def test_create_stt_unknown_name_lists_known_names():
    with pytest.raises(ValueError, match="fake"):
        create_stt("not-a-real-engine")
