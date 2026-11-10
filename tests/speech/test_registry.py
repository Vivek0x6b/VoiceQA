import importlib
import re
import sys

import pytest
import yaml

from voiceqa.speech import FakeStt, FakeTts, WavPromptTts
from voiceqa.speech.model_cache import STT_EXTRA_HINT
from voiceqa.speech.plugin_tts import PluginTts
from voiceqa.speech.registry import create_stt, create_tts, default_tts_name
from voiceqa.speech.tts_process import TtsError


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


@pytest.mark.parametrize(
    ("name", "home"), [("sapi", "win32"), ("say", "darwin")], ids=["sapi", "say"]
)
def test_create_tts_os_engines_off_their_os_name_the_os(name, home):
    if sys.platform == home:
        pytest.skip(f"{name} is available on this OS")
    with pytest.raises(TtsError, match="needs"):
        create_tts(name)


@pytest.mark.parametrize(
    ("platform", "expected"),
    [("win32", "sapi"), ("darwin", "say"), ("linux", "wav")],
)
def test_default_tts_name_per_os(monkeypatch, platform, expected):
    monkeypatch.setattr(sys, "platform", platform)
    assert default_tts_name() == expected


def test_importing_the_whisper_backend_without_the_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.delitem(sys.modules, "voiceqa.speech.whisper", raising=False)
    with pytest.raises(ImportError, match=re.escape(STT_EXTRA_HINT)):
        importlib.import_module("voiceqa.speech.whisper")


def test_create_stt_faster_whisper_without_the_extra_names_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.delitem(sys.modules, "voiceqa.speech.whisper", raising=False)
    with pytest.raises(ImportError, match=re.escape(STT_EXTRA_HINT)):
        create_stt("faster-whisper", model="tiny.en")


def test_create_tts_plugin_prefix_builds_a_plugin_engine():
    tts = create_tts("plugin:kokoro")
    assert isinstance(tts, PluginTts)
    assert tts.name == "plugin:kokoro"


def test_create_stt_rejects_plugins():
    with pytest.raises(NotImplementedError, match="STT plugins"):
        create_stt("plugin:whatever")


def test_create_tts_unknown_name_lists_known_names():
    with pytest.raises(ValueError, match="fake") as exc_info:
        create_tts("not-a-real-engine")
    assert "wav" in str(exc_info.value)


def test_create_stt_unknown_name_lists_known_names():
    with pytest.raises(ValueError, match="fake"):
        create_stt("not-a-real-engine")
