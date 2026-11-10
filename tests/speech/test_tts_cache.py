import sys
from pathlib import Path

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.speech import tts_cache
from voiceqa.speech.interfaces import Speech
from voiceqa.speech.tts_cache import CachedTts, cache_root, tts_cache_dir


class CountingTts:
    def __init__(self, name="counting"):
        self._name = name
        self.calls = []

    @property
    def name(self):
        return self._name

    def voices(self):
        return ["a", "b"]

    def synthesize(self, text, voice=None):
        self.calls.append((text, voice))
        samples = generate_tone(300 + 10 * len(self.calls), 0.1, sample_rate=16000)
        return Speech(samples=samples, rate=16000)


@pytest.fixture
def cache_env(monkeypatch, tmp_path):
    monkeypatch.setenv("VOICEQA_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache" / "tts"


def test_env_var_overrides_the_cache_root(cache_env):
    assert tts_cache_dir() == cache_env


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "/lad"}, Path("/lad/voiceqa/cache")),
        ("darwin", {}, Path("/home/u/Library/Caches/voiceqa")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/voiceqa")),
        ("linux", {}, Path("/home/u/.cache/voiceqa")),
        ("win32", {}, Path("/home/u/AppData/Local/voiceqa/cache")),
    ],
)
def test_cache_root_per_os(monkeypatch, platform, env, expected):
    for name in ("VOICEQA_CACHE_DIR", "LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/u")))
    assert cache_root() == expected


def test_second_request_comes_from_the_cache(cache_env):
    engine = CountingTts()
    tts = CachedTts(engine)

    first = tts.synthesize("hello", voice="a")
    second = tts.synthesize("hello", voice="a")

    assert engine.calls == [("hello", "a")]
    assert second.rate == first.rate
    np.testing.assert_array_equal(second.samples, first.samples)
    assert len(list(cache_env.glob("*.wav"))) == 1


def test_a_new_engine_instance_reads_the_same_cache(cache_env):
    CachedTts(CountingTts()).synthesize("hello")
    engine = CountingTts()
    CachedTts(engine).synthesize("hello")
    assert engine.calls == []


@pytest.mark.parametrize(
    "other",
    [("counting", "hello", "b"), ("counting", "hello!", "a"), ("other", "hello", "a")],
    ids=["voice", "text", "engine"],
)
def test_engine_voice_and_text_are_all_in_the_key(cache_env, other):
    CachedTts(CountingTts()).synthesize("hello", voice="a")
    name, text, voice = other
    engine = CountingTts(name)
    CachedTts(engine).synthesize(text, voice=voice)
    assert engine.calls == [(text, voice)]
    assert len(list(cache_env.glob("*.wav"))) == 2


def test_no_voice_and_an_empty_voice_name_are_different_keys(cache_env):
    CachedTts(CountingTts()).synthesize("hello", voice=None)
    engine = CountingTts()
    CachedTts(engine).synthesize("hello", voice="")
    assert engine.calls == [("hello", "")]


def test_writes_leave_no_temp_files(cache_env):
    tts = CachedTts(CountingTts())
    for text in ("one", "two", "three"):
        tts.synthesize(text)
    names = sorted(p.suffix for p in cache_env.iterdir())
    assert names == [".wav", ".wav", ".wav"]


def test_a_write_that_fails_halfway_leaves_nothing_behind(cache_env, monkeypatch):
    def half_write(path, samples, rate):
        Path(path).write_bytes(b"RIFF half")
        raise OSError("disk full")

    monkeypatch.setattr(tts_cache, "write_wav", half_write)
    engine = CountingTts()

    speech = CachedTts(engine).synthesize("hello")

    assert len(speech.samples) == 1600
    assert list(cache_env.iterdir()) == []


def test_a_broken_cache_file_is_replaced(cache_env):
    CachedTts(CountingTts()).synthesize("hello")
    (path,) = cache_env.glob("*.wav")
    path.write_bytes(b"RIFF junk")

    engine = CountingTts()
    speech = CachedTts(engine).synthesize("hello")

    assert engine.calls == [("hello", None)]
    assert len(speech.samples) == 1600
    assert path.stat().st_size > 1000


def test_cached_tts_keeps_the_engine_name_and_voices():
    tts = CachedTts(CountingTts("sapi"))
    assert tts.name == "sapi"
    assert tts.voices() == ["a", "b"]


def test_an_unwritable_cache_still_returns_speech(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a folder")
    monkeypatch.setenv("VOICEQA_CACHE_DIR", str(blocker))
    engine = CountingTts()
    speech = CachedTts(engine).synthesize("hello")
    assert len(speech.samples) == 1600
