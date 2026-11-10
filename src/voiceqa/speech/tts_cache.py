"""Per-user cache of synthesized speech, so reruns are fast and identical.

Files are keyed by a hash of (engine name, voice, text) and live in
tts_cache_dir(): <root>/tts, where the root is VOICEQA_CACHE_DIR if set,
else %LOCALAPPDATA%/voiceqa/cache on Windows, ~/Library/Caches/voiceqa on
macOS, and $XDG_CACHE_HOME/voiceqa or ~/.cache/voiceqa elsewhere.

A cache file is written to a temp file in the same folder and renamed into
place, so a second run never reads a half-written WAV. A cache that cannot
be read or written never stops synthesis: a broken file is synthesized
again and overwritten, and a failed write is skipped.
"""

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from voiceqa.audio.wav import read_wav, write_wav

from .interfaces import Speech, TextToSpeech

CACHE_DIR_ENV = "VOICEQA_CACHE_DIR"
_KEY_VERSION = 1


def cache_root() -> Path:
    if os.environ.get(CACHE_DIR_ENV):
        return Path(os.environ[CACHE_DIR_ENV]).expanduser()
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        return base / "voiceqa" / "cache"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "voiceqa"
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "voiceqa"


def tts_cache_dir() -> Path:
    return cache_root() / "tts"


def cache_key(engine: str, voice: str | None, text: str) -> str:
    raw = json.dumps([_KEY_VERSION, engine, voice, text], ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()


class CachedTts:
    """Wraps a TextToSpeech engine and caches what it synthesizes."""

    def __init__(self, engine: TextToSpeech) -> None:
        self._engine = engine

    @property
    def name(self) -> str:
        return self._engine.name

    def voices(self) -> list[str]:
        return self._engine.voices()

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        folder = tts_cache_dir()
        path = folder / f"{cache_key(self.name, voice, text)}.wav"
        cached = _read(path)
        if cached is not None:
            return cached
        speech = self._engine.synthesize(text, voice)
        _write(folder, path, speech)
        return speech


def _read(path: Path) -> Speech | None:
    try:
        samples, rate = read_wav(path)
    except (OSError, ValueError, EOFError):
        return None
    if samples.ndim != 1:
        return None
    return Speech(samples=samples, rate=rate)


def _write(folder: Path, path: Path, speech: Speech) -> None:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=folder, prefix=".", suffix=".tmp")
    except OSError:
        return
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        write_wav(tmp, speech.samples, speech.rate)
        os.replace(tmp, path)
    except OSError:
        # Another run may hold the target open on Windows; its copy is as good.
        tmp.unlink(missing_ok=True)
