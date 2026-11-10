"""Maps text to a WAV file, exactly, for reproducible prompts.

Construct either from a YAML index file (text -> file path, relative to the
index file's own folder) or from an in-memory mapping of text to WAV path
(handy for a single prompt with no index file on disk). Text is matched
after stripping leading and trailing whitespace; matching is case sensitive.
Looking up text with no matching prompt is an error naming the missing
text, never silence.

The returned Speech keeps the WAV file's own sample rate; resampling to the
rate a call needs happens elsewhere. Stereo files are downmixed to mono
with voiceqa.audio.dsp.to_mono.
"""

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from voiceqa.audio.dsp import to_mono
from voiceqa.audio.wav import read_wav

from .interfaces import Speech


class WavPromptTts:
    def __init__(
        self,
        index: str | os.PathLike | Mapping[str, str | os.PathLike],
        *,
        name: str = "wav",
    ) -> None:
        self._name = name
        if isinstance(index, Mapping):
            self._prompts = {str(text).strip(): Path(path) for text, path in index.items()}
        else:
            index_path = Path(index)
            with open(index_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            if not isinstance(raw, Mapping):
                raise ValueError(f"{index_path}: index must be a YAML mapping of text to file")
            folder = index_path.parent
            self._prompts = {str(text).strip(): folder / path for text, path in raw.items()}

    @property
    def name(self) -> str:
        return self._name

    def voices(self) -> list[str]:
        """WavPromptTts has no selectable voices; text picks the file directly."""
        return []

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        key = text.strip()
        try:
            path = self._prompts[key]
        except KeyError:
            raise ValueError(f"no WAV prompt for text {key!r}") from None
        samples, rate = read_wav(path)
        return Speech(samples=to_mono(samples), rate=rate)
