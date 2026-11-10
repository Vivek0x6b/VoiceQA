"""Speech and the two engine protocols: TextToSpeech and SpeechToText.

Audio crossing this boundary is int16 mono, same as the rest of
voiceqa.audio. Unlike telephony audio, which is always 8 kHz, Speech keeps
whatever rate its engine produced; resampling to 8 kHz (voiceqa.audio.dsp.
resample) happens later, only where a call actually needs it.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .models import Transcript


@dataclass(frozen=True, eq=False)
class Speech:
    """int16 mono audio at `rate` Hz, as produced by a TextToSpeech engine.

    eq is disabled because the default dataclass equality compares the
    `samples` arrays with `==`, which raises instead of returning a bool for
    arrays with more than one element. Compare `.rate` and `.samples` (with
    numpy's own array comparison) directly where a test needs it.
    """

    samples: np.ndarray
    rate: int

    def __post_init__(self) -> None:
        if self.samples.dtype != np.int16:
            raise TypeError(f"Speech samples must be int16, got {self.samples.dtype}")
        if self.samples.ndim != 1:
            raise ValueError(f"Speech samples must be mono (1-D), got shape {self.samples.shape}")
        if self.rate <= 0:
            raise ValueError(f"Speech rate must be positive, got {self.rate}")


class TextToSpeech(Protocol):
    """Converts text to Speech.

    `voice` picks among the names `voices()` returns. Engines with no voice
    concept of their own (WavPromptTts) ignore it.
    """

    @property
    def name(self) -> str: ...

    def voices(self) -> list[str]: ...

    def synthesize(self, text: str, voice: str | None = None) -> Speech: ...


class SpeechToText(Protocol):
    """Converts audio to a timed Transcript."""

    def transcribe(self, samples: np.ndarray, rate: int) -> Transcript: ...
