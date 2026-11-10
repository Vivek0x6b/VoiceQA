"""Deterministic fakes for tests: no OS, model or subprocess dependencies."""

import numpy as np

from voiceqa.audio.dsp import generate_tone

from .interfaces import Speech
from .models import Transcript


class FakeTts:
    """Tone bursts whose length follows the text, so a short and a long
    request produce audibly different, deterministic clips without any
    real speech synthesis.
    """

    def __init__(
        self,
        rate: int = 8000,
        freq_hz: float = 440.0,
        seconds_per_char: float = 0.05,
        min_seconds: float = 0.1,
        name: str = "fake",
    ) -> None:
        self._rate = rate
        self._freq_hz = freq_hz
        self._seconds_per_char = seconds_per_char
        self._min_seconds = min_seconds
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def voices(self) -> list[str]:
        return ["fake"]

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        duration = max(self._min_seconds, len(text) * self._seconds_per_char)
        samples = generate_tone(self._freq_hz, duration, sample_rate=self._rate)
        return Speech(samples=samples, rate=self._rate)


class FakeStt:
    """Returns a scripted Transcript regardless of the audio it is given.

    `transcript` is either a Transcript or plain data matching its schema
    (a dict, as model_dump() produces), so it stays picklable across a
    registry lookup and a spawned worker process.
    """

    def __init__(self, transcript: Transcript | dict, name: str = "fake") -> None:
        self._name = name
        self._transcript = (
            transcript
            if isinstance(transcript, Transcript)
            else Transcript.model_validate(transcript)
        )

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, samples: np.ndarray, rate: int) -> Transcript:
        return self._transcript
