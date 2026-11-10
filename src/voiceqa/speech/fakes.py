"""Deterministic fakes for tests: no OS, model or subprocess dependencies."""

import time

import numpy as np

from voiceqa.audio.dsp import generate_tone

from .interfaces import Speech
from .models import Segment, Transcript


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
    registry lookup and a spawned worker process. `delay` makes every job
    take that many seconds and `error` makes every job raise RuntimeError
    with that message, for testing the worker with slow or failing jobs.
    With `echo`, the transcript keeps the scripted engine, model and
    language but has one segment describing the job ("800 samples at 8000
    Hz"), so a result that went to the wrong job shows.
    """

    def __init__(
        self,
        transcript: Transcript | dict,
        name: str = "fake",
        delay: float = 0.0,
        error: str | None = None,
        echo: bool = False,
    ) -> None:
        self._name = name
        self._delay = delay
        self._error = error
        self._echo = echo
        self._transcript = (
            transcript
            if isinstance(transcript, Transcript)
            else Transcript.model_validate(transcript)
        )

    @property
    def name(self) -> str:
        return self._name

    def transcribe(self, samples: np.ndarray, rate: int) -> Transcript:
        if self._delay:
            time.sleep(self._delay)
        if self._error is not None:
            raise RuntimeError(self._error)
        if self._echo:
            text = f"{len(samples)} samples at {rate} Hz"
            segment = Segment(start=0.0, end=len(samples) / rate, text=text)
            return self._transcript.model_copy(update={"segments": [segment]})
        return self._transcript
