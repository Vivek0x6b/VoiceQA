"""faster-whisper speech to text, installed with the [stt] extra.

The model is loaded once, in the constructor. A model that is not on disk
is an error unless `download=True`, so nothing is fetched in the middle of a
call; `voiceqa models pull stt <model>` gets it ahead of time.
"""

import os

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError as exc:
    raise ImportError(
        "the faster-whisper STT backend needs the [stt] extra: pip install voiceqa[stt]"
    ) from exc

from voiceqa.audio.dsp import resample

from .model_cache import ModelNotAvailableError, find_model, pull_stt_model, resolve_models_dir
from .models import Segment, Transcript, Word

DEFAULT_MODEL = "base.en"
WHISPER_RATE = 16000


class FasterWhisperStt:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        models_dir: str | os.PathLike | None = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
        beam_size: int = 5,
        word_timestamps: bool = True,
        vad_filter: bool = True,
        cpu_threads: int = 0,
        download: bool = False,
    ) -> None:
        self._model_name = model
        self._language = language
        self._beam_size = beam_size
        self._word_timestamps = word_timestamps
        self._vad_filter = vad_filter

        cache = resolve_models_dir(models_dir)
        path = find_model(model, cache)
        if path is None:
            if not download:
                raise ModelNotAvailableError(
                    f"faster-whisper model {model} is not in {cache}. "
                    f"Download it first: voiceqa models pull stt {model}"
                )
            path = pull_stt_model(model, cache)
        self._model = WhisperModel(
            str(path), device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    @property
    def name(self) -> str:
        return "faster-whisper"

    @property
    def model(self) -> str:
        return self._model_name

    def transcribe(self, samples: np.ndarray, rate: int) -> Transcript:
        if samples.dtype != np.int16:
            raise TypeError(f"samples must be int16, got {samples.dtype}")
        if samples.ndim != 1:
            raise ValueError(f"samples must be mono (1-D), got shape {samples.shape}")
        language = self._language or ("en" if self._model_name.endswith(".en") else "")
        if len(samples) == 0:
            return Transcript(engine=self.name, model=self._model_name, language=language)

        audio = resample(samples, rate, WHISPER_RATE).astype(np.float32) / 32768
        segments, info = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=self._beam_size,
            word_timestamps=self._word_timestamps,
            vad_filter=self._vad_filter,
        )
        out = [
            Segment(
                start=s.start,
                end=s.end,
                text=s.text.strip(),
                words=[
                    Word(start=w.start, end=w.end, text=w.word.strip(), probability=w.probability)
                    for w in (s.words or [])
                ],
            )
            for s in segments
        ]
        return Transcript(
            segments=out, engine=self.name, model=self._model_name, language=info.language
        )
