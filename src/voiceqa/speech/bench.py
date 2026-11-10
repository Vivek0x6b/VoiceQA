"""STT speed check on the LibriVox test clip, for `voiceqa bench stt`.

Informational only: it reports numbers for this CPU and never passes or
fails anything.
"""

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from voiceqa.audio.wav import read_wav

from .interfaces import SpeechToText

CLIP_INDEX = "utterances.json"


class Utterance(BaseModel):
    start: float
    end: float
    text: str


@dataclass(frozen=True, eq=False)
class BenchClip:
    samples: np.ndarray
    rate: int
    utterances: list[Utterance]


class SttBenchResult(BaseModel):
    engine: str
    model: str
    audio_seconds: float
    load_seconds: float
    clip_seconds: float
    rtf: float
    utterances: int
    utterance_median_ms: float
    utterance_p95_ms: float
    utterance_max_ms: float


def default_clip_dir() -> Path:
    """The clip shipped in the wheel, or tests/data/speech in a source checkout."""
    packaged = Path(__file__).resolve().parent.parent / "data" / "speech"
    if (packaged / CLIP_INDEX).is_file():
        return packaged
    return Path(__file__).resolve().parents[3] / "tests" / "data" / "speech"


def load_bench_clip(folder: Path) -> BenchClip:
    index = json.loads((folder / CLIP_INDEX).read_text(encoding="utf-8"))
    samples, rate = read_wav(folder / index["audio"])
    if samples.ndim != 1:
        raise ValueError(f"{folder / index['audio']}: bench clip must be mono")
    utterances = [Utterance.model_validate(u) for u in index["utterances"]]
    return BenchClip(samples=samples, rate=rate, utterances=utterances)


def run_stt_bench(
    make_engine: Callable[[], SpeechToText],
    clip: BenchClip,
    engine_name: str,
    model: str,
    clock: Callable[[], float] = time.perf_counter,
) -> SttBenchResult:
    t0 = clock()
    engine = make_engine()
    load_seconds = clock() - t0

    # Warm-up, not counted. The clock is still read around it so a scripted
    # test clock sees the same call pattern for every run.
    clock()
    engine.transcribe(clip.samples, clip.rate)
    clock()

    t0 = clock()
    engine.transcribe(clip.samples, clip.rate)
    clip_seconds = clock() - t0

    latencies = []
    for u in clip.utterances:
        part = clip.samples[round(u.start * clip.rate) : round(u.end * clip.rate)]
        t0 = clock()
        engine.transcribe(part, clip.rate)
        latencies.append((clock() - t0) * 1000)

    audio_seconds = len(clip.samples) / clip.rate
    lat = np.array(latencies, dtype=np.float64)
    return SttBenchResult(
        engine=engine_name,
        model=model,
        audio_seconds=audio_seconds,
        load_seconds=load_seconds,
        clip_seconds=clip_seconds,
        rtf=clip_seconds / audio_seconds,
        utterances=len(latencies),
        utterance_median_ms=float(np.median(lat)) if len(lat) else 0.0,
        utterance_p95_ms=float(np.percentile(lat, 95)) if len(lat) else 0.0,
        utterance_max_ms=float(lat.max()) if len(lat) else 0.0,
    )
