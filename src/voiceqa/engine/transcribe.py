"""Transcript of a recorded call, written as transcript.json in the run folder.

Each track (caller.wav, agent.wav) is transcribed on its own and every
segment is tagged with its speaker. Times are seconds from sample 0 of the
WAVs, which is also the zero of events.json, and segments are sorted by
start time. The file uses the Transcript schema (voiceqa.speech.models).

Tracks are read in blocks and sent in chunks of up to about ten minutes, so
a long call never has to fit in memory as one array or one STT job. Each
chunk is cut at the quietest 20 ms frame in its last two seconds, to keep
words from being split, and its segment times are moved by where it starts.

The STT engine is anything with an async transcribe(samples, rate), in
practice an SttWorker, so the speech work stays out of this process.
"""

import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np

from voiceqa.audio.dsp import to_mono
from voiceqa.audio.wav import read_wav_blocks
from voiceqa.speech.models import Segment, Transcript

TRACKS = (("caller", "caller.wav"), ("agent", "agent.wav"))
TRANSCRIPT_FILE = "transcript.json"
CHUNK_SECONDS = 600.0
CUT_WINDOW_SECONDS = 2.0


class AsyncStt(Protocol):
    async def transcribe(self, samples: np.ndarray, rate: int) -> Transcript: ...


def _wav_rate(path: Path) -> int:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getframerate()
    except wave.Error as e:
        raise ValueError(f"{path}: not a PCM WAV file ({e})") from e


def _quiet_cut(samples: np.ndarray, end: int, window: int, frame: int) -> int:
    """Start of the quietest whole frame in samples[end - window : end]."""
    first = end - window
    whole = window // frame
    if whole == 0:
        return end
    frames = samples[first : first + whole * frame].reshape(whole, frame).astype(np.float64)
    return first + int(np.argmin((frames**2).sum(axis=1))) * frame


def read_chunks(
    path: str | Path, chunk_seconds: float = CHUNK_SECONDS
) -> Iterator[tuple[int, np.ndarray, int]]:
    """(start sample, int16 mono samples, rate) for each chunk of a WAV.

    An empty file still gives one empty chunk, so its track is sent to the
    engine once like any other.
    """
    rate = _wav_rate(Path(path))
    chunk = max(1, round(chunk_seconds * rate))
    window = min(round(CUT_WINDOW_SECONDS * rate), chunk // 2)
    frame = max(1, rate // 50)
    start = 0
    buffered = np.zeros(0, dtype=np.int16)
    sent = False
    for block in read_wav_blocks(path, rate):
        buffered = np.concatenate([buffered, to_mono(block) if block.ndim > 1 else block])
        while len(buffered) > chunk:
            cut = _quiet_cut(buffered, chunk, window, frame)
            yield start, buffered[:cut], rate
            sent = True
            start += cut
            buffered = buffered[cut:]
    if len(buffered) or not sent:
        yield start, buffered, rate


def _shifted(segment: Segment, offset: float, speaker: str) -> Segment:
    words = [
        w.model_copy(update={"start": w.start + offset, "end": w.end + offset})
        for w in segment.words
    ]
    return segment.model_copy(
        update={
            "start": segment.start + offset,
            "end": segment.end + offset,
            "words": words,
            "speaker": speaker,
        }
    )


async def transcribe_run(
    run_dir: str | Path, stt: AsyncStt, chunk_seconds: float = CHUNK_SECONDS
) -> Transcript:
    run_dir = Path(run_dir)
    paths = [(speaker, run_dir / name) for speaker, name in TRACKS]
    for _, path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir}: no {path.name} in the run folder")

    segments: list[Segment] = []
    parts: list[Transcript] = []
    for speaker, path in paths:
        for start, samples, rate in read_chunks(path, chunk_seconds):
            part = await stt.transcribe(samples, rate)
            parts.append(part)
            segments.extend(_shifted(s, start / rate, speaker) for s in part.segments)

    segments.sort(key=lambda s: s.start)
    languages = sorted({p.language for p in parts if p.language})
    return Transcript(
        segments=segments,
        engine=parts[0].engine,
        model=parts[0].model,
        language=",".join(languages),
    )


def write_transcript(run_dir: str | Path, transcript: Transcript) -> Path:
    path = Path(run_dir) / TRANSCRIPT_FILE
    path.write_text(transcript.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
