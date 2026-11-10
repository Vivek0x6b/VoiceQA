import asyncio
import json

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import write_wav
from voiceqa.engine.transcribe import read_chunks, transcribe_run, write_transcript
from voiceqa.speech import Segment, Transcript
from voiceqa.speech.worker import SttWorker

RATE = 8000
FRAME = RATE // 50


class EnergyStt:
    """Stands in for an STT engine: one segment per run of loud 20 ms
    frames, timed from the start of the audio it is given, with text
    naming the track it heard. Records every call it gets."""

    def __init__(self, text="speech"):
        self.text = text
        self.calls = []

    async def transcribe(self, samples, rate):
        self.calls.append((len(samples), rate))
        whole = len(samples) // FRAME
        frames = samples[: whole * FRAME].reshape(whole, FRAME).astype(np.float64)
        loud = np.sqrt((frames**2).mean(axis=1)) > 300
        segments = []
        start = None
        for i, is_loud in enumerate([*loud, False]):
            if is_loud and start is None:
                start = i
            elif not is_loud and start is not None:
                segments.append(
                    Segment(start=start * FRAME / rate, end=i * FRAME / rate, text=self.text)
                )
                start = None
        return Transcript(segments=segments, engine="energy", model="none", language="en")


def track(seconds, bursts):
    """Silence with a 440 Hz tone over each (start, end) in seconds."""
    samples = np.zeros(int(seconds * RATE), dtype=np.int16)
    for start, end in bursts:
        tone = generate_tone(440, end - start, sample_rate=RATE)
        samples[int(start * RATE) : int(start * RATE) + len(tone)] = tone
    return samples


def make_run(run_dir, caller_bursts, agent_bursts, seconds=9.0):
    run_dir.mkdir(exist_ok=True)
    write_wav(run_dir / "caller.wav", track(seconds, caller_bursts), RATE)
    write_wav(run_dir / "agent.wav", track(seconds, agent_bursts), RATE)
    return run_dir


def test_segments_are_tagged_with_their_speaker_and_sorted_by_start(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[(0.5, 1.0), (4.0, 4.5)], agent_bursts=[(2.0, 3.0)])

    transcript = asyncio.run(transcribe_run(run_dir, EnergyStt()))

    assert [(s.speaker, s.start) for s in transcript.segments] == [
        ("caller", pytest.approx(0.5, abs=0.03)),
        ("agent", pytest.approx(2.0, abs=0.03)),
        ("caller", pytest.approx(4.0, abs=0.03)),
    ]
    assert transcript.engine == "energy"
    assert transcript.model == "none"
    assert transcript.language == "en"


def test_agent_speech_two_seconds_in_keeps_its_offset(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[], agent_bursts=[(2.0, 2.6)])

    transcript = asyncio.run(transcribe_run(run_dir, EnergyStt()))

    (segment,) = transcript.segments
    assert segment.speaker == "agent"
    assert segment.start == pytest.approx(2.0, abs=0.2)
    assert segment.end == pytest.approx(2.6, abs=0.2)


def test_long_tracks_go_in_chunks_and_keep_their_times(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[], agent_bursts=[(2.0, 2.4), (7.0, 7.4)])
    stt = EnergyStt()

    transcript = asyncio.run(transcribe_run(run_dir, stt, chunk_seconds=3.0))

    assert len(stt.calls) > 2, "each track is split into chunks"
    assert all(length <= 4 * RATE for length, _ in stt.calls)
    starts = [s.start for s in transcript.segments]
    assert starts == [pytest.approx(2.0, abs=0.03), pytest.approx(7.0, abs=0.03)]


def test_word_times_move_with_their_chunk(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[], agent_bursts=[], seconds=7.0)

    class WordStt:
        async def transcribe(self, samples, rate):
            word = {"start": 0.25, "end": 0.5, "text": "hi", "probability": 0.9}
            seg = {"start": 0.25, "end": 0.5, "text": "hi", "words": [word]}
            return Transcript.model_validate(
                {"segments": [seg], "engine": "w", "model": "w", "language": "en"}
            )

    transcript = asyncio.run(transcribe_run(run_dir, WordStt(), chunk_seconds=3.0))

    agent = [s for s in transcript.segments if s.speaker == "agent"]
    chunk_starts = [start / RATE for start, _, _ in read_chunks(run_dir / "agent.wav", 3.0)]
    assert len(chunk_starts) >= 3
    assert [s.start for s in agent] == pytest.approx([t + 0.25 for t in chunk_starts])
    for seg in agent:
        (word,) = seg.words
        assert (word.start, word.end) == pytest.approx((seg.start, seg.end))


def test_chunks_cut_at_the_quietest_frame_near_the_chunk_end(tmp_path):
    # Loud everywhere except a short gap at 2.5 s, inside the cut window of
    # a 3 s chunk: the cut lands in the gap, not in the middle of a tone.
    samples = track(6.0, [(0.0, 2.5), (2.56, 6.0)])
    write_wav(tmp_path / "t.wav", samples, RATE)

    chunks = list(read_chunks(tmp_path / "t.wav", chunk_seconds=3.0))

    starts = [start for start, _, _ in chunks]
    assert starts[0] == 0
    assert 2.5 * RATE <= starts[1] <= 2.56 * RATE
    assert sum(len(c) for _, c, _ in chunks) == len(samples)
    assert all(rate == RATE for _, _, rate in chunks)
    joined = np.concatenate([c for _, c, _ in chunks])
    assert np.array_equal(joined, samples)


def test_an_empty_track_is_still_sent_once(tmp_path):
    write_wav(tmp_path / "t.wav", np.zeros(0, dtype=np.int16), RATE)
    chunks = list(read_chunks(tmp_path / "t.wav"))
    assert [(start, len(c), rate) for start, c, rate in chunks] == [(0, 0, RATE)]


def test_stereo_tracks_are_downmixed(tmp_path):
    stereo = np.stack([track(1.0, [(0.2, 0.4)])] * 2, axis=1)
    write_wav(tmp_path / "t.wav", stereo, RATE)
    ((_, chunk, _),) = list(read_chunks(tmp_path / "t.wav"))
    assert chunk.ndim == 1
    assert len(chunk) == RATE


def test_missing_track_is_an_error_naming_it(tmp_path):
    write_wav(tmp_path / "caller.wav", track(1.0, []), RATE)
    with pytest.raises(FileNotFoundError, match="agent.wav"):
        asyncio.run(transcribe_run(tmp_path, EnergyStt()))


def test_transcript_json_schema(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[(0.5, 1.0)], agent_bursts=[(2.0, 3.0)])
    transcript = asyncio.run(transcribe_run(run_dir, EnergyStt()))

    path = write_transcript(run_dir, transcript)

    assert path == run_dir / "transcript.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"segments", "engine", "model", "language"}
    assert data["engine"] == "energy"
    for seg in data["segments"]:
        assert set(seg) == {"start", "end", "text", "words", "speaker"}
        assert seg["speaker"] in ("caller", "agent")
    starts = [seg["start"] for seg in data["segments"]]
    assert starts == sorted(starts)
    assert Transcript.model_validate(data) == transcript


def test_tracks_that_disagree_on_language_list_both(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[], agent_bursts=[], seconds=1.0)

    class ByLength:
        def __init__(self):
            self.n = 0

        async def transcribe(self, samples, rate):
            self.n += 1
            lang = "en" if self.n == 1 else "de"
            return Transcript(engine="x", model="y", language=lang)

    transcript = asyncio.run(transcribe_run(run_dir, ByLength()))
    assert transcript.language == "de,en"


def test_runs_through_the_worker_process(tmp_path):
    run_dir = make_run(tmp_path, caller_bursts=[], agent_bursts=[], seconds=1.0)
    scripted = Transcript(
        segments=[Segment(start=0.25, end=0.75, text="hello")],
        engine="fake",
        model="scripted",
        language="en",
    )

    async def main():
        async with SttWorker("fake", transcript=scripted.model_dump()) as worker:
            return await asyncio.wait_for(transcribe_run(run_dir, worker), 20)

    transcript = asyncio.run(main())

    assert [(s.speaker, s.text) for s in transcript.segments] == [
        ("caller", "hello"),
        ("agent", "hello"),
    ]
