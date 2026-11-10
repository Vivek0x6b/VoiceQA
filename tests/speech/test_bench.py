import numpy as np
import pytest

from voiceqa.audio.wav import read_wav
from voiceqa.speech.bench import (
    BenchClip,
    Utterance,
    default_clip_dir,
    load_bench_clip,
    run_stt_bench,
)
from voiceqa.speech.models import Transcript


class CountingStt:
    def __init__(self):
        self.lengths = []

    def transcribe(self, samples, rate):
        self.lengths.append(len(samples))
        return Transcript(engine="counting", model="none", language="en")


class StepClock:
    """Advances by a scripted amount on every call, so timings are exact."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.now = 0.0

    def __call__(self):
        value = self.now
        if self._steps:
            self.now += self._steps.pop(0)
        return value


def make_clip(seconds=10.0, rate=8000):
    samples = np.zeros(int(seconds * rate), dtype=np.int16)
    utterances = [
        Utterance(start=0.5, end=2.5, text="one"),
        Utterance(start=3.0, end=4.0, text="two"),
        Utterance(start=5.0, end=9.0, text="three"),
    ]
    return BenchClip(samples=samples, rate=rate, utterances=utterances)


def test_bench_runs_one_warm_up_then_the_clip_then_each_utterance():
    clip = make_clip()
    engine = CountingStt()

    run_stt_bench(lambda: engine, clip, engine_name="counting", model="none")

    assert engine.lengths == [80000, 80000, 16000, 8000, 32000]


def test_bench_timings_come_from_the_clock():
    clip = make_clip(seconds=10.0)
    # clock calls: load start/end, warm-up start/end, clip start/end, then
    # start/end for each of the three utterances.
    clock = StepClock([1.5, 0, 9.0, 0, 2.0, 0, 0.2, 0, 0.1, 0, 0.6, 0])
    engine = CountingStt()

    result = run_stt_bench(lambda: engine, clip, engine_name="counting", model="none", clock=clock)

    assert result.load_seconds == pytest.approx(1.5)
    assert result.audio_seconds == pytest.approx(10.0)
    assert result.clip_seconds == pytest.approx(2.0)
    assert result.rtf == pytest.approx(0.2)
    assert result.utterances == 3
    assert result.utterance_median_ms == pytest.approx(200.0)
    assert result.utterance_max_ms == pytest.approx(600.0)
    assert result.utterance_p95_ms == pytest.approx(560.0)


def test_bench_result_round_trips_through_json():
    clip = make_clip()
    result = run_stt_bench(CountingStt, clip, engine_name="counting", model="none")
    assert type(result).model_validate_json(result.model_dump_json()) == result


def test_shipped_clip_loads_with_its_utterances():
    clip = load_bench_clip(default_clip_dir())
    assert clip.rate == 8000
    assert clip.samples.dtype == np.int16
    assert clip.samples.ndim == 1
    assert 10.0 <= len(clip.samples) / clip.rate <= 20.0
    assert len(clip.utterances) >= 3


def test_fixture_utterances_are_ordered_and_inside_the_clip():
    clip = load_bench_clip(default_clip_dir())
    duration = len(clip.samples) / clip.rate
    previous_end = 0.0
    for u in clip.utterances:
        assert previous_end <= u.start < u.end <= duration
        assert u.text.strip()
        previous_end = u.end


def test_fixture_utterances_are_louder_than_the_gaps_between_them():
    clip = load_bench_clip(default_clip_dir())
    x = clip.samples.astype(np.float64)

    def rms(a, b):
        seg = x[int(a * clip.rate) : int(b * clip.rate)]
        return float(np.sqrt(np.mean(seg**2)))

    for first, second in zip(clip.utterances, clip.utterances[1:], strict=False):
        gap = rms(first.end, second.start)
        assert rms(first.start, first.end) > 10 * gap
        assert rms(second.start, second.end) > 10 * gap


def test_fixture_wav_is_small_8k_mono():
    path = default_clip_dir() / "magi_8k.wav"
    samples, rate = read_wav(path)
    assert rate == 8000
    assert samples.ndim == 1
    assert path.stat().st_size < 400_000
