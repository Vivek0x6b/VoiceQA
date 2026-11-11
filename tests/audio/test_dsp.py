import time

import numpy as np
import pytest

from voiceqa.audio.dsp import StreamResampler, generate_tone, resample, to_mono


def peak_hz(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return float(np.fft.rfftfreq(len(samples), 1 / rate)[np.argmax(spectrum)])


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def test_to_mono_averages():
    stereo = np.array([[100, 300], [-100, -300], [32767, 32767]], dtype=np.int16)
    out = to_mono(stereo)
    assert out.dtype == np.int16
    assert out.tolist() == [200, -200, 32767]


def test_to_mono_passthrough_1d():
    mono = np.array([1, 2, 3], dtype=np.int16)
    out = to_mono(mono)
    np.testing.assert_array_equal(out, mono)
    assert out is not mono


def test_tone_length_and_peak():
    tone = generate_tone(1000, 0.5)
    assert tone.dtype == np.int16
    assert len(tone) == 4000
    assert abs(int(np.abs(tone).max()) - round(0.5 * 32767)) <= 1
    assert peak_hz(tone, 8000) == pytest.approx(1000, abs=2)


def test_resample_48k_to_8k_keeps_tone():
    tone = generate_tone(1000, 1.0, sample_rate=48000)
    out = resample(tone, 48000, 8000)
    assert out.dtype == np.int16
    assert len(out) == 8000
    assert peak_hz(out, 8000) == pytest.approx(1000, abs=2)
    assert rms(out[100:-100]) == pytest.approx(rms(tone), rel=0.03)


def test_resample_removes_alias():
    # Without a low-pass filter a 6 kHz tone would fold down to 2 kHz.
    tone = generate_tone(6000, 1.0, sample_rate=48000)
    out = resample(tone, 48000, 8000)
    assert rms(out[100:-100]) < 0.02 * rms(tone)


def test_resample_44k1_to_8k_length():
    out = resample(np.zeros(44100, dtype=np.int16), 44100, 8000)
    assert len(out) == 8000


def test_resample_upsample():
    tone = generate_tone(440, 1.0, sample_rate=8000)
    out = resample(tone, 8000, 16000)
    assert len(out) == 16000
    assert peak_hz(out, 16000) == pytest.approx(440, abs=2)


def test_resample_same_rate_is_copy():
    x = np.array([1, 2, 3], dtype=np.int16)
    out = resample(x, 8000, 8000)
    np.testing.assert_array_equal(out, x)
    assert out is not x


def test_resample_empty():
    out = resample(np.zeros(0, dtype=np.int16), 48000, 8000)
    assert out.shape == (0,)
    assert out.dtype == np.int16


def test_resample_clips_overshoot():
    # A full-scale square wave rings past full scale after filtering.
    # The output must clip, not wrap around to the opposite sign.
    t = np.arange(48000)
    square = np.where((t // 48) % 2 == 0, 32767, -32767).astype(np.int16)
    out = resample(square, 48000, 8000).astype(np.int32)
    flips = (
        (np.abs(out[1:-1]) > 30000)
        & (np.sign(out[1:-1]) != np.sign(out[:-2]))
        & (np.sign(out[1:-1]) != np.sign(out[2:]))
    )
    assert not flips.any()
    assert out.max() == 32767


def stream_in_frames(stream: StreamResampler, x: np.ndarray, frame: int) -> np.ndarray:
    return np.concatenate([stream.push(x[i : i + frame]) for i in range(0, len(x), frame)])


@pytest.mark.parametrize("frame", [160, 1, 37, 1000])
def test_stream_resampler_matches_one_pass(frame):
    rng = np.random.default_rng(3)
    x = (generate_tone(440, 2.0, 8000, 0.3) + rng.integers(-3000, 3000, 16000)).astype(np.int16)
    whole = resample(x, 8000, 16000)
    streamed = stream_in_frames(StreamResampler(8000, 16000), x, frame)
    assert streamed.dtype == np.int16
    # Only the last few outputs wait for input that has not arrived yet.
    assert len(whole) - 64 <= len(streamed) <= len(whole)
    diff = np.abs(streamed.astype(np.int32) - whole[: len(streamed)].astype(np.int32))
    assert diff.max() <= 1


def test_stream_resampler_needs_an_integer_factor():
    with pytest.raises(ValueError, match="integer"):
        StreamResampler(16000, 8000)


def test_resample_60s_48k_under_1s():
    rng = np.random.default_rng(0)
    x = rng.integers(-10000, 10000, size=60 * 48000).astype(np.int16)
    start = time.perf_counter()
    out = resample(x, 48000, 8000)
    elapsed = time.perf_counter() - start
    assert len(out) == 60 * 8000
    assert elapsed < 1.0, f"took {elapsed:.2f}s"
