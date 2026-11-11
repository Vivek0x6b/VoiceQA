"""Downmix, resampling and test signal generation for int16 audio."""

from math import ceil, gcd

import numpy as np

# Zero crossings of the sinc kernel on each side, in units of the cutoff period.
_ZERO_CROSSINGS = 16
_CUTOFF = 0.95
_CHUNK_ROWS = 8192


def to_mono(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples)
    if x.ndim == 1:
        return x.copy()
    mean = x.astype(np.float64).mean(axis=1)
    return np.clip(np.round(mean), -32768, 32767).astype(np.int16)


def _kernel_table(up: int, down: int) -> tuple[np.ndarray, int]:
    """Hann-windowed sinc taps, one row per fractional phase k / up."""
    cutoff = _CUTOFF * min(1.0, up / down)
    half = ceil(_ZERO_CROSSINGS / cutoff)
    offsets = np.arange(-half + 1, half + 1)
    d = offsets[None, :] - (np.arange(up) / up)[:, None]
    taps = cutoff * np.sinc(cutoff * d) * (0.5 + 0.5 * np.cos(np.pi * d / half))
    taps /= taps.sum(axis=1, keepdims=True)
    return taps.astype(np.float32), half


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    x = np.asarray(samples)
    if src_rate == dst_rate:
        return x.copy()
    g = gcd(src_rate, dst_rate)
    up, down = dst_rate // g, src_rate // g
    n_out = (len(x) * up + down // 2) // down
    out = np.zeros(n_out, dtype=np.float32)
    if n_out == 0:
        return out.astype(np.int16)

    taps, half = _kernel_table(up, down)
    width = taps.shape[1]
    padded = np.pad(x.astype(np.float32), (half, half + 1))
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)

    # Output n reads the window starting at (n * down) // up with phase
    # (n * down) % up. Outputs n = r, r + up, r + 2 * up, ... share a phase and
    # their windows are `down` samples apart, so each group is one strided
    # view times one kernel row.
    for r in range(min(up, n_out)):
        phase = (r * down) % up
        first = (r * down) // up
        count = len(range(r, n_out, up))
        for j in range(0, count, _CHUNK_ROWS):
            rows = min(_CHUNK_ROWS, count - j)
            start = first + j * down
            block = windows[start : start + rows * down : down]
            out[r + j * up : r + (j + rows) * up : up] = block @ taps[phase]

    return np.clip(np.round(out), -32768, 32767).astype(np.int16)


class StreamResampler:
    """`resample` for audio that arrives in pieces, by an integer factor.

    Resampling each piece on its own would put filter edge effects at every
    boundary. This keeps enough past input to run `resample` over the
    boundary and returns only outputs whose whole filter window has
    arrived, so the concatenated output matches one pass over the joined
    input (within 1 LSB of float rounding). The last (half - 1) * factor
    outputs, about 2 ms, wait for the next push.
    """

    def __init__(self, src_rate: int, dst_rate: int) -> None:
        if dst_rate % src_rate:
            raise ValueError(
                f"StreamResampler needs an integer factor, got {src_rate} to {dst_rate} Hz"
            )
        self._src = src_rate
        self._dst = dst_rate
        self._up = dst_rate // src_rate
        _, self._half = _kernel_table(self._up, 1)
        # Input samples still needed, starting at input index self._first.
        self._kept = np.zeros(0, dtype=np.int16)
        self._first = 0
        self._next_out = 0

    def push(self, samples: np.ndarray) -> np.ndarray:
        self._kept = np.concatenate([self._kept, np.asarray(samples, dtype=np.int16)])
        end = self._first + len(self._kept)
        # Output n reads input samples n // up - half up to n // up + half - 1.
        stop = max(self._next_out, (end - self._half + 1) * self._up)
        out = resample(self._kept, self._src, self._dst)
        offset = self._first * self._up
        ready = out[self._next_out - offset : stop - offset]
        self._next_out = stop
        drop = stop // self._up - self._half - self._first
        if drop > 0:
            self._kept = self._kept[drop:]
            self._first += drop
        return ready


def generate_tone(
    freq_hz: float, duration_s: float, sample_rate: int = 8000, amplitude: float = 0.5
) -> np.ndarray:
    n = round(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    return np.round(amplitude * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
