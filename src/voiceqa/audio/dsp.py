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


def generate_tone(
    freq_hz: float, duration_s: float, sample_rate: int = 8000, amplitude: float = 0.5
) -> np.ndarray:
    n = round(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    return np.round(amplitude * 32767 * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)
