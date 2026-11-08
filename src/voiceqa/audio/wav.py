"""16-bit PCM WAV read and write.

Mono audio is a 1-D int16 array. Multichannel audio is (frames, channels).
"""

import os
import wave

import numpy as np


def read_wav(path: str | os.PathLike) -> tuple[np.ndarray, int]:
    try:
        with wave.open(os.fspath(path), "rb") as w:
            channels = w.getnchannels()
            width = w.getsampwidth()
            rate = w.getframerate()
            raw = w.readframes(w.getnframes())
    except wave.Error as e:
        raise ValueError(f"{path}: not a PCM WAV file ({e})") from e
    if width != 2:
        raise ValueError(f"{path}: only 16-bit PCM is supported, got {8 * width}-bit")
    samples = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if channels > 1:
        samples = samples.reshape(-1, channels)
    return samples, rate


def write_wav(path: str | os.PathLike, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples)
    if samples.dtype != np.int16:
        raise TypeError(f"write_wav needs int16 samples, got {samples.dtype}")
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    with wave.open(os.fspath(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(samples.astype("<i2").tobytes())
