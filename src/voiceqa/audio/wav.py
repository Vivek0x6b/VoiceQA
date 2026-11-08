"""16-bit PCM WAV read and write.

Mono audio is a 1-D int16 array. Multichannel audio is (frames, channels).
"""

import os
import wave
from collections.abc import Iterator

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


def read_wav_blocks(path: str | os.PathLike, block_frames: int) -> Iterator[np.ndarray]:
    """Read a 16-bit WAV a block at a time, so long files need not fit in memory."""
    with wave.open(os.fspath(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM is supported")
        channels = w.getnchannels()
        while raw := w.readframes(block_frames):
            samples = np.frombuffer(raw, dtype="<i2").astype(np.int16)
            yield samples.reshape(-1, channels) if channels > 1 else samples


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


class WavWriter:
    """Writes a 16-bit WAV as audio comes in.

    The header is updated after every write, so the file on disk is a
    complete WAV at any moment, even if the process dies.
    """

    def __init__(self, path: str | os.PathLike, sample_rate: int, channels: int = 1) -> None:
        self._file = open(path, "wb")  # noqa: SIM115
        try:
            self._wav = wave.open(self._file, "wb")
            self._wav.setnchannels(channels)
            self._wav.setsampwidth(2)
            self._wav.setframerate(sample_rate)
            self._wav.writeframes(b"")
            self._file.flush()
        except BaseException:
            self._file.close()
            raise
        self.channels = channels
        self.frames = 0
        self._closed = False

    def __enter__(self) -> "WavWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples)
        if samples.dtype != np.int16:
            raise TypeError(f"WavWriter needs int16 samples, got {samples.dtype}")
        expected = () if self.channels == 1 else (self.channels,)
        if samples.ndim == 0 or samples.shape[1:] != expected:
            raise ValueError(f"expected {self.channels} channel(s), got shape {samples.shape}")
        # writeframes seeks back to fix the header, which also flushes the data.
        self._wav.writeframes(samples.astype("<i2").tobytes())
        self.frames += len(samples)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._wav.close()
        finally:
            self._file.close()
