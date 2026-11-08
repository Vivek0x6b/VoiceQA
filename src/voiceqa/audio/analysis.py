"""Scoring a recorded call: finding where audio actually starts."""

import os

import numpy as np

from voiceqa.audio.wav import read_wav_blocks
from voiceqa.telephony.media import FRAME_SAMPLES

DEFAULT_BLOCK_FRAMES = 8000


def first_loud_frame(
    path: str | os.PathLike,
    threshold_rms: float = 328,
    block_frames: int = DEFAULT_BLOCK_FRAMES,
) -> int | None:
    """Sample index of the first 20 ms frame in path whose RMS exceeds threshold_rms.

    Reads the WAV in blocks of block_frames samples, so a long recording
    never has to fit in memory. A frame that straddles two blocks is
    reassembled before its RMS is checked. A trailing partial frame at the
    end of the file is ignored. Returns None if no frame is loud enough.
    """
    carry = np.zeros(0, dtype=np.int16)
    pos = 0  # sample index where `carry` begins
    for block in read_wav_blocks(path, block_frames):
        data = np.concatenate([carry, block]) if carry.size else block
        whole = len(data) // FRAME_SAMPLES
        if whole:
            frames = data[: whole * FRAME_SAMPLES].reshape(whole, FRAME_SAMPLES)
            rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
            hit = np.flatnonzero(rms > threshold_rms)
            if hit.size:
                return pos + int(hit[0]) * FRAME_SAMPLES
            pos += whole * FRAME_SAMPLES
        carry = data[whole * FRAME_SAMPLES :]
    return None
