"""Scoring a recorded call: finding where audio actually starts, and timing pulses."""

import os
from dataclasses import dataclass

import numpy as np

from voiceqa.audio.wav import read_wav_blocks
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE

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


@dataclass(frozen=True)
class PulseTiming:
    pulses_sent: int
    pulses_matched: int
    delays_ms: list[float]


def pulse_round_trip_delays(
    caller: np.ndarray,
    agent: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    threshold: float = 328,
    quiet_ms: float = 100.0,
    max_delay_ms: float = 400.0,
) -> PulseTiming:
    """Round-trip delay of each pulse in a caller/agent pulse-train recording.

    An onset is the first sample whose magnitude exceeds threshold after at
    least quiet_ms of quiet (magnitude at or below threshold); this finds
    every burst in a pulse train, not just the first one. Each caller onset
    is paired with the first agent onset that follows it within
    max_delay_ms. A caller onset with no such agent onset (a dropped echo)
    is left unmatched. An agent onset that falls outside every caller
    onset's window (a spurious click) is never matched to one, and does
    not throw off the pairing of the onsets after it.

    caller and agent are whole int16 arrays (decoded audio, not WAV paths):
    calibration recordings are short enough to read in one go.
    """
    quiet_samples = round(quiet_ms / 1000 * sample_rate)
    max_delay_samples = round(max_delay_ms / 1000 * sample_rate)
    caller_onsets = _find_onsets(caller, threshold, quiet_samples)
    agent_onsets = _find_onsets(agent, threshold, quiet_samples)
    delays_samples = _pair_onsets(caller_onsets, agent_onsets, max_delay_samples)
    delays_ms = [d * 1000 / sample_rate for d in delays_samples]
    return PulseTiming(len(caller_onsets), len(delays_samples), delays_ms)


def _find_onsets(samples: np.ndarray, threshold: float, quiet_samples: int) -> list[int]:
    loud = np.abs(np.asarray(samples, dtype=np.int64)) > threshold
    if not loud.any():
        return []
    padded = np.concatenate(([False], loud, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    onsets = []
    prev_end = -quiet_samples  # the start of the array counts as quiet
    for start, end in zip(starts, ends, strict=True):
        if start - prev_end >= quiet_samples:
            onsets.append(int(start))
        prev_end = int(end)
    return onsets


def _pair_onsets(caller_onsets: list[int], agent_onsets: list[int], max_gap: int) -> list[int]:
    """Delay (in samples) from each caller onset to the next agent onset within max_gap.

    Walks both onset lists once: for each caller onset, agent onsets at or
    before it (stale, or a click already used) are skipped, then the next
    one is taken as the echo if it is close enough. An onset consumed this
    way (matched or skipped as stale) is never considered again.
    """
    delays = []
    j = 0
    for c in caller_onsets:
        while j < len(agent_onsets) and agent_onsets[j] <= c:
            j += 1
        if j < len(agent_onsets) and agent_onsets[j] - c <= max_gap:
            delays.append(agent_onsets[j] - c)
            j += 1
    return delays
