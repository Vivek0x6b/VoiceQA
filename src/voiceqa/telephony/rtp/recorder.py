"""Places received RTP payloads by timestamp into one continuous recording."""

from collections.abc import Callable

import numpy as np

from voiceqa.telephony.media import SAMPLE_RATE
from voiceqa.telephony.rtp.packet import RtpPacket, timestamp_delta

# Packets further than this outside the recorded span are treated as bogus,
# so one bad timestamp cannot make audio() allocate gigabytes.
MAX_JUMP_SAMPLES = 300 * SAMPLE_RATE


class Recorder:
    def __init__(self, decode: Callable[[bytes], np.ndarray]) -> None:
        self._decode = decode
        self._frames: dict[int, np.ndarray] = {}
        self._ssrc: int | None = None
        self._anchor_ts = 0
        self._anchor_pos = 0
        self._start = 0
        self._end = 0

    def add(self, packet: RtpPacket) -> bool:
        if packet.ssrc != self._ssrc:
            # A new stream (first packet, or the far end switched source)
            # continues from the end of what we have so far.
            self._ssrc = packet.ssrc
            self._anchor_ts = packet.timestamp
            self._anchor_pos = self._end
        pos = self._anchor_pos + timestamp_delta(packet.timestamp, self._anchor_ts)
        if self._frames and not (
            self._start - MAX_JUMP_SAMPLES <= pos <= self._end + MAX_JUMP_SAMPLES
        ):
            return False
        if pos in self._frames:
            return False

        samples = self._decode(packet.payload)
        if not self._frames:
            self._start, self._end = pos, pos + len(samples)
        else:
            self._start = min(self._start, pos)
            self._end = max(self._end, pos + len(samples))
        self._frames[pos] = samples
        return True

    def audio(self) -> np.ndarray:
        out = np.zeros(self._end - self._start, dtype=np.int16)
        for pos, samples in self._frames.items():
            out[pos - self._start : pos - self._start + len(samples)] = samples
        return out
