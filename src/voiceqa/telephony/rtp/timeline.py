"""Maps received RTP timestamps onto the local clock.

The first packet of a stream is placed at its arrival time. Later packets are
placed by their RTP timestamp relative to it, so network jitter and
reordering do not move audio around.

A packet whose timestamp says it belongs more than max_late before its
arrival, or more than max_early after it, is placed at its arrival time
instead. That happens in two very different cases:

- A burst: our event loop or the network held packets back and they all
  arrive at once. The disagreement goes away with the next packets, so the
  anchor stays.
- A timestamp jump: the sender paused without advancing its timestamp,
  restarted its clock, or drifted over a long call. The disagreement
  persists, so after reanchor_after seconds of it the stream is anchored
  again at the current packet.

A new SSRC is anchored at once.
"""

from collections import deque

from voiceqa.telephony.media import SAMPLE_RATE
from voiceqa.telephony.rtp.packet import RtpPacket, timestamp_delta

MAX_LATE = 0.5
MAX_EARLY = 0.1
REANCHOR_AFTER = 0.5
DUPLICATE_HISTORY = 64


class ReceiveTimeline:
    def __init__(
        self,
        max_late: float = MAX_LATE,
        max_early: float = MAX_EARLY,
        reanchor_after: float = REANCHOR_AFTER,
        history: int = DUPLICATE_HISTORY,
    ) -> None:
        self._max_late = max_late
        self._max_early = max_early
        self._reanchor_after = reanchor_after
        self._ssrc: int | None = None
        self._anchor_ts = 0
        self._anchor_time = 0.0
        self._off_since: float | None = None
        self._recent: deque[tuple[int, int]] = deque(maxlen=history)
        self._recent_set: set[tuple[int, int]] = set()

    def place(self, packet: RtpPacket, arrived: float) -> float | None:
        """Start time of the packet's audio on the local clock, or None for a duplicate."""
        key = (packet.ssrc, packet.timestamp)
        if key in self._recent_set:
            return None
        if len(self._recent) == self._recent.maxlen:
            self._recent_set.discard(self._recent[0])
        self._recent.append(key)
        self._recent_set.add(key)

        if packet.ssrc != self._ssrc:
            self._anchor(packet, arrived)
            return arrived
        at = self._anchor_time + timestamp_delta(packet.timestamp, self._anchor_ts) / SAMPLE_RATE
        if -self._max_late <= at - arrived <= self._max_early:
            self._off_since = None
            return at
        if self._off_since is None:
            self._off_since = arrived
        elif arrived - self._off_since >= self._reanchor_after:
            self._anchor(packet, arrived)
        return arrived

    def _anchor(self, packet: RtpPacket, arrived: float) -> None:
        self._ssrc = packet.ssrc
        self._anchor_ts = packet.timestamp
        self._anchor_time = arrived
        self._off_since = None
