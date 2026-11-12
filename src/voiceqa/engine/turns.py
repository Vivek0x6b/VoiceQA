"""TurnTap: the MediaSink between the RTP session and the call recorder in
an interactive call.

It passes every frame on to the recorder unchanged, hands each received
(agent) frame to the turn detector, keeps the last few seconds of agent
audio so a finished turn can be cut out by its times, and times what we
say by the sender's slots. RTP and SIP stay unaware of speech: they only
see a MediaSink.

Everything here runs on the event loop thread, called by the RTP session.
"""

from collections import deque
from collections.abc import Callable

import numpy as np

from voiceqa.telephony.media import SAMPLE_RATE, MediaSink

# Agent audio kept while no turn is open. A speech_start reaches the loop
# after min_speech_ms plus the detector's queue (1 s at most), but names
# the start of the first speech frame, which must still be here.
PREROLL_SECONDS = 3.0


class TurnTap:
    def __init__(
        self,
        sink: MediaSink,
        on_agent_frame: Callable[[float, np.ndarray], object] | None = None,
        preroll: float = PREROLL_SECONDS,
    ) -> None:
        self._sink = sink
        self._on_agent_frame = on_agent_frame
        self._preroll = preroll
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._newest = float("-inf")
        self._hold: float | None = None
        self._keep = True
        self._say_left = 0
        self._say_sent = 0
        self._say_first: float | None = None
        self._say_last: float | None = None
        self._on_say_start: Callable[[float], object] | None = None
        self._on_say_end: Callable[[float, float], object] | None = None
        self._say_mark: int | None = None
        self._on_say_mark: Callable[[float], object] | None = None

    @property
    def saying(self) -> bool:
        return self._say_left > 0

    @property
    def buffered_frames(self) -> int:
        return len(self._frames)

    def start_say(
        self,
        frames: int,
        on_start: Callable[[float], object],
        on_end: Callable[[float, float], object],
        mark: int | None = None,
        on_mark: Callable[[float], object] | None = None,
    ) -> None:
        """Times the next `frames` sent frames, which must be the audio
        about to be queued. on_start gets the first frame's slot time,
        on_end the first and last, on_mark the slot of frame number `mark`
        (counting from 0). Call it right before queueing the audio, on the
        loop thread, so no other frame goes out in between.
        """
        if self.saying:
            raise RuntimeError("a say is still going out")
        if frames <= 0:
            raise ValueError(f"a say needs at least one frame, got {frames}")
        self._say_left = frames
        self._say_sent = 0
        self._say_first = self._say_last = None
        self._on_say_start, self._on_say_end = on_start, on_end
        self._say_mark, self._on_say_mark = mark, on_mark

    def abort_say(self) -> tuple[float, float] | None:
        """Stops timing the current say. Returns the slot times of its first
        and last frame sent so far, or None if none went out."""
        sent = None
        if self._say_first is not None and self._say_last is not None:
            sent = (self._say_first, self._say_last)
        self._say_left = 0
        self._say_first = self._say_last = None
        self._on_say_start = self._on_say_end = self._on_say_mark = None
        self._say_mark = None
        return sent

    def stop_turns(self) -> None:
        """Stop feeding the detector and keeping agent audio, for good (the
        VAD failed). Frames still go to the recorder."""
        self._keep = False
        self._hold = None
        self._frames.clear()

    def hold_from(self, at: float | None) -> None:
        """Keep agent audio from `at` on, whatever the preroll, until
        called with None."""
        self._hold = at

    def cut(self, start: float, end: float) -> np.ndarray:
        """Agent audio from start to end, placed by frame time, with
        silence where no frame was received or kept."""
        out = np.zeros(max(0, round((end - start) * SAMPLE_RATE)), dtype=np.int16)
        for at, samples in self._frames:
            pos = round((at - start) * SAMPLE_RATE)
            lo, hi = max(pos, 0), min(pos + len(samples), len(out))
            if lo < hi:
                out[lo:hi] = samples[lo - pos : hi - pos]
        return out

    def sent(self, at: float, samples: np.ndarray) -> None:
        self._sink.sent(at, samples)
        if not self._say_left:
            return
        if self._say_first is None:
            self._say_first = at
            if self._on_say_start is not None:
                self._on_say_start(at)
        self._say_last = at
        if self._say_sent == self._say_mark and self._on_say_mark is not None:
            self._on_say_mark(at)
        self._say_sent += 1
        self._say_left -= 1
        if not self._say_left:
            first, last, on_end = self._say_first, at, self._on_say_end
            self.abort_say()
            if on_end is not None:
                on_end(first, last)

    def received(self, at: float, arrived: float, samples: np.ndarray) -> None:
        self._sink.received(at, arrived, samples)
        if not self._keep:
            return
        if self._on_agent_frame is not None:
            self._on_agent_frame(at, samples)
        self._frames.append((at, samples))
        self._newest = max(self._newest, at)
        keep_from = self._newest - self._preroll
        if self._hold is not None:
            keep_from = min(keep_from, self._hold)
        while self._frames and self._frames[0][0] < keep_from:
            self._frames.popleft()
