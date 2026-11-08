"""Recording both sides of a call on one clock, streamed to disk.

A CallRecorder writes into one directory:

- caller.wav: what we sent, as it went on the wire.
- agent.wav: what we received.
- mix.wav: both as stereo, caller left and agent right.
- events.json: call events (INVITE sent, ringing, answered, first RTP in
  and out, BYE) in seconds since the recorder was created.

All of them share time zero, so an event at t seconds lines up with sample
t * 8000 of every file. Gaps are filled with silence so the tracks stay
aligned, and both tracks are padded to the same length when the call ends.
"""

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from voiceqa.audio.wav import WavWriter, read_wav_blocks
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE, media_clock

# Frames wait this long before they are written, so reordered packets still
# land in order.
REORDER_SAMPLES = 5 * FRAME_SAMPLES
# A frame that starts within this distance of the end of the track is
# appended there. RTP timestamps on two clocks rarely line up to the sample.
SNAP_SAMPLES = FRAME_SAMPLES // 2
SILENCE_BLOCK = SAMPLE_RATE


class TrackWriter:
    """Streams one mono track to a WAV file, placing frames by sample position.

    A frame that starts after the end of the track is preceded by silence.
    One that starts before it (late, duplicate, or overlapping after the far
    end jumped back) is dropped and counted.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._wav = WavWriter(path, SAMPLE_RATE)
        self._pending: dict[int, np.ndarray] = {}
        self._newest = 0
        self.length = 0
        self.frames = 0
        self.dropped = 0
        self._closed = False

    def add(self, pos: int, samples: np.ndarray) -> None:
        if self._closed:
            return
        if pos in self._pending or pos <= self.length - SNAP_SAMPLES:
            self.dropped += 1
            return
        self._pending[pos] = samples
        self._newest = max(self._newest, pos)
        self._commit(self._newest - REORDER_SAMPLES)

    def flush(self) -> None:
        """Write every frame still waiting for reordered packets."""
        self._commit(None)

    def close(self, length: int | None = None) -> int:
        """Flush, pad with silence up to length samples, and close. Returns the final length."""
        if self._closed:
            return self.length
        self.flush()
        if length is not None and length > self.length:
            self._silence(length - self.length)
        self._closed = True
        self._wav.close()
        return self.length

    def _commit(self, upto: int | None) -> None:
        ready = sorted(p for p in self._pending if upto is None or p <= upto)
        for pos in ready:
            self._place(pos, self._pending.pop(pos))

    def _place(self, pos: int, samples: np.ndarray) -> None:
        offset = pos - self.length
        if offset <= -SNAP_SAMPLES:
            self.dropped += 1
            return
        if offset >= SNAP_SAMPLES:
            self._silence(offset)
        self._wav.write(samples)
        self.length += len(samples)
        self.frames += 1

    def _silence(self, n: int) -> None:
        while n > 0:
            block = min(n, SILENCE_BLOCK)
            self._wav.write(np.zeros(block, dtype=np.int16))
            self.length += block
            n -= block


class CallRecorder:
    """Records one call. Pass it to MediaSession.record() and to place_call(on_event=...).

    Create it before placing the call, so that time zero is just before the
    INVITE. clock must be the one the frames are timed with (media_clock);
    tests pass a fake.
    """

    def __init__(self, directory: str | os.PathLike, clock: Callable[[], float] = media_clock):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._t0 = self._clock()
        self._started_at = datetime.now(UTC)
        self._caller = TrackWriter(self.directory / "caller.wav")
        try:
            self._agent = TrackWriter(self.directory / "agent.wav")
        except BaseException:
            self._caller.close()
            raise
        self._events: list[dict] = []
        self._first_out = True
        self._first_in = True
        self._closed = False

    def __enter__(self) -> "CallRecorder":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def event(self, name: str, at: float | None = None, **detail: object) -> None:
        if self._closed:
            return
        t = (self._clock() if at is None else at) - self._t0
        self._events.append({"t": round(t, 6), "event": name, **detail})

    def sent(self, at: float, samples: np.ndarray) -> None:
        if self._closed:
            return
        if self._first_out:
            self._first_out = False
            self.event("first_rtp_out", at=at)
        self._caller.add(self._pos(at), samples)

    def received(self, at: float, arrived: float, samples: np.ndarray) -> None:
        if self._closed:
            return
        if self._first_in:
            self._first_in = False
            self.event("first_rtp_in", at=arrived)
        self._agent.add(self._pos(at), samples)

    def close(self) -> None:
        """Finish both tracks at the same length and write mix.wav and events.json."""
        if self._closed:
            return
        self._closed = True
        try:
            self._caller.flush()
            self._agent.flush()
        finally:
            length = max(self._pos(self._clock()), self._caller.length, self._agent.length)
            try:
                self._caller.close(length)
            finally:
                self._agent.close(length)
        self._write_mix()
        self._write_events(length)

    def _pos(self, at: float) -> int:
        return max(0, round((at - self._t0) * SAMPLE_RATE))

    def _write_mix(self) -> None:
        caller = read_wav_blocks(self.directory / "caller.wav", SAMPLE_RATE)
        agent = read_wav_blocks(self.directory / "agent.wav", SAMPLE_RATE)
        with WavWriter(self.directory / "mix.wav", SAMPLE_RATE, channels=2) as mix:
            for left, right in zip(caller, agent, strict=True):
                mix.write(np.stack([left, right], axis=1))

    def _write_events(self, length: int) -> None:
        def track(name: str, writer: TrackWriter) -> dict:
            return {"file": name, "frames": writer.frames, "dropped": writer.dropped}

        data = {
            "started_at": self._started_at.isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "sample_rate": SAMPLE_RATE,
            "duration": length / SAMPLE_RATE,
            "events": sorted(self._events, key=lambda e: e["t"]),
            "tracks": {
                "caller": track("caller.wav", self._caller),
                "agent": track("agent.wav", self._agent),
            },
        }
        path = self.directory / "events.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
