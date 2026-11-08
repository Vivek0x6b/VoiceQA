import json
import os
import tracemalloc

import numpy as np
import pytest

from voiceqa.audio.wav import read_wav
from voiceqa.telephony.media import FRAME_SAMPLES, SAMPLE_RATE
from voiceqa.telephony.recording import REORDER_SAMPLES, CallRecorder, TrackWriter


def frame(value: int) -> np.ndarray:
    return np.full(FRAME_SAMPLES, value, dtype=np.int16)


def track(path) -> np.ndarray:
    samples, rate = read_wav(path)
    assert rate == SAMPLE_RATE
    return samples


def test_track_in_order(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    for i, v in enumerate([100, 200, 300]):
        w.add(i * FRAME_SAMPLES, frame(v))
    assert w.close() == 3 * FRAME_SAMPLES
    np.testing.assert_array_equal(
        track(tmp_path / "t.wav"), np.concatenate([frame(100), frame(200), frame(300)])
    )
    assert (w.frames, w.dropped) == (3, 0)


def test_track_fills_gaps_with_silence(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    w.add(1000, frame(100))
    w.add(1000 + FRAME_SAMPLES + 333, frame(200))
    w.close()
    got = track(tmp_path / "t.wav")
    assert len(got) == 1000 + 2 * FRAME_SAMPLES + 333
    assert not got[:1000].any()
    np.testing.assert_array_equal(got[1000:1160], frame(100))
    assert not got[1160:1493].any()
    np.testing.assert_array_equal(got[1493:], frame(200))


def test_track_snaps_small_offsets_instead_of_cutting_or_padding(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    w.add(0, frame(100))
    w.add(FRAME_SAMPLES + 79, frame(200))
    w.add(2 * FRAME_SAMPLES - 79, frame(300))
    w.close()
    np.testing.assert_array_equal(
        track(tmp_path / "t.wav"), np.concatenate([frame(100), frame(200), frame(300)])
    )
    assert w.dropped == 0


def test_track_reorders_within_the_window_and_drops_what_comes_later(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    w.add(320, frame(300))
    w.add(0, frame(100))
    w.add(160, frame(200))
    w.add(160, frame(-900))  # duplicate
    # Far enough ahead that 0..320 are committed, then a frame for 0 again.
    w.add(320 + REORDER_SAMPLES + 160, frame(500))
    w.add(0, frame(-900))
    w.close()
    got = track(tmp_path / "t.wav")
    np.testing.assert_array_equal(got[:480], np.concatenate([frame(100), frame(200), frame(300)]))
    assert w.dropped == 2
    assert w.frames == 4


def test_track_pads_to_the_requested_length(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    w.add(0, frame(100))
    w.flush()
    assert w.length == FRAME_SAMPLES
    assert w.close(length=8000) == 8000
    assert len(track(tmp_path / "t.wav")) == 8000
    assert w.close() == 8000


def test_track_streams_to_disk_as_frames_arrive(tmp_path):
    path = tmp_path / "t.wav"
    w = TrackWriter(path)
    try:
        for i in range(1000):
            w.add(i * FRAME_SAMPLES, frame(i))
        committed = w.length
        assert committed >= 1000 * FRAME_SAMPLES - REORDER_SAMPLES
        assert len(track(path)) == committed
    finally:
        w.close()


TEN_MINUTES = 10 * 60 * 1000 // 20


def test_track_memory_stays_flat_over_ten_minutes(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    f = frame(1234)
    tracemalloc.start()
    try:
        for i in range(TEN_MINUTES):
            w.add(i * FRAME_SAMPLES, f.copy())
            if i == TEN_MINUTES // 10:
                early, _ = tracemalloc.get_traced_memory()
        late, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        w.close()
    assert late - early < 64 * 1024
    assert os.path.getsize(tmp_path / "t.wav") == 44 + 2 * TEN_MINUTES * FRAME_SAMPLES


def test_track_long_gap_is_written_in_pieces(tmp_path):
    w = TrackWriter(tmp_path / "t.wav")
    w.add(0, frame(1))
    tracemalloc.start()
    try:
        w.add(600 * SAMPLE_RATE, frame(2))
        w.flush()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        w.close()
    assert peak < 256 * 1024
    assert w.length == 600 * SAMPLE_RATE + FRAME_SAMPLES


class Clock:
    def __init__(self, t: float = 100.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_call_recorder_aligns_both_tracks_and_events(tmp_path):
    clock = Clock()
    rec = CallRecorder(tmp_path / "run", clock=clock)
    rec.event("invite_sent")
    clock.t = 100.2
    rec.event("ringing", status=180)
    clock.t = 100.5
    rec.event("answered")
    for i in range(5):
        rec.sent(101.0 + 0.02 * i, frame(1000))
    for i in range(5):
        rec.received(101.25 + 0.02 * i, 101.27 + 0.02 * i, frame(-2000))
    clock.t = 102.0
    rec.event("bye_sent")
    rec.close()

    caller = track(tmp_path / "run" / "caller.wav")
    agent = track(tmp_path / "run" / "agent.wav")
    mix = track(tmp_path / "run" / "mix.wav")
    assert len(caller) == len(agent) == 2 * SAMPLE_RATE
    assert mix.shape == (2 * SAMPLE_RATE, 2)
    np.testing.assert_array_equal(mix[:, 0], caller)
    np.testing.assert_array_equal(mix[:, 1], agent)
    assert np.flatnonzero(caller)[[0, -1]].tolist() == [8000, 8000 + 800 - 1]
    assert np.flatnonzero(agent)[[0, -1]].tolist() == [10000, 10000 + 800 - 1]

    data = json.loads((tmp_path / "run" / "events.json").read_text())
    assert data["sample_rate"] == SAMPLE_RATE
    assert data["duration"] == pytest.approx(2.0)
    assert data["started_at"].endswith("Z")
    events = [(e["event"], e["t"]) for e in data["events"]]
    assert events == [
        ("invite_sent", 0.0),
        ("ringing", pytest.approx(0.2)),
        ("answered", pytest.approx(0.5)),
        ("first_rtp_out", pytest.approx(1.0)),
        ("first_rtp_in", pytest.approx(1.27)),
        ("bye_sent", pytest.approx(2.0)),
    ]
    assert data["events"][1]["status"] == 180
    assert data["tracks"]["caller"] == {"file": "caller.wav", "frames": 5, "dropped": 0}
    assert data["tracks"]["agent"] == {"file": "agent.wav", "frames": 5, "dropped": 0}


def test_call_recorder_pads_to_the_furthest_frame(tmp_path):
    clock = Clock(0.0)
    rec = CallRecorder(tmp_path, clock=clock)
    clock.t = 0.5
    # Placed by RTP timestamp slightly after the moment the call is closed.
    rec.received(0.6, 0.5, frame(7))
    rec.close()
    assert len(track(tmp_path / "caller.wav")) == len(track(tmp_path / "agent.wav")) == 4960


def test_call_recorder_close_twice_and_ignore_late_frames(tmp_path):
    rec = CallRecorder(tmp_path, clock=Clock(0.0))
    rec.close()
    rec.sent(0.0, frame(1))
    rec.received(0.0, 0.0, frame(1))
    rec.event("bye_received")
    rec.close()
    assert len(track(tmp_path / "caller.wav")) == 0
    events = json.loads((tmp_path / "events.json").read_text())["events"]
    assert events == []


def test_call_recorder_as_context_manager(tmp_path):
    with CallRecorder(tmp_path, clock=Clock(0.0)) as rec:
        rec.sent(0.0, frame(1))
    assert len(track(tmp_path / "mix.wav")) == FRAME_SAMPLES
