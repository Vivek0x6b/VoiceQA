import numpy as np
import pytest

from voiceqa.engine.turns import TurnTap

RATE = 8000
FRAME = 160
FRAME_S = FRAME / RATE
# A media_clock-like start time, so a bug that times things from zero shows.
T0 = 5000.0


class Sink:
    def __init__(self):
        self.sent_frames = []
        self.received_frames = []

    def sent(self, at, samples):
        self.sent_frames.append((at, samples))

    def received(self, at, arrived, samples):
        self.received_frames.append((at, arrived, samples))


def frame(value):
    return np.full(FRAME, value, dtype=np.int16)


def test_tap_passes_every_frame_on_and_feeds_only_agent_frames():
    sink, fed = Sink(), []
    tap = TurnTap(sink, on_agent_frame=lambda at, s: fed.append((at, s)))
    tap.sent(T0, frame(1))
    tap.received(T0 + 0.5, T0 + 0.51, frame(2))
    assert [at for at, _ in sink.sent_frames] == [T0]
    assert [(at, arrived) for at, arrived, _ in sink.received_frames] == [(T0 + 0.5, T0 + 0.51)]
    assert len(fed) == 1 and fed[0][0] == T0 + 0.5 and (fed[0][1] == 2).all()


def test_tap_times_a_say_by_the_slots_of_its_first_and_last_frame():
    tap = TurnTap(Sink())
    starts, ends = [], []
    tap.sent(T0, frame(0))
    tap.start_say(3, starts.append, lambda first, last: ends.append((first, last)))
    assert tap.saying
    for i in range(1, 6):
        tap.sent(T0 + i * FRAME_S, frame(0))
    assert starts == [T0 + FRAME_S]
    assert ends == [(T0 + FRAME_S, T0 + 3 * FRAME_S)]
    assert not tap.saying


def test_tap_say_uses_the_slots_actually_sent_after_skipped_ones():
    tap = TurnTap(Sink())
    ends = []
    tap.start_say(2, lambda at: None, lambda first, last: ends.append((first, last)))
    tap.sent(T0, frame(0))
    # The sender skipped ten slots after a stall.
    tap.sent(T0 + 11 * FRAME_S, frame(0))
    assert ends == [(T0, T0 + 11 * FRAME_S)]


def test_tap_refuses_a_second_say_while_one_is_going_out():
    tap = TurnTap(Sink())
    tap.start_say(2, lambda at: None, lambda first, last: None)
    with pytest.raises(RuntimeError):
        tap.start_say(1, lambda at: None, lambda first, last: None)


def test_tap_abort_say_reports_what_went_out_and_stops_the_callbacks():
    tap = TurnTap(Sink())
    ends = []
    tap.start_say(5, lambda at: None, lambda first, last: ends.append((first, last)))
    tap.sent(T0, frame(0))
    tap.sent(T0 + FRAME_S, frame(0))
    assert tap.abort_say() == (T0, T0 + FRAME_S)
    for i in range(2, 8):
        tap.sent(T0 + i * FRAME_S, frame(0))
    assert ends == []
    assert not tap.saying
    assert tap.abort_say() is None


def test_tap_abort_say_before_any_frame_went_out():
    tap = TurnTap(Sink())
    tap.start_say(5, lambda at: None, lambda first, last: None)
    assert tap.abort_say() is None


def test_tap_cut_places_frames_by_time_with_silence_for_gaps():
    tap = TurnTap(Sink())
    # Frames 0, 1, 3 arrive, frame 2 is lost, and 4 comes before 3.
    for i in (0, 1, 4, 3):
        tap.received(T0 + i * FRAME_S, T0 + i * FRAME_S, frame(i + 1))
    audio = tap.cut(T0 + FRAME_S / 2, T0 + 4 * FRAME_S)
    assert audio.dtype == np.int16
    assert len(audio) == round(3.5 * FRAME)
    half = FRAME // 2
    assert (audio[:half] == 1).all()
    assert (audio[half : half + FRAME] == 2).all()
    assert (audio[half + FRAME : half + 2 * FRAME] == 0).all()
    assert (audio[half + 2 * FRAME :] == 4).all()


def test_tap_keeps_only_the_preroll_unless_held():
    tap = TurnTap(Sink(), preroll=1.0)
    n = 150  # 3 s of frames
    for i in range(n):
        tap.received(T0 + i * FRAME_S, T0 + i * FRAME_S, frame(7))
    newest = T0 + (n - 1) * FRAME_S
    audio = tap.cut(T0, newest + FRAME_S)
    kept = np.flatnonzero(audio)
    # Everything older than one second before the newest frame is gone.
    assert kept[0] / RATE == pytest.approx(newest - 1.0 - T0, abs=FRAME_S)

    held = TurnTap(Sink(), preroll=1.0)
    held.hold_from(T0 + 0.5)
    for i in range(n):
        held.received(T0 + i * FRAME_S, T0 + i * FRAME_S, frame(7))
    audio = held.cut(T0, newest + FRAME_S)
    assert np.flatnonzero(audio)[0] / RATE == pytest.approx(0.5, abs=FRAME_S)
    held.hold_from(None)
    held.received(newest + FRAME_S, newest + FRAME_S, frame(7))
    audio = held.cut(T0, newest + 2 * FRAME_S)
    assert np.flatnonzero(audio)[0] / RATE == pytest.approx(newest - T0, abs=1.0 + FRAME_S)
