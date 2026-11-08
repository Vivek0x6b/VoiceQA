import pytest

from voiceqa.telephony.rtp.packet import RtpPacket
from voiceqa.telephony.rtp.timeline import ReceiveTimeline


def _pkt(ts: int, ssrc: int = 1) -> RtpPacket:
    return RtpPacket(payload_type=0, sequence=0, timestamp=ts & 0xFFFFFFFF, ssrc=ssrc, payload=b"")


def test_frames_are_placed_by_timestamp_not_by_arrival_jitter():
    tl = ReceiveTimeline()
    arrivals = [10.0, 10.031, 10.039, 10.075]
    got = [tl.place(_pkt(5000 + 160 * i), t) for i, t in enumerate(arrivals)]
    assert got == pytest.approx([10.0, 10.02, 10.04, 10.06])


def test_reordered_frame_goes_before_the_one_that_overtook_it():
    tl = ReceiveTimeline()
    assert tl.place(_pkt(320), 5.0) == pytest.approx(5.0)
    assert tl.place(_pkt(0), 5.001) == pytest.approx(4.96)
    assert tl.place(_pkt(160), 5.002) == pytest.approx(4.98)


def test_duplicates_are_dropped_within_the_history():
    tl = ReceiveTimeline(history=4)
    assert tl.place(_pkt(0), 1.0) is not None
    assert tl.place(_pkt(0), 1.01) is None
    for i in range(1, 5):
        tl.place(_pkt(160 * i), 1.0 + 0.02 * i)
    # The first frame has left the four-frame history, so a very late copy
    # is placed again. Whoever writes the audio drops it as late.
    assert tl.place(_pkt(0), 1.2) == pytest.approx(1.0)


def test_timestamp_wrap():
    tl = ReceiveTimeline()
    tl.place(_pkt(2**32 - 160), 3.0)
    assert tl.place(_pkt(0), 3.02) == pytest.approx(3.02)


def test_new_ssrc_starts_at_its_arrival_time():
    tl = ReceiveTimeline()
    tl.place(_pkt(0, ssrc=1), 1.0)
    tl.place(_pkt(160, ssrc=1), 1.02)
    assert tl.place(_pkt(999_999, ssrc=2), 4.5) == pytest.approx(4.5)
    assert tl.place(_pkt(999_999 + 160, ssrc=2), 4.52) == pytest.approx(4.52)


def test_within_the_limits_the_timestamp_wins_over_arrival():
    tl = ReceiveTimeline(max_late=0.5, max_early=0.1)
    tl.place(_pkt(0), 1.0)
    assert tl.place(_pkt(160), 1.4) == pytest.approx(1.02)
    assert tl.place(_pkt(8000), 1.92) == pytest.approx(2.0)


def test_a_burst_after_a_stall_keeps_the_anchor():
    tl = ReceiveTimeline(max_late=0.5)
    tl.place(_pkt(0), 1.0)
    # The loop was blocked for 0.6 s; 30 queued packets arrive together.
    placed = [tl.place(_pkt(160 * k), 1.63) for k in range(1, 31)]
    late = [k for k in range(1, 31) if 1.0 + 0.02 * k < 1.63 - 0.5]
    for k, at in zip(range(1, 31), placed, strict=True):
        assert at == pytest.approx(1.63 if k in late else 1.0 + 0.02 * k)
    # Packets after the burst are still placed by timestamp.
    assert tl.place(_pkt(160 * 31), 1.64) == pytest.approx(1.62)
    assert tl.place(_pkt(160 * 40), 1.75) == pytest.approx(1.8)


def test_one_wild_timestamp_does_not_move_the_stream():
    tl = ReceiveTimeline()
    tl.place(_pkt(0), 1.0)
    assert tl.place(_pkt(160 + 2**31 - 1), 1.02) == pytest.approx(1.02)
    assert tl.place(_pkt(320), 1.041) == pytest.approx(1.04)


@pytest.mark.parametrize("ts_offset", [-16000, 2**30])
def test_a_lasting_timestamp_jump_reanchors(ts_offset):
    tl = ReceiveTimeline(reanchor_after=0.5)
    tl.place(_pkt(0), 1.0)
    # From 2.0 s on, the timestamps are off by ts_offset (backwards: a sender
    # that paused without advancing them; forwards: a restarted clock).
    times = [2.0 + 0.02 * k for k in range(40)]
    placed = [tl.place(_pkt(ts_offset + 160 * (50 + k)), t) for k, t in enumerate(times)]
    assert placed == pytest.approx(times)
    # Anchored again, so arrival jitter no longer moves the frames.
    assert tl.place(_pkt(ts_offset + 160 * 90), 2.8 + 0.013) == pytest.approx(2.8)
    assert tl.place(_pkt(ts_offset + 160 * 91), 2.82 + 0.03) == pytest.approx(2.82)
