import numpy as np

from voiceqa.audio.g711 import decode_pcmu, encode_pcmu
from voiceqa.telephony.rtp.packet import RtpPacket
from voiceqa.telephony.rtp.recorder import Recorder


def _frame(value: int) -> bytes:
    return encode_pcmu(np.full(160, value, dtype=np.int16))


def _pkt(ts: int, value: int, ssrc: int = 1) -> RtpPacket:
    return RtpPacket(payload_type=0, sequence=0, timestamp=ts, ssrc=ssrc, payload=_frame(value))


def _expected(*values: int) -> np.ndarray:
    return np.concatenate([decode_pcmu(_frame(v)) for v in values])


def test_empty():
    audio = Recorder(decode_pcmu).audio()
    assert audio.dtype == np.int16
    assert len(audio) == 0


def test_in_order():
    rec = Recorder(decode_pcmu)
    for i, v in enumerate([1000, 2000, 3000]):
        assert rec.add(_pkt(5000 + 160 * i, v))
    np.testing.assert_array_equal(rec.audio(), _expected(1000, 2000, 3000))


def test_reordered_duplicate_and_gap():
    rec = Recorder(decode_pcmu)
    base = 777
    assert rec.add(_pkt(base + 320, 3000))
    assert rec.add(_pkt(base, 1000))
    assert rec.add(_pkt(base + 160, 2000))
    assert not rec.add(_pkt(base + 160, -9000))
    assert rec.add(_pkt(base + 640, 5000))
    audio = rec.audio()
    assert audio.dtype == np.int16
    np.testing.assert_array_equal(audio[:480], _expected(1000, 2000, 3000))
    np.testing.assert_array_equal(audio[480:640], np.zeros(160, dtype=np.int16))
    np.testing.assert_array_equal(audio[640:], _expected(5000))


def test_timestamp_wrap():
    rec = Recorder(decode_pcmu)
    rec.add(_pkt(2**32 - 160, 1000))
    rec.add(_pkt(0, 2000))
    np.testing.assert_array_equal(rec.audio(), _expected(1000, 2000))


def test_ssrc_change_appends():
    rec = Recorder(decode_pcmu)
    rec.add(_pkt(0, 1000, ssrc=1))
    rec.add(_pkt(160, 2000, ssrc=1))
    rec.add(_pkt(999_999, 3000, ssrc=2))
    rec.add(_pkt(999_999 + 160, 4000, ssrc=2))
    np.testing.assert_array_equal(rec.audio(), _expected(1000, 2000, 3000, 4000))


def test_wild_timestamp_dropped():
    rec = Recorder(decode_pcmu)
    rec.add(_pkt(1000, 1000))
    assert not rec.add(_pkt(1000 + 2**31 - 1, 2000))
    assert not rec.add(_pkt(1000 - 2**31, 2000))
    assert len(rec.audio()) == 160
