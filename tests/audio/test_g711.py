import warnings

import numpy as np
import pytest

from voiceqa.audio.g711 import decode_pcma, decode_pcmu, encode_pcma, encode_pcmu

ALL_INT16 = np.arange(-32768, 32768, dtype=np.int16)


@pytest.fixture
def audioop():
    # audioop is gone in 3.13 and warns on import in 3.11 and 3.12.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return pytest.importorskip("audioop")


@pytest.mark.parametrize(
    ("linear", "code"),
    [
        (0, 0xFF),
        (1, 0xFF),
        (-1, 0x7E),
        (8, 0xFE),
        (-8, 0x7E),
        (100, 0xF2),
        (-100, 0x72),
        (1000, 0xCE),
        (-1000, 0x4E),
        (4096, 0xAF),
        (-4096, 0x2F),
        (32124, 0x80),
        (-32124, 0x00),
        (32767, 0x80),
        (-32768, 0x00),
    ],
)
def test_pcmu_encode_reference(linear, code):
    assert encode_pcmu(np.array([linear], dtype=np.int16)) == bytes([code])


@pytest.mark.parametrize(
    ("code", "linear"),
    [
        (0x00, -32124),
        (0x01, -31100),
        (0x7F, 0),
        (0x80, 32124),
        (0x81, 31100),
        (0xFF, 0),
        (0x55, -716),
        (0xD5, 716),
        (0x2A, -5372),
        (0xAA, 5372),
    ],
)
def test_pcmu_decode_reference(code, linear):
    out = decode_pcmu(bytes([code]))
    assert out.dtype == np.int16
    assert out.tolist() == [linear]


def test_pcmu_all_codes_round_trip():
    codes = bytes(range(256))
    expected = bytearray(codes)
    expected[0x7F] = 0xFF  # negative zero decodes to 0, which encodes as 0xFF
    assert encode_pcmu(decode_pcmu(codes)) == bytes(expected)


def test_pcmu_round_trip_error_bound():
    out = decode_pcmu(encode_pcmu(ALL_INT16))
    err = np.abs(out.astype(np.int32) - ALL_INT16.astype(np.int32))
    assert err.max() <= 1024
    # Segment 0 has a step of 8 on the 16-bit scale.
    small = np.abs(ALL_INT16.astype(np.int32)) < 100
    assert err[small].max() < 8


def test_pcmu_silence():
    data = encode_pcmu(np.zeros(160, dtype=np.int16))
    assert data == b"\xff" * 160
    assert not decode_pcmu(data).any()


def test_pcmu_clipping():
    assert encode_pcmu(np.array([40000, -40000], dtype=np.int32)) == b"\x80\x00"


def test_pcmu_empty():
    assert encode_pcmu([]) == b""
    out = decode_pcmu(b"")
    assert out.shape == (0,)
    assert out.dtype == np.int16


def test_encode_rejects_float():
    with pytest.raises(TypeError):
        encode_pcmu(np.zeros(4))


def test_pcmu_matches_audioop(audioop):
    expected = audioop.lin2ulaw(ALL_INT16.astype("<i2").tobytes(), 2)
    assert encode_pcmu(ALL_INT16) == expected
    codes = bytes(range(256))
    expected_lin = np.frombuffer(audioop.ulaw2lin(codes, 2), dtype="<i2")
    np.testing.assert_array_equal(decode_pcmu(codes), expected_lin)


@pytest.mark.parametrize(
    ("linear", "code"),
    [
        (0, 0xD5),
        (1, 0xD5),
        (-1, 0x55),
        (8, 0xD5),
        (-8, 0x55),
        (100, 0xD3),
        (-100, 0x53),
        (1000, 0xFA),
        (-1000, 0x7A),
        (4096, 0x85),
        (-4096, 0x1A),
        (32767, 0xAA),
        (-32768, 0x2A),
    ],
)
def test_pcma_encode_reference(linear, code):
    assert encode_pcma(np.array([linear], dtype=np.int16)) == bytes([code])


@pytest.mark.parametrize(
    ("code", "linear"),
    [
        (0xD5, 8),
        (0x55, -8),
        (0xAA, 32256),
        (0x2A, -32256),
        (0x00, -5504),
        (0x80, 5504),
        (0x01, -5248),
        (0x81, 5248),
        (0x7F, -848),
        (0xFF, 848),
    ],
)
def test_pcma_decode_reference(code, linear):
    out = decode_pcma(bytes([code]))
    assert out.dtype == np.int16
    assert out.tolist() == [linear]


def test_pcma_all_codes_round_trip():
    codes = bytes(range(256))
    assert encode_pcma(decode_pcma(codes)) == codes


def test_pcma_round_trip_error_bound():
    out = decode_pcma(encode_pcma(ALL_INT16))
    err = np.abs(out.astype(np.int32) - ALL_INT16.astype(np.int32))
    assert err.max() <= 1024
    # Segments 0 and 1 have a step of 16 on the 16-bit scale.
    small = np.abs(ALL_INT16.astype(np.int32)) < 256
    assert err[small].max() < 16


def test_pcma_silence():
    data = encode_pcma(np.zeros(160, dtype=np.int16))
    assert data == b"\xd5" * 160
    assert decode_pcma(data).tolist() == [8] * 160


def test_pcma_clipping():
    assert encode_pcma(np.array([40000, -40000], dtype=np.int32)) == b"\xaa\x2a"


def test_pcma_empty():
    assert encode_pcma([]) == b""
    out = decode_pcma(b"")
    assert out.shape == (0,)
    assert out.dtype == np.int16


def test_pcma_matches_audioop(audioop):
    expected = audioop.lin2alaw(ALL_INT16.astype("<i2").tobytes(), 2)
    assert encode_pcma(ALL_INT16) == expected
    codes = bytes(range(256))
    expected_lin = np.frombuffer(audioop.alaw2lin(codes, 2), dtype="<i2")
    np.testing.assert_array_equal(decode_pcma(codes), expected_lin)
