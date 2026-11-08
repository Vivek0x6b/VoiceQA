import struct

import pytest

from voiceqa.telephony.rtp.packet import (
    RtpPacket,
    RtpStream,
    build_packet,
    parse_packet,
    timestamp_delta,
)


def test_known_header_bytes():
    pkt = RtpPacket(
        payload_type=0,
        sequence=0x1234,
        timestamp=0xDEADBEEF,
        ssrc=0x01020304,
        payload=b"\xff",
        marker=True,
    )
    assert build_packet(pkt) == bytes.fromhex("80801234deadbeef01020304ff")


def test_round_trip():
    pkt = RtpPacket(payload_type=8, sequence=7, timestamp=160, ssrc=99, payload=b"\xd5" * 160)
    assert parse_packet(build_packet(pkt)) == pkt


def test_parse_with_csrc_extension_padding():
    first = 0x80 | 0x20 | 0x10 | 2  # V=2, P, X, CC=2
    header = struct.pack("!BBHII", first, 0x80 | 8, 5, 1000, 42)
    csrcs = struct.pack("!II", 11, 22)
    extension = struct.pack("!HH", 0xBEDE, 1) + b"\x01\x02\x03\x04"
    payload = b"abcdef"
    padding = b"\x00\x00\x03"
    pkt = parse_packet(header + csrcs + extension + payload + padding)
    assert pkt == RtpPacket(
        payload_type=8, sequence=5, timestamp=1000, ssrc=42, payload=payload, marker=True
    )


def _header(first: int = 0x80) -> bytes:
    return struct.pack("!BBHII", first, 0, 1, 2, 3)


@pytest.mark.parametrize(
    "data",
    [
        b"\x80" * 11,
        _header(0x40) + b"x",
        _header(0xA0) + b"ab\x00",
        _header(0xA0) + b"ab\x09",
        _header(0x8F) + b"\x00" * 8,
        _header(0x90) + struct.pack("!HH", 0xBEDE, 3) + b"\x00" * 4,
        _header(0x90) + b"\x00\x00",
    ],
    ids=[
        "short",
        "version1",
        "padding-zero",
        "padding-too-long",
        "csrc-truncated",
        "extension-truncated",
        "extension-header-truncated",
    ],
)
def test_parse_rejects(data):
    with pytest.raises(ValueError):
        parse_packet(data)


def test_stream_sequence_and_timestamp_wrap():
    stream = RtpStream(0, ssrc=1, sequence=0xFFFF, timestamp=2**32 - 160)
    packets = [parse_packet(stream.next_packet(b"\xff" * 160)) for _ in range(3)]
    assert [p.sequence for p in packets] == [0xFFFF, 0, 1]
    assert [p.timestamp for p in packets] == [2**32 - 160, 0, 160]
    assert [p.marker for p in packets] == [True, False, False]
    assert {p.ssrc for p in packets} == {1}
    assert {p.payload_type for p in packets} == {0}


def test_stream_skip():
    stream = RtpStream(8, ssrc=1, sequence=10, timestamp=1000)
    stream.next_packet(b"\xd5" * 160)
    stream.skip(480)
    pkt = parse_packet(stream.next_packet(b"\xd5" * 160))
    assert pkt.timestamp == 1000 + 160 + 480
    assert pkt.sequence == 11


def test_stream_random_start():
    a, b = RtpStream(0), RtpStream(0)
    assert (a.ssrc, a.sequence, a.timestamp) != (b.ssrc, b.sequence, b.timestamp)
    assert 0 <= a.sequence < 2**16
    assert 0 <= a.timestamp < 2**32


def test_timestamp_delta():
    assert timestamp_delta(5, 2**32 - 5) == 10
    assert timestamp_delta(2**32 - 5, 5) == -10
    assert timestamp_delta(160, 0) == 160
