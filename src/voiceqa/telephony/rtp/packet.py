"""RTP packet header build and parse (RFC 3550), plus outgoing stream state.

Pure byte code with no I/O, so it can be swapped for a native version later.
"""

import secrets
import struct
from dataclasses import dataclass

RTP_VERSION = 2
HEADER_SIZE = 12

_HEADER = struct.Struct("!BBHII")


@dataclass(frozen=True)
class RtpPacket:
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False


def build_packet(packet: RtpPacket) -> bytes:
    second = (0x80 if packet.marker else 0) | (packet.payload_type & 0x7F)
    header = _HEADER.pack(RTP_VERSION << 6, second, packet.sequence, packet.timestamp, packet.ssrc)
    return header + packet.payload


def parse_packet(data: bytes) -> RtpPacket:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"RTP packet too short: {len(data)} bytes")
    first, second, sequence, timestamp, ssrc = _HEADER.unpack_from(data)
    if first >> 6 != RTP_VERSION:
        raise ValueError(f"unsupported RTP version {first >> 6}")

    end = len(data)
    if first & 0x20:
        pad = data[-1]
        if pad == 0 or pad > end - HEADER_SIZE:
            raise ValueError(f"bad RTP padding length {pad}")
        end -= pad

    start = HEADER_SIZE + 4 * (first & 0x0F)
    if first & 0x10:
        if start + 4 > end:
            raise ValueError("RTP header extension truncated")
        (words,) = struct.unpack_from("!H", data, start + 2)
        start += 4 + 4 * words
    if start > end:
        raise ValueError("RTP header longer than packet")

    return RtpPacket(
        payload_type=second & 0x7F,
        sequence=sequence,
        timestamp=timestamp,
        ssrc=ssrc,
        payload=bytes(data[start:end]),
        marker=bool(second & 0x80),
    )


def timestamp_delta(ts: int, base: int) -> int:
    """Signed distance from base to ts, allowing for 32-bit wraparound."""
    d = (ts - base) & 0xFFFFFFFF
    return d - 2**32 if d >= 2**31 else d


class RtpStream:
    """Header state of one outgoing stream. Attributes hold the values of the next packet."""

    def __init__(
        self,
        payload_type: int,
        ssrc: int | None = None,
        sequence: int | None = None,
        timestamp: int | None = None,
    ) -> None:
        self.payload_type = payload_type
        self.ssrc = secrets.randbits(32) if ssrc is None else ssrc
        self.sequence = secrets.randbits(16) if sequence is None else sequence
        self.timestamp = secrets.randbits(32) if timestamp is None else timestamp
        self._first = True

    def next_packet(self, payload: bytes) -> bytes:
        data = build_packet(
            RtpPacket(
                payload_type=self.payload_type,
                sequence=self.sequence,
                timestamp=self.timestamp,
                ssrc=self.ssrc,
                payload=payload,
                marker=self._first,
            )
        )
        self._first = False
        self.sequence = (self.sequence + 1) & 0xFFFF
        # One G.711 byte is one sample, so the payload length is the timestamp step.
        self.timestamp = (self.timestamp + len(payload)) & 0xFFFFFFFF
        return data

    def skip(self, samples: int) -> None:
        self.timestamp = (self.timestamp + samples) & 0xFFFFFFFF
