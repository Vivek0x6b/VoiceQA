"""SDP offer and answer for one G.711 audio stream (RFC 4566, RFC 3264)."""

import secrets
from collections.abc import Sequence

from voiceqa.telephony.media import LocalMedia, PayloadType, RemoteMedia
from voiceqa.telephony.sip.errors import SdpError


def _build(local: LocalMedia, payload_types: Sequence[PayloadType], session_id: int | None) -> str:
    sid = session_id if session_id is not None else secrets.randbelow(2**31)
    pts = " ".join(str(int(pt)) for pt in payload_types)
    lines = [
        "v=0",
        f"o=voiceqa {sid} {sid} IN IP4 {local.host}",
        "s=voiceqa",
        f"c=IN IP4 {local.host}",
        "t=0 0",
        f"m=audio {local.port} RTP/AVP {pts}",
    ]
    lines += [f"a=rtpmap:{int(pt)} {pt.name}/8000" for pt in payload_types]
    lines += [f"a=ptime:{local.ptime_ms}", "a=sendrecv"]
    return "\r\n".join(lines) + "\r\n"


def build_offer(local: LocalMedia, session_id: int | None = None) -> str:
    return _build(local, local.payload_types, session_id)


def build_answer(
    local: LocalMedia, payload_type: PayloadType, session_id: int | None = None
) -> str:
    if payload_type not in local.payload_types:
        raise ValueError(f"{payload_type.name} is not one of our payload types")
    return _build(local, (payload_type,), session_id)


def _connection_host(value: str) -> str:
    parts = value.split()
    if len(parts) != 3 or parts[0] != "IN":
        raise SdpError(f"bad connection line: c={value}")
    if parts[1] != "IP4":
        raise SdpError(f"only IPv4 connections are supported, got c={value}")
    return parts[2].split("/")[0]


def parse_answer(sdp: str | bytes, offered: Sequence[PayloadType]) -> RemoteMedia:
    return _parse(sdp, offered, "answer")


def parse_offer(sdp: str | bytes, supported: Sequence[PayloadType]) -> RemoteMedia:
    """The far end's media and the first codec in its offer that we support."""
    return _parse(sdp, supported, "offer")


def _parse(sdp: str | bytes, accepted: Sequence[PayloadType], role: str) -> RemoteMedia:
    if isinstance(sdp, bytes):
        sdp = sdp.decode("utf-8", errors="replace")

    session_host = None
    media_host = None
    media = None
    in_audio = False
    seen_m = False
    for raw in sdp.splitlines():
        line = raw.strip()
        kind, eq, value = line.partition("=")
        if not eq:
            continue
        if kind == "m":
            if media is not None:
                break  # only the first audio stream is used
            seen_m = True
            in_audio = value.startswith("audio ")
            if in_audio:
                media = value.split()
        elif kind == "c":
            if not seen_m:
                session_host = _connection_host(value)
            elif in_audio:
                media_host = _connection_host(value)

    if media is None:
        raise SdpError(f"{role} has no audio stream")
    if len(media) < 4 or not media[1].split("/")[0].isdigit():
        raise SdpError(f"bad m= line: m={' '.join(media)}")
    port = int(media[1].split("/")[0])
    if port == 0:
        raise SdpError(f"{role} rejected the audio stream (port 0)")
    if media[2] != "RTP/AVP":
        raise SdpError(f"only RTP/AVP is supported, got {media[2]}")
    host = media_host or session_host
    if host is None:
        raise SdpError(f"{role} has no connection (c=) line")

    wanted = {int(pt) for pt in accepted}
    for fmt in media[3:]:
        if fmt.isdigit() and int(fmt) in wanted:
            return RemoteMedia(host=host, port=port, payload_type=PayloadType(int(fmt)))
    raise SdpError(f"no common codec, {role} has {' '.join(media[3:])}")
