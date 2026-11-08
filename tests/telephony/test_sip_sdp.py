import pytest

from voiceqa.telephony.media import LocalMedia, PayloadType, RemoteMedia
from voiceqa.telephony.sip.errors import SdpError
from voiceqa.telephony.sip.sdp import build_offer, parse_answer

BOTH = (PayloadType.PCMU, PayloadType.PCMA)


def answer(*lines: str) -> str:
    return "\r\n".join(lines) + "\r\n"


BASIC = answer(
    "v=0",
    "o=agent 1 1 IN IP4 203.0.113.5",
    "s=-",
    "c=IN IP4 203.0.113.5",
    "t=0 0",
    "m=audio 18000 RTP/AVP 8 0 101",
    "a=rtpmap:8 PCMA/8000",
    "a=rtpmap:101 telephone-event/8000",
    "a=ptime:20",
)


def test_offer_text():
    local = LocalMedia(host="192.168.1.10", port=40000)
    assert build_offer(local, session_id=1) == (
        "v=0\r\n"
        "o=voiceqa 1 1 IN IP4 192.168.1.10\r\n"
        "s=voiceqa\r\n"
        "c=IN IP4 192.168.1.10\r\n"
        "t=0 0\r\n"
        "m=audio 40000 RTP/AVP 0 8\r\n"
        "a=rtpmap:0 PCMU/8000\r\n"
        "a=rtpmap:8 PCMA/8000\r\n"
        "a=ptime:20\r\n"
        "a=sendrecv\r\n"
    )


def test_offer_keeps_preference_order():
    local = LocalMedia(host="10.0.0.1", port=4000, payload_types=(PayloadType.PCMA,))
    offer = build_offer(local)
    assert "m=audio 4000 RTP/AVP 8\r\n" in offer
    assert "a=rtpmap:8 PCMA/8000\r\n" in offer
    assert "PCMU" not in offer


def test_parse_basic_answer_picks_first_offered_in_answer_order():
    remote = parse_answer(BASIC, BOTH)
    assert remote == RemoteMedia(host="203.0.113.5", port=18000, payload_type=PayloadType.PCMA)
    assert remote.payload_type is PayloadType.PCMA


def test_parse_answer_bytes_and_lf_only():
    remote = parse_answer(BASIC.replace("\r\n", "\n").encode(), BOTH)
    assert remote.port == 18000


def test_media_connection_overrides_session():
    sdp = BASIC.replace("a=ptime:20", "c=IN IP4 198.51.100.7\r\na=ptime:20")
    assert parse_answer(sdp, BOTH).host == "198.51.100.7"


def test_connection_only_at_media_level():
    sdp = BASIC.replace("c=IN IP4 203.0.113.5\r\n", "").replace(
        "a=ptime:20", "c=IN IP4 198.51.100.7"
    )
    assert parse_answer(sdp, BOTH).host == "198.51.100.7"


def test_skips_codecs_we_did_not_offer():
    assert parse_answer(BASIC, (PayloadType.PCMU,)).payload_type is PayloadType.PCMU


def test_remote_ptime_does_not_break_parse():
    sdp = BASIC.replace("a=ptime:20", "a=ptime:30")
    assert parse_answer(sdp, BOTH).ptime_ms == 20


def test_port_zero_is_rejected_stream():
    with pytest.raises(SdpError, match="rejected"):
        parse_answer(BASIC.replace("m=audio 18000", "m=audio 0"), BOTH)


def test_no_common_codec():
    sdp = BASIC.replace("RTP/AVP 8 0 101", "RTP/AVP 18 101")
    with pytest.raises(SdpError, match="codec"):
        parse_answer(sdp, BOTH)


@pytest.mark.parametrize(
    "sdp, match",
    [
        ("", "audio"),
        (BASIC.replace("m=audio", "m=video"), "audio"),
        (BASIC.replace("c=IN IP4 203.0.113.5\r\n", ""), "connection"),
        (BASIC.replace("IN IP4 203.0.113.5\r\nt=", "IN IP6 ::1\r\nt="), "IPv4"),
        (BASIC.replace("RTP/AVP", "RTP/SAVP"), "RTP/AVP"),
        (BASIC.replace("m=audio 18000", "m=audio abc"), "m="),
    ],
)
def test_bad_answers(sdp, match):
    with pytest.raises(SdpError, match=match):
        parse_answer(sdp, BOTH)


def test_uses_first_audio_stream_after_video():
    sdp = BASIC.replace(
        "m=audio 18000 RTP/AVP 8 0 101",
        "m=video 5000 RTP/AVP 96\r\nc=IN IP4 192.0.2.99\r\nm=audio 18000 RTP/AVP 8 0 101",
    )
    assert parse_answer(sdp, BOTH).host == "203.0.113.5"
