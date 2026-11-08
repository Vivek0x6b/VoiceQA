import pytest

from voiceqa.telephony.sip.errors import SipParseError
from voiceqa.telephony.sip.message import (
    Headers,
    Request,
    Response,
    build_response,
    new_branch,
    new_call_id,
    new_tag,
    parse_message,
    parse_name_addr,
    parse_via,
)

INVITE = (
    b"INVITE sip:agent@example.com SIP/2.0\r\n"
    b"v: SIP/2.0/UDP 10.0.0.1:5060;branch=z9hG4bKaaa\r\n"
    b"Via: SIP/2.0/UDP 10.0.0.2:5060;branch=z9hG4bKbbb\r\n"
    b"VIA: SIP/2.0/UDP 10.0.0.3;branch=z9hG4bKccc, SIP/2.0/UDP 10.0.0.4;branch=z9hG4bKddd\r\n"
    b'f: "Bob" <sip:bob@example.com>;tag=12\r\n'
    b"t: <sip:agent@example.com>\r\n"
    b"i: abc@host\r\n"
    b"CSeq: 7 INVITE\r\n"
    b"m: <sip:bob@10.0.0.1:5060>\r\n"
    b"c: application/sdp\r\n"
    b"l: 4\r\n"
    b"\r\n"
    b"v=0\r\nextra"
)


def test_parse_request_with_compact_headers():
    msg = parse_message(INVITE)
    assert isinstance(msg, Request)
    assert msg.method == "INVITE"
    assert msg.uri == "sip:agent@example.com"
    assert msg.headers.get("Call-ID") == "abc@host"
    assert msg.headers.get("call-id") == "abc@host"
    assert msg.headers.get("FROM") == '"Bob" <sip:bob@example.com>;tag=12'
    assert msg.headers.get("Contact") == "<sip:bob@10.0.0.1:5060>"
    assert msg.headers.get("Content-Type") == "application/sdp"
    assert msg.cseq == (7, "INVITE")


def test_multiple_via_lines_and_comma_joined_values():
    vias = parse_message(INVITE).headers.get_all("Via")
    assert [parse_via(v).branch for v in vias] == [
        "z9hG4bKaaa",
        "z9hG4bKbbb",
        "z9hG4bKccc",
        "z9hG4bKddd",
    ]


def test_content_length_is_honored():
    assert parse_message(INVITE).body == b"v=0\r"


def test_short_body_rejected():
    data = INVITE.replace(b"l: 4", b"l: 400")
    with pytest.raises(SipParseError, match="Content-Length"):
        parse_message(data)


def test_missing_content_length_takes_rest_of_datagram():
    data = b"OPTIONS sip:a@b SIP/2.0\r\nCall-ID: x\r\n\r\nhello"
    assert parse_message(data).body == b"hello"


def test_header_folding():
    data = b"OPTIONS sip:a@b SIP/2.0\r\nSubject: one\r\n  two\r\nl: 0\r\n\r\n"
    assert parse_message(data).headers.get("Subject") == "one two"


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"hello world\r\n\r\n",
        b"INVITE sip:a@b HTTP/1.1\r\n\r\n",
        b"SIP/2.0 abc OK\r\n\r\n",
        b"INVITE sip:a@b SIP/2.0\r\nno colon here\r\n\r\n",
        b"\xff\xfe\x00",
    ],
)
def test_garbage_rejected(data):
    with pytest.raises(SipParseError):
        parse_message(data)


def test_parse_response():
    data = b"SIP/2.0 486 Busy Here\r\nCSeq: 1 INVITE\r\nContent-Length: 0\r\n\r\n"
    msg = parse_message(data)
    assert isinstance(msg, Response)
    assert msg.status == 486
    assert msg.reason == "Busy Here"
    assert msg.cseq == (1, "INVITE")


def test_to_bytes_round_trip_uses_crlf_and_sets_content_length():
    headers = Headers()
    headers.add("Via", "SIP/2.0/UDP 1.2.3.4:5060;branch=z9hG4bK1")
    headers.add("Call-ID", "x")
    headers.add("Content-Length", "999")
    req = Request("BYE", "sip:a@b", headers, b"body")
    data = req.to_bytes()
    assert data.startswith(b"BYE sip:a@b SIP/2.0\r\n")
    assert b"Content-Length: 4\r\n\r\nbody" in data
    assert b"999" not in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    back = parse_message(data)
    assert back.headers.get("Via") == "SIP/2.0/UDP 1.2.3.4:5060;branch=z9hG4bK1"
    assert back.body == b"body"


def test_headers_set_and_remove():
    h = Headers()
    h.add("Route", "<sip:a>")
    h.add("route", "<sip:b>")
    assert h.get_all("ROUTE") == ["<sip:a>", "<sip:b>"]
    h.set("Route", "<sip:c>")
    assert h.get_all("Route") == ["<sip:c>"]
    h.remove("route")
    assert "Route" not in h
    assert h.get("Route") is None


def test_parse_name_addr_with_display_name():
    na = parse_name_addr('"Bob, Jr" <sip:bob@x;transport=udp>;tag=12;foo')
    assert na.display == "Bob, Jr"
    assert na.uri == "sip:bob@x;transport=udp"
    assert na.tag == "12"
    assert na.params["foo"] is None


def test_parse_name_addr_without_brackets():
    na = parse_name_addr("sip:bob@x;tag=9")
    assert na.uri == "sip:bob@x"
    assert na.tag == "9"


def test_contact_list_split_respects_quotes():
    h = Headers()
    h.add("Contact", '"A, B" <sip:a@x>, <sip:b@y>')
    assert h.get_all("Contact") == ['"A, B" <sip:a@x>', "<sip:b@y>"]


def test_parse_via():
    via = parse_via("SIP/2.0/UDP 1.2.3.4:5070;branch=z9hG4bKx;rport;received=5.6.7.8")
    assert via.transport == "UDP"
    assert via.host == "1.2.3.4"
    assert via.port == 5070
    assert via.branch == "z9hG4bKx"
    assert via.params["rport"] is None
    assert via.params["received"] == "5.6.7.8"
    assert parse_via("SIP/2.0/UDP host").port is None


def test_generated_ids():
    assert new_branch().startswith("z9hG4bK")
    assert new_branch() != new_branch()
    assert new_tag() != new_tag()
    assert new_call_id() != new_call_id()


def test_build_response_copies_dialog_headers():
    req = parse_message(INVITE)
    resp = build_response(req, 200, "OK")
    assert resp.status == 200
    assert len(resp.headers.get_all("Via")) == 4
    for name in ("From", "To", "Call-ID", "CSeq"):
        assert resp.headers.get(name) == req.headers.get(name)
    assert resp.headers.get("Contact") is None
