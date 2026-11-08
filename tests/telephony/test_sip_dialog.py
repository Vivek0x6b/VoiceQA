from voiceqa.telephony.sip.dialog import Dialog, ack_for_2xx, build_invite, via_value
from voiceqa.telephony.sip.message import (
    Headers,
    Request,
    Response,
    parse_message,
    parse_name_addr,
    parse_via,
)

VIA = via_value("10.0.0.1", 5062, "z9hG4bKone")


def make_invite(**kw):
    args = dict(
        target_uri="sip:agent@pbx.example",
        from_uri="sip:1001@pbx.example",
        from_tag="ftag",
        display_name=None,
        call_id="cid1",
        cseq=1,
        contact_uri="sip:1001@10.0.0.1:5062",
        via=VIA,
        sdp="v=0\r\n",
        user_agent="VoiceQA/test",
    )
    args.update(kw)
    return build_invite(**args)


def ok_response(invite, *, record_route=(), contact="<sip:agent@192.0.2.9:5070>"):
    h = Headers()
    h.add("Via", invite.headers.get("Via"))
    h.add("From", invite.headers.get("From"))
    h.add("To", invite.headers.get("To") + ";tag=rtag")
    h.add("Call-ID", invite.headers.get("Call-ID"))
    h.add("CSeq", invite.headers.get("CSeq"))
    if contact:
        h.add("Contact", contact)
    for rr in record_route:
        h.add("Record-Route", rr)
    return Response(200, "OK", h)


def test_via_value():
    assert VIA == "SIP/2.0/UDP 10.0.0.1:5062;branch=z9hG4bKone;rport"
    assert parse_via(VIA).branch == "z9hG4bKone"


def test_build_invite_headers():
    inv = parse_message(make_invite().to_bytes())
    assert inv.method == "INVITE"
    assert inv.uri == "sip:agent@pbx.example"
    assert inv.headers.get("Via") == VIA
    assert inv.headers.get("Max-Forwards") == "70"
    assert inv.headers.get("From") == "<sip:1001@pbx.example>;tag=ftag"
    assert inv.headers.get("To") == "<sip:agent@pbx.example>"
    assert inv.headers.get("Call-ID") == "cid1"
    assert inv.cseq == (1, "INVITE")
    assert inv.headers.get("Contact") == "<sip:1001@10.0.0.1:5062>"
    assert inv.headers.get("Content-Type") == "application/sdp"
    assert inv.headers.get("User-Agent") == "VoiceQA/test"
    assert "BYE" in inv.headers.get("Allow")
    assert inv.body == b"v=0\r\n"


def test_build_invite_display_name_is_quoted_and_extra_headers_added():
    inv = make_invite(
        display_name='QA "bot"', cseq=2, extra_headers=[("Authorization", "Digest x")]
    )
    na = parse_name_addr(inv.headers.get("From"))
    assert inv.headers.get("From").startswith('"QA \\"bot\\"" <')
    assert na.display == 'QA "bot"'
    assert inv.headers.get("Authorization") == "Digest x"
    assert inv.cseq == (2, "INVITE")


def test_dialog_from_response():
    inv = make_invite()
    d = Dialog.from_response(
        inv, ok_response(inv, record_route=["<sip:p1;lr>, <sip:p2;lr>", "<sip:p3;lr>"])
    )
    assert d.call_id == "cid1"
    assert d.local_tag == "ftag"
    assert d.remote_tag == "rtag"
    assert d.remote_target == "sip:agent@192.0.2.9:5070"
    assert d.route_set == ("<sip:p3;lr>", "<sip:p2;lr>", "<sip:p1;lr>")
    assert d.local_seq == 1


def test_dialog_without_contact_targets_request_uri():
    inv = make_invite()
    d = Dialog.from_response(inv, ok_response(inv, contact=None))
    assert d.remote_target == "sip:agent@pbx.example"


def test_bye_increments_cseq_and_carries_route():
    inv = make_invite()
    d = Dialog.from_response(inv, ok_response(inv, record_route=["<sip:p1;lr>", "<sip:p2;lr>"]))
    via2 = via_value("10.0.0.1", 5062, "z9hG4bKtwo")
    bye = d.make_request("BYE", via2)
    assert bye.method == "BYE"
    assert bye.uri == "sip:agent@192.0.2.9:5070"
    assert bye.cseq == (2, "BYE")
    assert bye.headers.get("Via") == via2
    assert bye.headers.get("From") == "<sip:1001@pbx.example>;tag=ftag"
    assert parse_name_addr(bye.headers.get("To")).tag == "rtag"
    assert bye.headers.get("Call-ID") == "cid1"
    assert bye.headers.get_all("Route") == ["<sip:p2;lr>", "<sip:p1;lr>"]
    assert bye.headers.get("Max-Forwards") == "70"
    assert d.make_request("BYE", via2).cseq == (3, "BYE")


def test_ack_for_2xx_keeps_invite_cseq_and_credentials():
    inv = make_invite(
        cseq=2,
        extra_headers=[("Authorization", "Digest a"), ("Proxy-Authorization", "Digest b")],
    )
    d = Dialog.from_response(inv, ok_response(inv))
    ack = ack_for_2xx(d, inv, via_value("10.0.0.1", 5062, "z9hG4bKack"))
    assert ack.method == "ACK"
    assert ack.uri == "sip:agent@192.0.2.9:5070"
    assert ack.cseq == (2, "ACK")
    assert parse_via(ack.headers.get("Via")).branch == "z9hG4bKack"
    assert ack.headers.get("Authorization") == "Digest a"
    assert ack.headers.get("Proxy-Authorization") == "Digest b"
    assert d.local_seq == 2


def test_matches_in_dialog_request():
    inv = make_invite()
    d = Dialog.from_response(inv, ok_response(inv))
    h = Headers()
    h.add("From", "<sip:agent@pbx.example>;tag=rtag")
    h.add("To", "<sip:1001@pbx.example>;tag=ftag")
    h.add("Call-ID", "cid1")
    h.add("CSeq", "5 BYE")
    assert d.matches(Request("BYE", "sip:1001@10.0.0.1", h))
    h.set("Call-ID", "other")
    assert not d.matches(Request("BYE", "sip:1001@10.0.0.1", h))
    h.set("Call-ID", "cid1")
    h.set("To", "<sip:1001@pbx.example>;tag=wrong")
    assert not d.matches(Request("BYE", "sip:1001@10.0.0.1", h))
