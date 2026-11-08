import asyncio
import hashlib
import socket

import pytest

from voiceqa.telephony.media import LocalMedia, PayloadType, RemoteMedia
from voiceqa.telephony.sip import (
    SdpError,
    SipConfig,
    SipStatusError,
    SipTimeoutError,
    place_call,
)
from voiceqa.telephony.sip.message import (
    Headers,
    Request,
    Response,
    build_response,
    new_branch,
    parse_message,
    parse_name_addr,
    parse_params,
    parse_via,
)

LOCAL = LocalMedia(host="127.0.0.1", port=40000)
REALM = "pbx.test"
USER = "1001"
PASSWORD = "secret"


def sdp_answer(port=30000, pts="8 0"):
    return (
        "v=0\r\no=uas 1 1 IN IP4 127.0.0.1\r\ns=-\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n"
        f"m=audio {port} RTP/AVP {pts}\r\na=sendrecv\r\n"
    )


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


class FakeUas(asyncio.DatagramProtocol):
    def __init__(
        self,
        *,
        provisional=(100, 180),
        final=200,
        final_reason="OK",
        answer=None,
        challenges=(),
        challenge_code=401,
        drop_invites=0,
        drop_byes=0,
        silent=False,
        retransmit_ok=0,
        record_route=(),
        challenge_bye=False,
    ):
        self.provisional = provisional
        self.final = final
        self.final_reason = final_reason
        self.answer = sdp_answer() if answer is None else answer
        self.challenges = list(challenges)
        self.challenge_code = challenge_code
        self.drop_invites = drop_invites
        self.drop_byes = drop_byes
        self.silent = silent
        self.retransmit_ok = retransmit_ok
        self.record_route = record_route
        self.challenge_bye = challenge_bye
        self.received = []
        self.peer = None
        self.answered = {}
        self.invite_count = 0
        self.last_nonce = None
        self.dialog_request = None
        self.ok = None

    def connection_made(self, transport):
        self.transport = transport
        self.lost = asyncio.get_running_loop().create_future()

    def connection_lost(self, exc):
        self.lost.set_result(None)

    @property
    def port(self):
        return self.transport.get_extra_info("sockname")[1]

    def requests(self, method):
        return [m for m in self.received if isinstance(m, Request) and m.method == method]

    def transactions(self, method):
        # Timer A and E resend byte-identical copies when a reply is slow, and
        # a resent INVITE gets the same final response and so a second ACK.
        # Tests about the call flow count each distinct request once.
        out = {}
        for r in self.requests(method):
            out.setdefault(r.to_bytes(), r)
        return list(out.values())

    def responses(self, method):
        return [m for m in self.received if isinstance(m, Response) and m.cseq[1] == method]

    def send(self, msg):
        self.transport.sendto(msg.to_bytes(), self.peer)

    def reply(self, req, status, reason, tag="uastag"):
        resp = build_response(req, status, reason)
        if status > 100 and tag:
            resp.headers.set("To", req.headers.get("To") + f";tag={tag}")
        return resp

    def datagram_received(self, data, addr):
        msg = parse_message(data)
        self.received.append(msg)
        self.peer = addr
        if isinstance(msg, Response) or self.silent:
            return
        if msg.method == "INVITE":
            self.on_invite(msg)
        elif msg.method == "BYE":
            self.on_bye(msg)

    def challenge(self, req, nonce, stale, code):
        header = "WWW-Authenticate" if code == 401 else "Proxy-Authenticate"
        resp = self.reply(req, code, "Unauthorized" if code == 401 else "Proxy Auth Required")
        value = f'Digest realm="{REALM}", nonce="{nonce}", qop="auth", algorithm=MD5'
        if stale:
            value += ", stale=true"
        resp.headers.add(header, value)
        self.last_nonce = nonce
        return resp

    def digest_ok(self, req, code):
        name = "Authorization" if code == 401 else "Proxy-Authorization"
        value = req.headers.get(name)
        if not value:
            return False
        p = parse_params(value[len("Digest ") :].replace(",", ";"))
        ha1 = md5(f"{p['username']}:{p['realm']}:{PASSWORD}")
        ha2 = md5(f"{req.method}:{p['uri']}")
        expected = md5(f"{ha1}:{p['nonce']}:{p['nc']}:{p['cnonce']}:{p['qop']}:{ha2}")
        return (
            p["username"] == USER
            and p["nonce"] == self.last_nonce
            and p["uri"] == req.uri
            and p["response"] == expected
        )

    def on_invite(self, req):
        branch = parse_via(req.headers.get_all("Via")[0]).branch
        if branch in self.answered:
            self.send(self.answered[branch])
            return
        if self.drop_invites:
            self.drop_invites -= 1
            return
        self.invite_count += 1
        if self.challenges:
            nonce, stale = self.challenges.pop(0)
            resp = self.challenge(req, nonce, stale, self.challenge_code)
            self.answered[branch] = resp
            self.send(resp)
            return
        if self.last_nonce is not None and not self.digest_ok(req, self.challenge_code):
            resp = self.reply(req, 403, "Forbidden")
            self.answered[branch] = resp
            self.send(resp)
            return
        for code in self.provisional:
            self.send(self.reply(req, code, "Trying" if code == 100 else "Ringing"))
        resp = self.reply(req, self.final, self.final_reason)
        if self.final == 200:
            resp.headers.add("Contact", f"<sip:agent@127.0.0.1:{self.port}>")
            for rr in self.record_route:
                resp.headers.add("Record-Route", rr)
            resp.headers.add("Content-Type", "application/sdp")
            resp.body = self.answer.encode()
            self.dialog_request = req
            self.ok = resp
            loop = asyncio.get_running_loop()
            for i in range(self.retransmit_ok):
                loop.call_later(0.05 * (i + 1), self.send, resp)
        self.answered[branch] = resp
        self.send(resp)

    def on_bye(self, req):
        if self.drop_byes:
            self.drop_byes -= 1
            return
        if self.challenge_bye and not req.headers.get("Proxy-Authorization"):
            self.send(self.challenge(req, "byenonce", False, 407))
            return
        if self.challenge_bye and not self.digest_ok(req, 407):
            self.send(self.reply(req, 403, "Forbidden", tag=None))
            return
        self.send(self.reply(req, 200, "OK", tag=None))

    def send_bye(self, *, call_id=None, from_tag="uastag"):
        inv = self.dialog_request
        h = Headers()
        h.add("Via", f"SIP/2.0/UDP 127.0.0.1:{self.port};branch={new_branch()}")
        h.add("From", inv.headers.get("To") + f";tag={from_tag}")
        h.add("To", inv.headers.get("From"))
        h.add("Call-ID", call_id or inv.headers.get("Call-ID"))
        h.add("CSeq", "1 BYE")
        bye = Request("BYE", parse_name_addr(inv.headers.get("Contact")).uri, h)
        self.send(bye)
        return bye


async def start_uas(**kw):
    loop = asyncio.get_running_loop()
    _, uas = await loop.create_datagram_endpoint(lambda: FakeUas(**kw), local_addr=("127.0.0.1", 0))
    return uas


def one_transaction(reqs):
    # Timer E may resend a request if the reply is slow (T1 is 20 ms here), so
    # count transactions, not packets: every copy must be byte-identical.
    assert reqs, "request was never sent"
    assert len({r.to_bytes() for r in reqs}) == 1, "more than one transaction"
    return reqs[0]


def config(uas, **kw):
    args = dict(
        server_host="127.0.0.1",
        server_port=uas.port,
        username=USER,
        password=PASSWORD,
        from_domain=REALM,
        local_host="127.0.0.1",
        timer_t1=0.02,
        timer_t2=0.16,
    )
    args.update(kw)
    return SipConfig(**args)


async def until(cond, timeout=2.0):
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.005)


def run(scenario, **uas_kw):
    async def main():
        uas = await start_uas(**uas_kw)
        try:
            return await scenario(uas)
        finally:
            await asyncio.sleep(0)
            uas.transport.abort()
            await uas.lost

    return asyncio.run(main())


def test_basic_call_and_hangup():
    async def scenario(uas):
        call = await place_call(config(uas), "sip:agent@pbx.test", LOCAL)
        await until(lambda: uas.transactions("ACK"))
        local_addr = call.local_addr
        await call.hangup()
        await call.hangup()
        return call, local_addr, uas

    call, local_addr, uas = run(scenario, provisional=(100, 180, 183))
    assert call.remote_media == RemoteMedia("127.0.0.1", 30000, PayloadType.PCMA)
    assert call.ended

    invite = one_transaction(uas.transactions("INVITE"))
    assert invite.uri == "sip:agent@pbx.test"
    via = parse_via(invite.headers.get("Via"))
    assert via.branch.startswith("z9hG4bK")
    assert "rport" in via.params
    assert (via.host, via.port) == local_addr == ("127.0.0.1", uas.peer[1])
    assert invite.headers.get("Contact") == f"<sip:{USER}@127.0.0.1:{local_addr[1]}>"
    assert parse_name_addr(invite.headers.get("From")).uri == f"sip:{USER}@{REALM}"
    assert parse_name_addr(invite.headers.get("From")).tag
    assert invite.headers.get("To") == "<sip:agent@pbx.test>"
    assert b"m=audio 40000 RTP/AVP 0 8\r\n" in invite.body
    assert invite.headers.get("Content-Type") == "application/sdp"

    (ack,) = uas.transactions("ACK")
    assert ack.uri == f"sip:agent@127.0.0.1:{uas.port}"
    assert ack.cseq == (1, "ACK")
    assert parse_name_addr(ack.headers.get("To")).tag == "uastag"
    assert parse_via(ack.headers.get("Via")).branch != via.branch

    bye = one_transaction(uas.transactions("BYE"))
    assert bye.cseq == (2, "BYE")
    assert bye.headers.get("Call-ID") == invite.headers.get("Call-ID") == call.call_id
    assert parse_name_addr(bye.headers.get("To")).tag == "uastag"


def test_target_without_scheme_uses_server_host():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas.transactions("INVITE")[0].uri

    assert run(scenario) == "sip:agent@127.0.0.1"


def test_wildcard_bind_advertises_routable_address():
    async def scenario(uas):
        call = await place_call(config(uas, local_host="0.0.0.0"), "agent", LOCAL)
        await call.hangup()
        return call.local_addr, parse_via(uas.transactions("INVITE")[0].headers.get("Via"))

    local_addr, via = run(scenario)
    assert local_addr[0] == "127.0.0.1"
    assert via.host == "127.0.0.1"


@pytest.mark.parametrize("code, header", [(401, "Authorization"), (407, "Proxy-Authorization")])
def test_digest_challenge_then_accept(code, header):
    seen = []

    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL, on_event=lambda n, **d: seen.append(n))
        await until(lambda: len(uas.transactions("ACK")) == 2)
        await call.hangup()
        return uas

    uas = run(scenario, challenges=[("n1", False)], challenge_code=code)
    assert "call_rejected" not in seen
    first, second = uas.transactions("INVITE")
    assert first.cseq == (1, "INVITE")
    assert second.cseq == (2, "INVITE")
    assert header not in first.headers
    assert uas.digest_ok(second, code)
    assert first.headers.get("Call-ID") == second.headers.get("Call-ID")
    assert first.headers.get("From") == second.headers.get("From")
    b1 = parse_via(first.headers.get("Via")).branch
    b2 = parse_via(second.headers.get("Via")).branch
    assert b1 != b2
    fail_ack, ok_ack = uas.transactions("ACK")
    assert parse_via(fail_ack.headers.get("Via")).branch == b1
    assert fail_ack.cseq == (1, "ACK")
    assert ok_ack.cseq == (2, "ACK")
    assert ok_ack.headers.get(header) == second.headers.get(header)
    assert uas.transactions("BYE")[0].cseq == (3, "BYE")


def test_events_count_each_invite_but_ringing_and_answer_once():
    seen = []

    async def scenario(uas):
        on_event = lambda name, **detail: seen.append((name, detail))  # noqa: E731
        call = await place_call(config(uas), "agent", LOCAL, on_event=on_event)
        await asyncio.sleep(0.2)  # let the resent 200s arrive
        await call.hangup()

    run(scenario, provisional=(100, 183, 180), challenges=[("n1", False)], retransmit_ok=2)
    assert seen == [
        ("invite_sent", {}),
        ("invite_sent", {}),
        ("ringing", {"status": 183}),
        ("answered", {}),
        ("bye_sent", {}),
    ]


def test_failed_call_reports_no_answer():
    seen = []

    async def scenario(uas):
        with pytest.raises(SipStatusError):
            await place_call(
                config(uas), "agent", LOCAL, on_event=lambda n, **d: seen.append((n, d))
            )

    run(scenario, final=486, final_reason="Busy Here")
    assert [name for name, _ in seen] == ["invite_sent", "ringing", "call_rejected"]
    assert seen[-1] == ("call_rejected", {"status": 486, "reason": "Busy Here"})


@pytest.mark.parametrize("challenges", [[("n1", False), ("n2", False)], [("n1", False)] * 2])
def test_rejected_credentials_fail(challenges):
    seen = []

    async def scenario(uas):
        with pytest.raises(SipStatusError) as info:
            await place_call(
                config(uas), "agent", LOCAL, on_event=lambda n, **d: seen.append((n, d))
            )
        await until(lambda: len(uas.transactions("ACK")) == 2)
        return uas, info.value

    uas, err = run(scenario, challenges=challenges)
    assert err.status == 401
    assert uas.invite_count == 2
    rejected = [detail for name, detail in seen if name == "call_rejected"]
    assert rejected == [{"status": 401, "reason": err.reason}]


def test_wrong_password_fails():
    async def scenario(uas):
        with pytest.raises(SipStatusError) as info:
            await place_call(config(uas, password="nope"), "agent", LOCAL)
        return info.value

    assert run(scenario, challenges=[("n1", False)]).status == 403


def test_stale_nonce_is_retried():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas

    uas = run(scenario, challenges=[("n1", False), ("n2", True)])
    assert [r.cseq[0] for r in uas.transactions("INVITE")] == [1, 2, 3]
    assert uas.digest_ok(uas.transactions("INVITE")[2], 401)


def test_endless_stale_challenges_stop():
    async def scenario(uas):
        with pytest.raises(SipStatusError):
            await place_call(config(uas), "agent", LOCAL)
        return uas

    uas = run(scenario, challenges=[("n1", False)] + [(f"s{i}", True) for i in range(10)])
    assert uas.invite_count == 5


def test_lost_invite_is_retransmitted():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas

    uas = run(scenario, drop_invites=1)
    invites = uas.requests("INVITE")
    assert len(invites) >= 2
    assert len({parse_via(i.headers.get("Via")).branch for i in invites}) == 1


def test_silent_server_times_out():
    async def scenario(uas):
        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(SipTimeoutError):
            await place_call(config(uas, timer_t1=0.005), "agent", LOCAL)
        return loop.time() - start, uas

    elapsed, uas = run(scenario, silent=True)
    assert elapsed >= 0.3
    assert len(uas.requests("INVITE")) >= 5


def test_busy_is_acked_and_raised():
    async def scenario(uas):
        with pytest.raises(SipStatusError) as info:
            await place_call(config(uas), "agent", LOCAL)
        await until(lambda: uas.transactions("ACK"))
        return uas, info.value

    uas, err = run(scenario, final=486, final_reason="Busy Here")
    assert (err.status, err.reason) == (486, "Busy Here")
    assert "486" in str(err)
    invite = one_transaction(uas.transactions("INVITE"))
    (ack,) = uas.transactions("ACK")
    assert parse_via(ack.headers.get("Via")).branch == parse_via(invite.headers.get("Via")).branch
    assert not uas.transactions("BYE")


def test_remote_hangup():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        assert not call.ended
        bye = uas.send_bye()
        uas.send(bye)
        await asyncio.wait_for(call.wait_ended(), 2)
        await until(lambda: len(uas.responses("BYE")) == 2)
        await call.hangup()
        return call, uas

    call, uas = run(scenario)
    assert call.ended
    assert [r.status for r in uas.responses("BYE")] == [200, 200]
    assert not uas.transactions("BYE")


def test_retransmitted_200_is_acked_again():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await until(lambda: len(uas.requests("ACK")) == 3)
        await call.hangup()
        return uas

    acks = run(scenario, retransmit_ok=2).requests("ACK")
    assert len({a.to_bytes() for a in acks}) == 1


def test_record_route_becomes_route_set():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas

    uas = run(scenario, record_route=["<sip:p1.test;lr>", "<sip:p2.test;lr>"])
    (ack,) = uas.transactions("ACK")
    bye = one_transaction(uas.transactions("BYE"))
    assert ack.headers.get_all("Route") == ["<sip:p2.test;lr>", "<sip:p1.test;lr>"]
    assert bye.headers.get_all("Route") == ["<sip:p2.test;lr>", "<sip:p1.test;lr>"]


@pytest.mark.parametrize(
    "answer, match",
    [(sdp_answer(port=0), "rejected"), ("", "audio"), (sdp_answer(pts="18"), "codec")],
)
def test_bad_sdp_answer_acks_and_hangs_up(answer, match):
    async def scenario(uas):
        with pytest.raises(SdpError, match=match):
            await place_call(config(uas), "agent", LOCAL)
        return uas

    uas = run(scenario, answer=answer)
    assert len(uas.transactions("ACK")) == 1
    one_transaction(uas.transactions("BYE"))


def test_bad_sdp_answer_hangs_up_once_when_bye_is_resent():
    async def scenario(uas):
        with pytest.raises(SdpError):
            await place_call(config(uas), "agent", LOCAL)
        return uas

    byes = run(scenario, answer=sdp_answer(port=0), drop_byes=1).requests("BYE")
    assert len(byes) == 2
    one_transaction(byes)


def test_lost_bye_is_retransmitted():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas

    byes = run(scenario, drop_byes=1).requests("BYE")
    assert len(byes) == 2
    assert byes[0].to_bytes() == byes[1].to_bytes()


def test_unanswered_bye_times_out_and_closes():
    async def scenario(uas):
        call = await place_call(config(uas, timer_t1=0.005, timer_t2=0.04), "agent", LOCAL)
        with pytest.raises(SipTimeoutError):
            await call.hangup()
        assert call.ended
        await call.hangup()
        return uas

    run(scenario, drop_byes=1000)


def test_challenged_bye_is_retried_with_credentials():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        await call.hangup()
        return uas

    uas = run(scenario, challenge_bye=True)
    first, second = uas.transactions("BYE")
    assert second.cseq[0] == first.cseq[0] + 1
    assert uas.digest_ok(second, 407)


def test_stray_messages_do_not_break_the_call():
    async def scenario(uas):
        call = await place_call(config(uas), "agent", LOCAL)
        addr = uas.peer
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            raw.sendto(b"\x00garbage", addr)
        finally:
            raw.close()
        stray = build_response(uas.dialog_request, 200, "OK")
        stray.headers.set("Via", f"SIP/2.0/UDP 127.0.0.1:1;branch={new_branch()}")
        uas.send(stray)
        options = Request("OPTIONS", f"sip:{USER}@127.0.0.1", Headers())
        for name, value in [
            ("Via", f"SIP/2.0/UDP 127.0.0.1:{uas.port};branch={new_branch()}"),
            ("From", "<sip:probe@x>;tag=p"),
            ("To", f"<sip:{USER}@127.0.0.1>"),
            ("Call-ID", "probe"),
            ("CSeq", "1 OPTIONS"),
        ]:
            options.headers.add(name, value)
        uas.send(options)
        uas.send_bye(from_tag="wrongtag")
        await until(lambda: uas.responses("OPTIONS") and uas.responses("BYE"))
        assert not call.ended
        uas.send_bye()
        await asyncio.wait_for(call.wait_ended(), 2)
        await call.hangup()
        return uas

    uas = run(scenario)
    assert uas.responses("OPTIONS")[0].status == 200
    assert [r.status for r in uas.responses("BYE")] == [481, 200]
