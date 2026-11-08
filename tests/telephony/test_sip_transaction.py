import asyncio
import socket

import pytest

from voiceqa.telephony.sip.dialog import build_invite, via_value
from voiceqa.telephony.sip.errors import SipTimeoutError
from voiceqa.telephony.sip.message import Headers, Request, build_response, parse_via
from voiceqa.telephony.sip.transaction import (
    InviteClientTransaction,
    NonInviteClientTransaction,
    ack_for_failure,
)
from voiceqa.telephony.sip.transport import UdpTransport

BRANCH = "z9hG4bKtx1"
# Windows timers tick about every 16 ms, so lower bounds get absolute slack.
SLACK = 0.02


def invite():
    req = build_invite(
        target_uri="sip:agent@pbx",
        from_uri="sip:me@pbx",
        from_tag="ft",
        display_name=None,
        call_id="cid",
        cseq=1,
        contact_uri="sip:me@127.0.0.1:5999",
        via=via_value("127.0.0.1", 5999, BRANCH),
        sdp="",
        user_agent="t",
    )
    req.headers.add("Route", "<sip:proxy;lr>")
    return req


def bye():
    h = Headers()
    h.add("Via", via_value("127.0.0.1", 5999, "z9hG4bKbye"))
    h.add("From", "<sip:me@pbx>;tag=ft")
    h.add("To", "<sip:agent@pbx>;tag=rt")
    h.add("Call-ID", "cid")
    h.add("CSeq", "2 BYE")
    return Request("BYE", "sip:agent@pbx", h)


def reply(req, status, reason="X"):
    resp = build_response(req, status, reason)
    if status > 100:
        resp.headers.set("To", resp.headers.get("To").split(";tag=")[0] + ";tag=rt")
    return resp


class Recorder:
    def __init__(self):
        self.sent = []

    def __call__(self, req):
        self.sent.append((asyncio.get_running_loop().time(), req))

    @property
    def methods(self):
        return [r.method for _, r in self.sent]

    def gaps(self):
        times = [t for t, _ in self.sent]
        return [b - a for a, b in zip(times, times[1:], strict=False)]

    async def wait_for(self, count, timeout=3.0):
        async with asyncio.timeout(timeout):
            while len(self.sent) < count:
                await asyncio.sleep(0.002)


def test_invite_retransmits_with_doubling_until_provisional():
    async def main():
        rec = Recorder()
        tx = InviteClientTransaction(invite(), rec, t1=0.05)
        tx.start()
        await rec.wait_for(4)
        tx.receive(reply(tx.request, 100, "Trying"))
        await asyncio.sleep(0.3)
        tx.close()
        return rec, tx

    rec, tx = asyncio.run(main())
    assert rec.methods == ["INVITE"] * 4
    gaps = rec.gaps()
    for gap, expected in zip(gaps, [0.05, 0.1, 0.2], strict=True):
        assert gap >= expected - SLACK
    assert [r.status for r in tx.provisional] == [100]


def test_invite_timer_b_times_out():
    async def main():
        rec = Recorder()
        tx = InviteClientTransaction(invite(), rec, t1=0.01)
        tx.start()
        start = asyncio.get_running_loop().time()
        with pytest.raises(SipTimeoutError):
            await tx.final()
        elapsed = asyncio.get_running_loop().time() - start
        tx.close()
        return rec, elapsed

    rec, elapsed = asyncio.run(main())
    assert elapsed >= 0.6
    assert len(rec.sent) >= 5


def test_invite_failure_is_acked_on_same_branch():
    async def main():
        rec = Recorder()
        tx = InviteClientTransaction(invite(), rec, t1=0.05)
        tx.start()
        tx.receive(reply(tx.request, 486, "Busy Here"))
        final = await tx.final()
        tx.receive(reply(tx.request, 486, "Busy Here"))
        await asyncio.sleep(0.15)
        tx.close()
        return rec, final

    rec, final = asyncio.run(main())
    assert final.status == 486
    assert rec.methods == ["INVITE", "ACK", "ACK"]
    ack = rec.sent[1][1]
    assert parse_via(ack.headers.get("Via")).branch == BRANCH
    assert ack.cseq == (1, "ACK")
    assert ack.headers.get("To").endswith(";tag=rt")
    assert ack.uri == "sip:agent@pbx"


def test_ack_for_failure_fields():
    inv = invite()
    ack = ack_for_failure(inv, reply(inv, 503, "Unavailable"))
    assert ack.headers.get_all("Via") == [inv.headers.get("Via")]
    assert ack.headers.get("Call-ID") == "cid"
    assert ack.headers.get("From") == inv.headers.get("From")
    assert ack.headers.get_all("Route") == ["<sip:proxy;lr>"]
    assert ack.headers.get("Max-Forwards") == "70"
    assert ack.body == b""


def test_retransmitted_2xx_goes_to_callback_and_lingers():
    async def main():
        rec = Recorder()
        seen = []
        terminated = []
        tx = InviteClientTransaction(invite(), rec, t1=0.005, on_retransmitted_2xx=seen.append)
        tx.on_terminated = lambda: terminated.append(True)
        tx.start()
        tx.receive(reply(tx.request, 180, "Ringing"))
        tx.receive(reply(tx.request, 200, "OK"))
        final = await tx.final()
        tx.receive(reply(tx.request, 200, "OK"))
        assert not terminated
        await asyncio.sleep(0.4)
        tx.receive(reply(tx.request, 200, "OK"))
        tx.close()
        return rec, final, seen, terminated

    rec, final, seen, terminated = asyncio.run(main())
    assert final.status == 200
    assert rec.methods == ["INVITE"]
    assert len(seen) == 1
    assert terminated == [True]


def test_non_invite_retransmits_capped_at_t2():
    async def main():
        rec = Recorder()
        tx = NonInviteClientTransaction(bye(), rec, t1=0.04, t2=0.16)
        tx.start()
        await rec.wait_for(6)
        tx.receive(reply(tx.request, 200, "OK"))
        final = await tx.final()
        await asyncio.sleep(0.2)
        tx.close()
        return rec, final

    rec, final = asyncio.run(main())
    assert final.status == 200
    assert rec.methods == ["BYE"] * 6
    gaps = rec.gaps()
    assert gaps[0] >= 0.04 - SLACK
    assert gaps[1] >= 0.08 - SLACK
    assert all(g < 0.25 for g in gaps[3:])


def test_non_invite_uses_t2_after_provisional():
    async def main():
        rec = Recorder()
        tx = NonInviteClientTransaction(bye(), rec, t1=0.02, t2=0.1)
        tx.start()
        tx.receive(reply(tx.request, 100, "Trying"))
        await rec.wait_for(3)
        tx.close()
        return rec

    gaps = asyncio.run(main()).gaps()
    assert gaps[1] >= 0.1 - SLACK


def test_non_invite_timer_f_times_out():
    async def main():
        tx = NonInviteClientTransaction(bye(), Recorder(), t1=0.005, t2=0.02)
        tx.start()
        with pytest.raises(SipTimeoutError):
            await tx.final()
        tx.close()

    asyncio.run(main())


def test_udp_transport_round_trip_and_garbage_dropped():
    async def main():
        got = []
        a = await UdpTransport.open("127.0.0.1", 0, lambda m, addr: None)
        b = await UdpTransport.open("127.0.0.1", 0, lambda m, addr: got.append((m, addr)))
        raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            raw.sendto(b"not sip at all", b.local_addr)
            await asyncio.sleep(0.05)
            a.send(bye(), b.local_addr)
            async with asyncio.timeout(2):
                while not got:
                    await asyncio.sleep(0.005)
        finally:
            raw.close()
            a.close()
            b.close()
            await asyncio.sleep(0)
        return got, a.local_addr

    got, a_addr = asyncio.run(main())
    assert len(got) == 1
    msg, addr = got[0]
    assert msg.method == "BYE"
    assert addr == a_addr
