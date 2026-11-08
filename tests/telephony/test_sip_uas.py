import asyncio

import pytest

from voiceqa.telephony.media import LocalMedia, PayloadType
from voiceqa.telephony.sip import SipConfig, SipStatusError, place_call
from voiceqa.telephony.sip.dialog import build_invite, via_value
from voiceqa.telephony.sip.message import (
    Headers,
    Request,
    Response,
    new_branch,
    parse_message,
    parse_name_addr,
)
from voiceqa.telephony.sip.sdp import build_offer, parse_answer
from voiceqa.telephony.sip.uas import ServerConfig, SipServer

CALLER_MEDIA = LocalMedia(host="127.0.0.1", port=40000)
T1 = 0.02


class FakeMedia:
    def __init__(self, host, port, payload_types):
        self.local = LocalMedia(host=host, port=port, payload_types=payload_types)
        self.remote = None
        self.closed = False

    async def start(self, remote):
        self.remote = remote

    def send(self, samples):
        pass

    async def drain(self):
        pass

    def recording(self):
        return None

    async def close(self):
        self.closed = True


class Agent:
    """A SipServer with fake media that records what happened to each call."""

    def __init__(self, behaviour=None, media_types=(PayloadType.PCMU, PayloadType.PCMA)):
        self.media_types = media_types
        self.media = []
        self.calls = []
        self.behaviour = behaviour

    async def open_media(self, host):
        m = FakeMedia(host, 30000 + len(self.media), self.media_types)
        self.media.append(m)
        return m

    async def on_call(self, call):
        self.calls.append(call)
        if self.behaviour is not None:
            await self.behaviour(call)


def server_config(**kw):
    args = dict(host="127.0.0.1", port=0, timer_t1=T1, timer_t2=0.16)
    args.update(kw)
    return ServerConfig(**args)


def caller_config(server, **kw):
    args = dict(
        server_host="127.0.0.1",
        server_port=server.local_addr[1],
        username="1001",
        local_host="127.0.0.1",
        timer_t1=T1,
        timer_t2=0.16,
    )
    args.update(kw)
    return SipConfig(**args)


async def until(cond, timeout=3.0):
    async with asyncio.timeout(timeout):
        while not cond():
            await asyncio.sleep(0.005)


def run(scenario, behaviour=None, media_types=(PayloadType.PCMU, PayloadType.PCMA), **cfg):
    async def main():
        agent = Agent(behaviour, media_types)
        server = await SipServer.open(server_config(**cfg), agent.open_media, agent.on_call)
        try:
            return await scenario(server, agent)
        finally:
            await server.close()

    return asyncio.run(main())


class RawPeer(asyncio.DatagramProtocol):
    """A bare UDP endpoint for driving the server one message at a time."""

    def __init__(self):
        self.received = []

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        self.received.append(parse_message(data))

    @property
    def port(self):
        return self.transport.get_extra_info("sockname")[1]

    def responses(self, method, status=None):
        return [
            m
            for m in self.received
            if isinstance(m, Response)
            and m.cseq[1] == method
            and (status is None or m.status == status)
        ]

    def requests(self, method):
        return [m for m in self.received if isinstance(m, Request) and m.method == method]


async def raw_peer():
    loop = asyncio.get_running_loop()
    _, peer = await loop.create_datagram_endpoint(RawPeer, local_addr=("127.0.0.1", 0))
    return peer


def raw_invite(peer, server, *, offer=None, call_id="rawcall", branch=None):
    return build_invite(
        target_uri=f"sip:agent@127.0.0.1:{server.local_addr[1]}",
        from_uri="sip:1001@127.0.0.1",
        from_tag="rawtag",
        display_name=None,
        call_id=call_id,
        cseq=1,
        contact_uri=f"sip:1001@127.0.0.1:{peer.port}",
        via=via_value("127.0.0.1", peer.port, branch or new_branch()),
        sdp=build_offer(CALLER_MEDIA) if offer is None else offer,
        user_agent="raw",
    )


def raw_request(peer, method, *, call_id="nocall", to_tag=None, branch=None):
    h = Headers()
    h.add("Via", via_value("127.0.0.1", peer.port, branch or new_branch()))
    h.add("From", "<sip:1001@127.0.0.1>;tag=rawtag")
    h.add("To", "<sip:agent@127.0.0.1>" + (f";tag={to_tag}" if to_tag else ""))
    h.add("Call-ID", call_id)
    h.add("CSeq", f"5 {method}")
    return Request(method, "sip:agent@127.0.0.1", h)


def in_dialog(peer, ok, method, cseq):
    h = Headers()
    h.add("Via", via_value("127.0.0.1", peer.port, new_branch()))
    h.add("From", ok.headers.get("From"))
    h.add("To", ok.headers.get("To"))
    h.add("Call-ID", ok.headers.get("Call-ID"))
    h.add("CSeq", f"{cseq} {method}")
    return Request(method, parse_name_addr(ok.headers.get("Contact")).uri, h)


def send(peer, server, msg):
    peer.transport.sendto(msg.to_bytes(), ("127.0.0.1", server.local_addr[1]))


def test_answers_a_call_with_an_sdp_answer_for_the_offered_codec():
    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.calls)
        (media,) = agent.media
        assert call.remote_media.port == media.local.port
        assert call.remote_media.payload_type is PayloadType.PCMU
        assert media.remote.port == CALLER_MEDIA.port
        assert media.remote.payload_type is PayloadType.PCMU
        assert agent.calls[0].remote_media == media.remote
        await call.hangup()

    run(scenario)


def test_sends_trying_and_ringing_before_ok():
    async def scenario(server, agent):
        peer = await raw_peer()
        invite = raw_invite(peer, server)
        send(peer, server, invite)
        await until(lambda: peer.responses("INVITE", 200))
        statuses = [r.status for r in peer.responses("INVITE")]
        assert statuses[:3] == [100, 180, 200]
        ok = peer.responses("INVITE", 200)[0]
        assert parse_name_addr(ok.headers.get("To")).tag
        assert ok.headers.get("Content-Type") == "application/sdp"
        assert parse_answer(ok.body, CALLER_MEDIA.payload_types).port == agent.media[0].local.port
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        peer.transport.close()

    run(scenario)


def test_caller_bye_ends_the_call_and_closes_media():
    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.calls)
        await call.hangup()
        await until(lambda: agent.calls[0].ended)
        assert agent.media[0].closed
        assert server.calls == []

    run(scenario)


def test_agent_bye_ends_the_call_on_both_sides():
    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.calls)
        await agent.calls[0].hangup()
        async with asyncio.timeout(3):
            await call.wait_ended()
        assert agent.media[0].closed
        await call.hangup()

    run(scenario)


def test_behaviour_can_hang_up_by_itself():
    async def hang_up_at_once(call):
        await call.hangup()

    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        async with asyncio.timeout(3):
            await call.wait_ended()
        await until(lambda: agent.media[0].closed)
        await call.hangup()

    run(scenario, hang_up_at_once)


def test_behaviour_is_cancelled_when_the_caller_hangs_up():
    cancelled = []

    async def forever(call):
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.calls)
        await call.hangup()
        await until(lambda: cancelled)

    run(scenario, forever)


def test_no_common_codec_is_rejected_with_488_and_no_media_is_opened():
    async def scenario(server, agent):
        pcma_only = LocalMedia(host="127.0.0.1", port=40000, payload_types=(PayloadType.PCMA,))
        with pytest.raises(SipStatusError) as err:
            await place_call(caller_config(server), "agent", pcma_only)
        assert err.value.status == 488
        assert agent.media == []

    run(scenario, payload_types=(PayloadType.PCMU,))


def test_invite_without_sdp_is_rejected_with_488():
    async def scenario(server, agent):
        peer = await raw_peer()
        invite = raw_invite(peer, server)
        invite.body = b""
        send(peer, server, invite)
        await until(lambda: peer.responses("INVITE", 488))
        assert agent.media == []
        peer.transport.close()

    run(scenario)


def test_ok_is_resent_until_ack_arrives():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server))
        await until(lambda: len(peer.responses("INVITE", 200)) >= 3)
        ok = peer.responses("INVITE", 200)[0]
        assert all(r.to_bytes() == ok.to_bytes() for r in peer.responses("INVITE", 200))
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        await asyncio.sleep(0.05)
        count = len(peer.responses("INVITE", 200))
        await asyncio.sleep(0.4)  # well past the T2 cap of 0.16 s
        assert len(peer.responses("INVITE", 200)) == count
        assert not agent.calls[0].ended
        peer.transport.close()

    run(scenario)


def test_missing_ack_ends_the_call_with_bye():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server))
        await until(lambda: peer.requests("BYE"), timeout=64 * T1 + 3)
        await until(lambda: agent.media[0].closed)
        assert server.calls == []
        peer.transport.close()

    run(scenario)


def test_resent_invite_gets_the_same_answer_and_one_call():
    async def scenario(server, agent):
        peer = await raw_peer()
        invite = raw_invite(peer, server)
        send(peer, server, invite)
        await until(lambda: peer.responses("INVITE", 200))
        send(peer, server, invite)
        await until(lambda: len(peer.responses("INVITE", 200)) >= 2)
        oks = {r.to_bytes() for r in peer.responses("INVITE", 200)}
        assert len(oks) == 1
        assert len(agent.media) == 1
        ok = peer.responses("INVITE", 200)[0]
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        peer.transport.close()

    run(scenario)


def test_resent_bye_gets_200_again():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server))
        await until(lambda: peer.responses("INVITE", 200))
        ok = peer.responses("INVITE", 200)[0]
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        bye = in_dialog(peer, ok, "BYE", 2)
        send(peer, server, bye)
        await until(lambda: peer.responses("BYE"))
        send(peer, server, bye)
        await until(lambda: len(peer.responses("BYE")) >= 2)
        assert [r.status for r in peer.responses("BYE")] == [200, 200]
        peer.transport.close()

    run(scenario)


def test_same_call_id_from_a_new_transaction_is_a_merged_request():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server))
        await until(lambda: peer.responses("INVITE", 200))
        send(peer, server, raw_invite(peer, server))
        await until(lambda: peer.responses("INVITE", 482))
        assert len(agent.media) == 1
        ok = peer.responses("INVITE", 200)[0]
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        peer.transport.close()

    run(scenario)


@pytest.mark.parametrize(
    "method, to_tag, status",
    [
        ("BYE", "x", 481),
        ("OPTIONS", None, 200),
        ("CANCEL", None, 481),
        ("INVITE", "x", 481),
        ("REGISTER", None, 501),
    ],
)
def test_out_of_dialog_requests(method, to_tag, status):
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_request(peer, method, to_tag=to_tag))
        await until(lambda: peer.responses(method))
        resp = peer.responses(method)[0]
        assert resp.status == status
        assert parse_name_addr(resp.headers.get("To")).tag
        peer.transport.close()

    run(scenario)


def test_cancel_after_answer_gets_200_and_the_call_stays_up():
    async def scenario(server, agent):
        peer = await raw_peer()
        branch = new_branch()
        invite = raw_invite(peer, server, branch=branch)
        send(peer, server, invite)
        await until(lambda: peer.responses("INVITE", 200))
        cancel = raw_request(peer, "CANCEL", call_id="rawcall", branch=branch)
        send(peer, server, cancel)
        await until(lambda: peer.responses("CANCEL"))
        assert peer.responses("CANCEL")[0].status == 200
        ok = peer.responses("INVITE", 200)[0]
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        assert len(server.calls) == 1
        peer.transport.close()

    run(scenario)


def test_reinvite_in_dialog_is_refused_without_ending_the_call():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server))
        await until(lambda: peer.responses("INVITE", 200))
        ok = peer.responses("INVITE", 200)[0]
        send(peer, server, in_dialog(peer, ok, "ACK", 1))
        reinvite = in_dialog(peer, ok, "INVITE", 2)
        send(peer, server, reinvite)
        await until(lambda: peer.responses("INVITE", 488))
        assert not agent.calls[0].ended
        peer.transport.close()

    run(scenario)


def test_two_calls_at_once():
    async def scenario(server, agent):
        cfg = caller_config(server)
        a, b = await asyncio.gather(
            place_call(cfg, "agent", CALLER_MEDIA),
            place_call(cfg, "agent", LocalMedia(host="127.0.0.1", port=40002)),
        )
        await until(lambda: len(agent.calls) == 2)
        assert {a.remote_media.port, b.remote_media.port} == {m.local.port for m in agent.media}
        await a.hangup()
        await until(lambda: len(server.calls) == 1)
        await b.hangup()
        await until(lambda: server.calls == [])

    run(scenario)


def test_close_hangs_up_active_calls():
    async def main():
        agent = Agent()
        server = await SipServer.open(server_config(), agent.open_media, agent.on_call)
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.calls)
        await server.close()
        async with asyncio.timeout(3):
            await call.wait_ended()
        assert agent.media[0].closed
        await call.hangup()

    asyncio.run(main())


def test_contact_and_media_use_the_bind_host():
    async def scenario(server, agent):
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA)
        await until(lambda: agent.media)
        assert agent.media[0].local.host == "127.0.0.1"
        assert call.remote_media.host == "127.0.0.1"
        await call.hangup()

    run(scenario)


def test_media_without_the_offered_codec_gets_500_and_is_closed():
    async def scenario(server, agent):
        peer = await raw_peer()
        pcma_only = LocalMedia(host="127.0.0.1", port=40000, payload_types=(PayloadType.PCMA,))
        send(peer, server, raw_invite(peer, server, offer=build_offer(pcma_only)))
        await until(lambda: peer.responses("INVITE", 500))
        await until(lambda: agent.media and agent.media[0].closed)
        assert server.calls == []
        peer.transport.close()

    run(scenario, media_types=(PayloadType.PCMU,))
