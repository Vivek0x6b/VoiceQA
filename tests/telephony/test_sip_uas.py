import asyncio

import pytest

from voiceqa.telephony.media import LocalMedia, PayloadType, RemoteMedia
from voiceqa.telephony.sip import SipConfig, SipStatusError, place_call
from voiceqa.telephony.sip.dialog import build_invite, via_value
from voiceqa.telephony.sip.message import (
    Headers,
    Request,
    Response,
    build_response,
    new_branch,
    parse_message,
    parse_name_addr,
)
from voiceqa.telephony.sip.sdp import build_answer, build_offer, parse_answer, parse_offer
from voiceqa.telephony.sip.uas import ServerConfig, SipServer

CALLER_MEDIA = LocalMedia(host="127.0.0.1", port=40000)
T1 = 0.02


class FakeMedia:
    def __init__(self, host, port, payload_types):
        self.local = LocalMedia(host=host, port=port, payload_types=payload_types)
        self.remote = None
        self.closed = False
        self.starts = 0

    async def start(self, remote):
        self.remote = remote
        self.starts += 1

    def send(self, samples):
        pass

    async def drain(self):
        pass

    def record(self, sink):
        pass

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


class Events:
    def __init__(self):
        self.seen = []

    def __call__(self, name, **detail):
        self.seen.append((name, detail))

    @property
    def names(self):
        return [name for name, _ in self.seen]


def test_caller_reports_call_events():
    async def scenario(server, agent):
        events = Events()
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA, on_event=events)
        await until(lambda: agent.calls)
        await call.hangup()
        return events.seen

    seen = run(scenario)
    assert [name for name, _ in seen] == ["invite_sent", "ringing", "answered", "bye_sent"]
    assert seen[1][1] == {"status": 180}


def test_caller_reports_bye_from_the_far_end():
    async def hang_up(call):
        await asyncio.sleep(0.05)
        await call.hangup()

    async def scenario(server, agent):
        events = Events()
        call = await place_call(caller_config(server), "agent", CALLER_MEDIA, on_event=events)
        async with asyncio.timeout(3):
            await call.wait_ended()
        await call.hangup()
        return events.names

    names = run(scenario, behaviour=hang_up)
    assert names == ["invite_sent", "ringing", "answered", "bye_received"]


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


def offerless_invite(peer, server, content_type=None, body=b""):
    invite = raw_invite(peer, server)
    invite.headers.remove("Content-Type")
    if content_type is not None:
        invite.headers.add("Content-Type", content_type)
    invite.body = body
    return invite


def answer_bye(peer, server):
    bye = peer.requests("BYE")[0]
    send(peer, server, build_response(bye, 200, "OK"))


@pytest.mark.parametrize(
    "content_type, body",
    [(None, b""), ("text/plain", b"hello"), ("application/sdp", b"")],
)
def test_invite_without_sdp_gets_an_offer_and_the_answer_comes_in_the_ack(content_type, body):
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, offerless_invite(peer, server, content_type, body))
        await until(lambda: peer.responses("INVITE", 200))
        assert [r.status for r in peer.responses("INVITE")][:3] == [100, 180, 200]
        ok = peer.responses("INVITE", 200)[0]
        assert ok.headers.get("Content-Type") == "application/sdp"
        (media,) = agent.media
        offer = parse_offer(ok.body, (PayloadType.PCMU,))
        assert (offer.host, offer.port) == (media.local.host, media.local.port)
        assert parse_offer(ok.body, (PayloadType.PCMA,)).payload_type is PayloadType.PCMA
        # Nothing starts before the answer arrives.
        await asyncio.sleep(0.1)
        assert media.starts == 0
        assert agent.calls == []

        answer = build_answer(CALLER_MEDIA, PayloadType.PCMA)
        send(peer, server, with_sdp(in_dialog(peer, ok, "ACK", 1), answer))
        await until(lambda: agent.calls)
        expected = RemoteMedia("127.0.0.1", CALLER_MEDIA.port, PayloadType.PCMA)
        assert media.remote == expected
        assert media.starts == 1
        assert agent.calls[0].remote_media == expected

        # The 200 stops being resent once the ACK is in.
        await asyncio.sleep(0.05)
        count = len(peer.responses("INVITE", 200))
        await asyncio.sleep(0.4)  # well past the T2 cap of 0.16 s
        assert len(peer.responses("INVITE", 200)) == count

        # A refresh gets our offer from the 200 as the current SDP.
        send(peer, server, with_sdp(in_dialog(peer, ok, "INVITE", 2), build_offer(CALLER_MEDIA)))
        await until(lambda: answers(peer, "INVITE", 2))
        assert answers(peer, "INVITE", 2)[0].body == ok.body
        send(peer, server, in_dialog(peer, ok, "ACK", 2))
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 3)
        assert answers(peer, "BYE", 3)[0].status == 200

    run(scenario)


NO_COMMON_CODEC = (
    "v=0\r\no=raw 1 1 IN IP4 127.0.0.1\r\ns=raw\r\nc=IN IP4 127.0.0.1\r\nt=0 0\r\n"
    "m=audio 40000 RTP/AVP 18\r\na=rtpmap:18 G729/8000\r\n"
)


@pytest.mark.parametrize("answer", [None, NO_COMMON_CODEC, "not sdp at all"])
def test_ack_without_a_usable_answer_ends_the_offerless_call_with_bye(answer):
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, offerless_invite(peer, server))
        await until(lambda: peer.responses("INVITE", 200))
        ok = peer.responses("INVITE", 200)[0]
        ack = in_dialog(peer, ok, "ACK", 1)
        if answer is not None:
            ack = with_sdp(ack, answer)
        send(peer, server, ack)
        await until(lambda: peer.requests("BYE"))
        answer_bye(peer, server)
        (media,) = agent.media
        await until(lambda: media.closed and server.calls == [])
        assert media.starts == 0
        assert agent.calls == []
        peer.transport.close()

    run(scenario)


def test_missing_ack_ends_the_offerless_call_with_bye():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, offerless_invite(peer, server))
        await until(lambda: peer.requests("BYE"), timeout=64 * T1 + 3)
        answer_bye(peer, server)
        (media,) = agent.media
        await until(lambda: media.closed and server.calls == [])
        assert media.starts == 0
        assert agent.calls == []
        peer.transport.close()

    run(scenario)


def test_garbage_sdp_offer_is_rejected_with_488():
    async def scenario(server, agent):
        peer = await raw_peer()
        send(peer, server, raw_invite(peer, server, offer="not sdp at all"))
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
        ("UPDATE", "x", 481),
        ("UPDATE", None, 481),
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


async def answered_raw_call(server, agent):
    peer = await raw_peer()
    send(peer, server, raw_invite(peer, server))
    await until(lambda: peer.responses("INVITE", 200))
    ok = peer.responses("INVITE", 200)[0]
    send(peer, server, in_dialog(peer, ok, "ACK", 1))
    await until(lambda: agent.calls)
    return peer, ok


def with_sdp(req, sdp):
    req.headers.add("Contact", "<sip:1001@127.0.0.1>")
    req.headers.add("Content-Type", "application/sdp")
    req.body = sdp.encode()
    return req


def answers(peer, method, cseq, status=200):
    return [r for r in peer.responses(method, status) if r.cseq[0] == cseq]


async def hang_up_raw(peer, server, ok, cseq):
    send(peer, server, in_dialog(peer, ok, "BYE", cseq))
    await until(lambda: answers(peer, "BYE", cseq))
    peer.transport.close()


def assert_media_untouched(agent, call):
    (media,) = agent.media
    assert media.starts == 1
    assert not media.closed
    assert not call.ended


def test_reinvite_with_offer_gets_the_same_sdp_until_acked():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        send(peer, server, with_sdp(in_dialog(peer, ok, "INVITE", 2), build_offer(CALLER_MEDIA)))
        await until(lambda: len(answers(peer, "INVITE", 2)) >= 3)
        first = answers(peer, "INVITE", 2)[0]
        assert first.body == ok.body
        assert first.headers.get("Content-Type") == "application/sdp"
        assert first.headers.get("Contact") == ok.headers.get("Contact")
        assert "UPDATE" in first.headers.get("Allow")
        assert first.headers.get("To") == ok.headers.get("To")
        assert all(r.to_bytes() == first.to_bytes() for r in answers(peer, "INVITE", 2))
        send(peer, server, in_dialog(peer, ok, "ACK", 2))
        await asyncio.sleep(0.05)
        count = len(answers(peer, "INVITE", 2))
        await asyncio.sleep(0.4)  # well past the T2 cap of 0.16 s
        assert len(answers(peer, "INVITE", 2)) == count
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 3)
        assert answers(peer, "BYE", 3)[0].status == 200

    run(scenario)


def test_reinvite_without_sdp_gets_our_sdp_as_an_offer():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        send(peer, server, in_dialog(peer, ok, "INVITE", 2))
        await until(lambda: answers(peer, "INVITE", 2))
        offer = answers(peer, "INVITE", 2)[0]
        assert offer.body == ok.body
        assert offer.headers.get("Content-Type") == "application/sdp"
        ack = with_sdp(in_dialog(peer, ok, "ACK", 2), build_offer(CALLER_MEDIA))
        send(peer, server, ack)
        await asyncio.sleep(0.05)
        count = len(answers(peer, "INVITE", 2))
        await asyncio.sleep(0.4)
        assert len(answers(peer, "INVITE", 2)) == count
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 3)

    run(scenario)


def test_update_with_and_without_sdp_gets_200():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        send(peer, server, with_sdp(in_dialog(peer, ok, "UPDATE", 2), build_offer(CALLER_MEDIA)))
        await until(lambda: answers(peer, "UPDATE", 2))
        with_body = answers(peer, "UPDATE", 2)[0]
        assert with_body.body == ok.body
        assert with_body.headers.get("Content-Type") == "application/sdp"
        assert with_body.headers.get("Contact") == ok.headers.get("Contact")
        send(peer, server, in_dialog(peer, ok, "UPDATE", 3))
        await until(lambda: answers(peer, "UPDATE", 3))
        bare = answers(peer, "UPDATE", 3)[0]
        assert bare.body == b""
        assert "Content-Type" not in bare.headers
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 4)

    run(scenario)


def test_reinvite_without_a_common_codec_gets_488_and_the_call_stays_up():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        pcma_only = LocalMedia(host="127.0.0.1", port=40000, payload_types=(PayloadType.PCMA,))
        send(peer, server, with_sdp(in_dialog(peer, ok, "INVITE", 2), build_offer(pcma_only)))
        await until(lambda: peer.responses("INVITE", 488))
        assert not answers(peer, "INVITE", 2)
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 3)
        assert answers(peer, "BYE", 3)[0].status == 200

    run(scenario, payload_types=(PayloadType.PCMU,))


def test_reinvite_with_wrong_tags_gets_481():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        stray = raw_request(peer, "INVITE", call_id=ok.headers.get("Call-ID"), to_tag="wrong")
        send(peer, server, stray)
        await until(lambda: peer.responses("INVITE", 481))
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 6)

    run(scenario)


def test_reinvite_with_an_old_cseq_gets_500():
    async def scenario(server, agent):
        peer, ok = await answered_raw_call(server, agent)
        send(peer, server, with_sdp(in_dialog(peer, ok, "INVITE", 1), build_offer(CALLER_MEDIA)))
        await until(lambda: peer.responses("INVITE", 500))
        assert_media_untouched(agent, agent.calls[0])
        await hang_up_raw(peer, server, ok, 2)

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
