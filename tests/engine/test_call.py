import asyncio
import json
import socket

from voiceqa.audio.dsp import generate_tone
from voiceqa.engine.call import run_call
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.media import PayloadType
from voiceqa.telephony.sip.message import build_response, new_tag, parse_message


async def start_agent(**kw):
    args = dict(host="127.0.0.1", port=0, timer_t1=0.02)
    args.update(kw)
    return await LocalAgent.start(AgentConfig(**args))


def test_call_against_local_echo_agent_answers_and_records(tmp_path):
    async def main():
        agent = await start_agent()
        try:
            return await run_call(
                target="agent",
                server_host="127.0.0.1",
                server_port=agent.local_addr[1],
                username="1001",
                password="",
                audio=generate_tone(440, 1.0),
                record_dir=tmp_path,
                duration=1.5,
            )
        finally:
            await agent.close()

    result = asyncio.run(main())

    assert result.answered
    assert result.status == 200
    assert result.frames_sent > 0
    assert result.frames_received > 0
    assert result.first_audio_latency is not None
    assert result.first_audio_latency < 1.0
    for name in ("caller.wav", "agent.wav", "mix.wav", "events.json"):
        assert (tmp_path / name).exists()


class _RejectingUas(asyncio.DatagramProtocol):
    """Answers every INVITE with 100 Trying then a final rejection. Kept small
    on purpose: just enough SIP to make place_call raise SipStatusError."""

    def __init__(self, status=486, reason="Busy Here"):
        self.status = status
        self.reason = reason

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        req = parse_message(data)
        if req.method != "INVITE":
            return
        for status, reason in ((100, "Trying"), (self.status, self.reason)):
            resp = build_response(req, status, reason)
            if status > 100:
                to = req.headers.get("To") or ""
                resp.headers.set("To", f"{to};tag={new_tag()}")
            self.transport.sendto(resp.to_bytes(), addr)

    def error_received(self, exc):
        pass


async def start_rejecting_uas(**kw):
    loop = asyncio.get_running_loop()
    _, proto = await loop.create_datagram_endpoint(
        lambda: _RejectingUas(**kw), local_addr=("127.0.0.1", 0)
    )
    return proto


def test_call_rejected_by_the_far_end_reports_status_and_still_records(tmp_path):
    async def main():
        uas = await start_rejecting_uas(status=486, reason="Busy Here")
        try:
            port = uas.transport.get_extra_info("sockname")[1]
            return await run_call(
                target="agent",
                server_host="127.0.0.1",
                server_port=port,
                username="1001",
                password="",
                audio=generate_tone(440, 0.5),
                record_dir=tmp_path,
                duration=1.5,
            )
        finally:
            uas.transport.close()

    result = asyncio.run(main())

    assert result.answered is False
    assert result.status == 486
    assert result.reason == "Busy Here"
    assert result.duration is None
    assert result.first_audio_latency is None
    assert (tmp_path / "events.json").exists()
    events = json.loads((tmp_path / "events.json").read_text())
    names = [e["event"] for e in events["events"]]
    assert "call_rejected" in names


class _BadSdpUas(asyncio.DatagramProtocol):
    """Answers INVITE with 200 OK offering a codec we never offered, and
    answers BYE right away so the call does not have to wait out a SIP
    retransmission timer while cleaning up after the bad answer."""

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        req = parse_message(data)
        if req.method == "BYE":
            self.transport.sendto(build_response(req, 200, "OK").to_bytes(), addr)
            return
        if req.method != "INVITE":
            return
        self.transport.sendto(build_response(req, 100, "Trying").to_bytes(), addr)
        ok = build_response(req, 200, "OK")
        to = req.headers.get("To") or ""
        ok.headers.set("To", f"{to};tag={new_tag()}")
        ok.headers.set("Content-Type", "application/sdp")
        ok.headers.set("Contact", f"<sip:uas@{addr[0]}:{addr[1]}>")
        ok.body = (
            b"v=0\r\n"
            b"o=uas 1 1 IN IP4 127.0.0.1\r\n"
            b"s=uas\r\n"
            b"c=IN IP4 127.0.0.1\r\n"
            b"t=0 0\r\n"
            b"m=audio 40000 RTP/AVP 3\r\n"
            b"a=rtpmap:3 GSM/8000\r\n"
        )
        self.transport.sendto(ok.to_bytes(), addr)

    def error_received(self, exc):
        pass


async def start_bad_sdp_uas():
    loop = asyncio.get_running_loop()
    _, proto = await loop.create_datagram_endpoint(_BadSdpUas, local_addr=("127.0.0.1", 0))
    return proto


def test_call_with_no_common_codec_in_the_answer_is_reported_not_answered(tmp_path):
    async def main():
        uas = await start_bad_sdp_uas()
        try:
            port = uas.transport.get_extra_info("sockname")[1]
            return await run_call(
                target="agent",
                server_host="127.0.0.1",
                server_port=port,
                username="1001",
                password="",
                audio=generate_tone(440, 0.2),
                record_dir=tmp_path,
                duration=1.0,
            )
        finally:
            uas.transport.close()

    result = asyncio.run(main())

    assert result.answered is False
    assert result.reason is not None
    assert (tmp_path / "events.json").exists()


def test_call_with_unresolvable_server_is_reported_not_answered(tmp_path, monkeypatch):
    def raise_gaierror(*args, **kwargs):
        raise socket.gaierror("nodename nor servname provided, or not known")

    monkeypatch.setattr(socket, "getaddrinfo", raise_gaierror)

    result = asyncio.run(
        run_call(
            target="agent",
            server_host="nope.invalid.example",
            server_port=5060,
            username="1001",
            password="",
            audio=generate_tone(440, 0.2),
            record_dir=tmp_path,
            duration=1.0,
        )
    )

    assert result.answered is False
    assert result.reason is not None
    assert (tmp_path / "events.json").exists()


def test_codec_option_offers_only_that_payload_type(tmp_path):
    async def main():
        agent = await start_agent()
        try:
            return await run_call(
                target="agent",
                server_host="127.0.0.1",
                server_port=agent.local_addr[1],
                username="1001",
                password="",
                audio=generate_tone(440, 0.3),
                record_dir=tmp_path,
                duration=0.6,
                codec=PayloadType.PCMA,
            )
        finally:
            await agent.close()

    result = asyncio.run(main())
    assert result.answered
