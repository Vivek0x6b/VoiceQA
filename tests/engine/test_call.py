import asyncio
import json

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
