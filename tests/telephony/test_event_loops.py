import asyncio
import sys

import pytest

from voiceqa.telephony.media import PayloadType, RemoteMedia
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.sip.transport import UdpTransport

# Linux and macOS use the selector loop. Windows defaults to the proactor loop
# but has both, so running each here keeps the selector path covered on every OS.
LOOPS = [pytest.param(asyncio.SelectorEventLoop, id="selector")]
if sys.platform == "win32":
    LOOPS.append(pytest.param(asyncio.ProactorEventLoop, id="proactor"))


def run_on(loop_factory, coro):
    loop = loop_factory()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.mark.parametrize("loop_factory", LOOPS)
def test_sip_transport_binds(loop_factory):
    async def main():
        t = await UdpTransport.open("127.0.0.1", 0, lambda m, addr: None)
        port = t.local_addr[1]
        await asyncio.wait_for(t.aclose(), 2)
        return port

    assert run_on(loop_factory, main()) != 0


@pytest.mark.parametrize("loop_factory", LOOPS)
def test_rtp_session_opens_and_closes(loop_factory):
    async def main():
        a = await RtpSession.open("127.0.0.1")
        b = await RtpSession.open("127.0.0.1")
        await a.start(RemoteMedia("127.0.0.1", b.local.port, PayloadType.PCMU))
        await asyncio.sleep(0.1)
        await a.close()
        await b.close()
        return a.packets_sent

    assert run_on(loop_factory, main()) > 0
