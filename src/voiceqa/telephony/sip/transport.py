"""SIP over UDP (RFC 3261 section 18)."""

import asyncio
import logging
from collections.abc import Callable

from voiceqa.telephony.sip.errors import SipParseError
from voiceqa.telephony.sip.message import Request, Response, parse_message

log = logging.getLogger(__name__)

Address = tuple[str, int]
OnMessage = Callable[[Request | Response, Address], None]


class UdpTransport(asyncio.DatagramProtocol):
    def __init__(self, on_message: OnMessage) -> None:
        self._on_message = on_message
        self._transport: asyncio.DatagramTransport | None = None
        self._local: Address = ("0.0.0.0", 0)
        self._lost: asyncio.Future[None] | None = None

    @classmethod
    async def open(cls, host: str, port: int, on_message: OnMessage) -> "UdpTransport":
        loop = asyncio.get_running_loop()
        _, protocol = await loop.create_datagram_endpoint(
            lambda: cls(on_message), local_addr=(host, port)
        )
        return protocol

    @property
    def local_addr(self) -> Address:
        return self._local

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        assert isinstance(transport, asyncio.DatagramTransport)
        self._transport = transport
        self._lost = asyncio.get_running_loop().create_future()
        sockname = transport.get_extra_info("sockname")
        self._local = (sockname[0], sockname[1])

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        try:
            msg = parse_message(data)
        except SipParseError as exc:
            log.debug("dropped datagram from %s: %s", addr, exc)
            return
        try:
            self._on_message(msg, (addr[0], addr[1]))
        except Exception:
            log.exception("error handling SIP message from %s", addr)

    def error_received(self, exc: Exception) -> None:
        # Windows reports ICMP port unreachable as ConnectionResetError here.
        log.debug("UDP error: %s", exc)

    def send(self, msg: Request | Response, addr: Address) -> None:
        if self._transport is not None and not self._transport.is_closing():
            self._transport.sendto(msg.to_bytes(), addr)

    def connection_lost(self, exc: Exception | None) -> None:
        if self._lost is not None and not self._lost.done():
            self._lost.set_result(None)

    def close(self) -> None:
        # abort() rather than close(): the Windows proactor transport never
        # calls connection_lost if close() lands while a send is in flight,
        # which leaks the socket.
        if self._transport is not None:
            self._transport.abort()

    async def aclose(self) -> None:
        # Two loop turns let sends already handed to the OS complete first.
        for _ in range(2):
            await asyncio.sleep(0)
        self.close()
        if self._lost is not None:
            await self._lost
