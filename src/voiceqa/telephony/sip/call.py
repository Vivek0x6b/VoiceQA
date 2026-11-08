"""Placing one outbound call: INVITE with digest auth, ACK, and BYE either way.

Every request goes to the configured server, used as an outbound proxy.
Responses we send go back to the source address of the request (RFC 3581).
"""

import asyncio
import contextlib
import itertools
import socket
from collections.abc import Callable
from dataclasses import dataclass, field

from voiceqa import __version__
from voiceqa.telephony.media import LocalMedia, RemoteMedia
from voiceqa.telephony.sip.auth import authorization_value, choose_challenge
from voiceqa.telephony.sip.dialog import Dialog, ack_for_2xx, build_invite, via_value
from voiceqa.telephony.sip.errors import SdpError, SipAuthError, SipError, SipStatusError
from voiceqa.telephony.sip.message import (
    Request,
    Response,
    build_response,
    new_branch,
    new_call_id,
    new_tag,
    parse_via,
)
from voiceqa.telephony.sip.sdp import build_offer, parse_answer
from voiceqa.telephony.sip.transaction import (
    InviteClientTransaction,
    NonInviteClientTransaction,
)
from voiceqa.telephony.sip.transport import Address, UdpTransport, routing_ip

MAX_AUTH_RETRIES = 4
_Transaction = InviteClientTransaction | NonInviteClientTransaction
_WILDCARDS = {"", "0.0.0.0"}

# Called with an event name and details: invite_sent (each new INVITE,
# including the one answering an auth challenge), ringing (first 18x, with
# status), answered (first 2xx), bye_sent, bye_received.
OnEvent = Callable[..., None]


def _ignore(name: str, **detail: object) -> None:
    pass


@dataclass(frozen=True)
class SipConfig:
    server_host: str
    username: str
    password: str = field(default="", repr=False)
    server_port: int = 5060
    from_domain: str | None = None
    local_host: str = "0.0.0.0"
    local_port: int = 0
    display_name: str | None = None
    timer_t1: float = 0.5
    timer_t2: float = 4.0
    user_agent: str = f"VoiceQA/{__version__}"


def _target_uri(target: str, server_host: str) -> str:
    if target.startswith(("sip:", "sips:")):
        return target
    return f"sip:{target}@{server_host}"


async def _resolve(host: str, port: int) -> Address:
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_DGRAM)
    except OSError as exc:
        raise SipError(f"cannot resolve SIP server {host}: {exc}") from exc
    addr = infos[0][4]
    return addr[0], addr[1]


class SipCall:
    def __init__(
        self,
        config: SipConfig,
        server: Address,
        transport_host: str,
        on_event: OnEvent = _ignore,
    ) -> None:
        self.call_id = new_call_id()
        self._config = config
        self._on_event = on_event
        self._rang = False
        self._answered = False
        self._server = server
        self._host = transport_host
        self._transport: UdpTransport | None = None
        self._transactions: dict[tuple[str, str], _Transaction] = {}
        self._dialog: Dialog | None = None
        self._ack: Request | None = None
        self._remote_media: RemoteMedia | None = None
        self._ended = asyncio.Event()
        self._closed = False

    @property
    def remote_media(self) -> RemoteMedia:
        if self._remote_media is None:
            raise SipError("call is not established")
        return self._remote_media

    @property
    def local_addr(self) -> Address:
        """The address we advertise in Via and Contact."""
        assert self._transport is not None
        return self._host, self._transport.local_addr[1]

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    async def wait_ended(self) -> None:
        await self._ended.wait()

    async def hangup(self) -> None:
        """Send BYE if the call is still up, then release the socket. Safe to call twice."""
        try:
            if self._dialog is not None and not self.ended and not self._closed:
                await self._send_bye()
        finally:
            await self._aclose()

    async def _open(self) -> None:
        self._transport = await UdpTransport.open(
            self._config.local_host, self._config.local_port, self._on_message
        )

    def _stop(self) -> bool:
        if self._closed:
            return False
        self._closed = True
        self._ended.set()
        for tx in list(self._transactions.values()):
            tx.close()
        return True

    def _close(self) -> None:
        if self._stop() and self._transport is not None:
            self._transport.close()

    async def _aclose(self) -> None:
        if self._stop() and self._transport is not None:
            await self._transport.aclose()

    def _via(self) -> str:
        return via_value(*self.local_addr, new_branch())

    def _send(self, req: Request) -> None:
        assert self._transport is not None
        self._transport.send(req, self._server)

    async def _transact(self, req: Request) -> Response:
        t1 = self._config.timer_t1
        tx: _Transaction
        if req.method == "INVITE":
            self._on_event("invite_sent")
            tx = InviteClientTransaction(req, self._send, t1, self._resend_ack)
        else:
            tx = NonInviteClientTransaction(req, self._send, t1, self._config.timer_t2)
        key = (tx.branch, req.method)
        self._transactions[key] = tx
        tx.on_terminated = lambda: self._transactions.pop(key, None)
        tx.start()
        return await tx.final()

    async def _with_auth(
        self, make: Callable[[list[tuple[str, str]]], Request]
    ) -> tuple[Request, Response]:
        """Send a request, answering 401/407 digest challenges.

        A realm that challenges again after we answered it means the
        credentials were refused, unless the server says the nonce was stale.
        """
        creds: dict[str, tuple[str, str]] = {}
        answered: set[str] = set()
        for attempt in itertools.count():
            req = make(list(creds.values()))
            resp = await self._transact(req)
            if resp.status not in (401, 407) or attempt >= MAX_AUTH_RETRIES:
                return req, resp
            proxy = resp.status == 407
            header = "Proxy-Authenticate" if proxy else "WWW-Authenticate"
            try:
                challenge = choose_challenge(resp.headers.get_all(header))
            except SipAuthError:
                return req, resp
            if challenge.realm in answered and not challenge.stale:
                return req, resp
            answered.add(challenge.realm)
            value = authorization_value(
                challenge,
                username=self._config.username,
                password=self._config.password,
                method=req.method,
                uri=req.uri,
            )
            name = "Proxy-Authorization" if proxy else "Authorization"
            creds[challenge.realm] = (name, value)
        raise AssertionError("unreachable")

    async def _establish(self, target: str, local: LocalMedia) -> None:
        cfg = self._config
        host, port = self.local_addr
        offer = build_offer(local)
        from_tag = new_tag()
        cseq = itertools.count(1)

        def make(extra: list[tuple[str, str]]) -> Request:
            return build_invite(
                target_uri=target,
                from_uri=f"sip:{cfg.username}@{cfg.from_domain or cfg.server_host}",
                from_tag=from_tag,
                display_name=cfg.display_name,
                call_id=self.call_id,
                cseq=next(cseq),
                contact_uri=f"sip:{cfg.username}@{host}:{port}",
                via=self._via(),
                sdp=offer,
                user_agent=cfg.user_agent,
                extra_headers=extra,
            )

        invite, resp = await self._with_auth(make)
        if resp.status >= 300:
            raise SipStatusError(resp.status, resp.reason)
        self._dialog = Dialog.from_response(invite, resp)
        self._ack = ack_for_2xx(self._dialog, invite, self._via())
        self._send(self._ack)
        try:
            self._remote_media = parse_answer(resp.body, local.payload_types)
        except SdpError:
            # The far end thinks the call is up, so end it before giving up.
            with contextlib.suppress(SipError):
                await self._send_bye()
            raise

    async def _send_bye(self) -> None:
        dialog = self._dialog
        assert dialog is not None

        def make(extra: list[tuple[str, str]]) -> Request:
            bye = dialog.make_request("BYE", self._via())
            for name, value in extra:
                bye.headers.add(name, value)
            return bye

        self._on_event("bye_sent")
        try:
            await self._with_auth(make)
        finally:
            # Any final answer, or none at all, ends the dialog for us.
            self._ended.set()

    def _resend_ack(self, _response: Response) -> None:
        if self._ack is not None:
            self._send(self._ack)

    def _on_message(self, msg: Request | Response, addr: Address) -> None:
        if isinstance(msg, Response):
            vias = msg.headers.get_all("Via")
            if not vias:
                return
            key = (parse_via(vias[0]).branch or "", msg.cseq[1])
            tx = self._transactions.get(key)
            if tx is not None:
                if msg.cseq[1] == "INVITE":
                    self._report_invite_response(msg.status)
                tx.receive(msg)
            return
        self._on_request(msg, addr)

    def _report_invite_response(self, status: int) -> None:
        if 180 <= status < 200 and not self._rang and not self._answered:
            self._rang = True
            self._on_event("ringing", status=status)
        elif 200 <= status < 300 and not self._answered:
            self._answered = True
            self._on_event("answered")

    def _on_request(self, req: Request, addr: Address) -> None:
        assert self._transport is not None
        if req.method == "ACK":
            return
        if req.method == "BYE":
            if self._dialog is not None and self._dialog.matches(req):
                if not self.ended:
                    self._on_event("bye_received")
                self._ended.set()
                resp = build_response(req, 200, "OK")
            else:
                resp = build_response(req, 481, "Call/Transaction Does Not Exist")
        elif req.method == "OPTIONS":
            resp = build_response(req, 200, "OK")
        elif req.method == "INVITE":
            resp = build_response(req, 488, "Not Acceptable Here")
        else:
            resp = build_response(req, 501, "Not Implemented")
        self._transport.send(resp, addr)


async def place_call(
    config: SipConfig,
    target_uri: str,
    local_media: LocalMedia,
    on_event: OnEvent | None = None,
) -> SipCall:
    """Call target_uri and return once it is answered and ACKed.

    target_uri may be a full SIP URI or just a user part, which is sent to
    the configured server. on_event is told about call events as they happen
    (see OnEvent), for example CallRecorder.event. Raises SipStatusError on a
    final failure, SipTimeoutError if the server never answers, and SdpError
    if the answer has no usable audio stream (the call is hung up first).
    """
    server = await _resolve(config.server_host, config.server_port)
    host = routing_ip(server) if config.local_host in _WILDCARDS else config.local_host
    call = SipCall(config, server, host, on_event or _ignore)
    await call._open()
    try:
        await call._establish(_target_uri(target_uri, config.server_host), local_media)
    except Exception:
        await call._aclose()
        raise
    except BaseException:
        call._close()
        raise
    return call
