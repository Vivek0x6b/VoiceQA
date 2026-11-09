"""Answering calls: a minimal SIP UAS over UDP for the local test agent.

Every new INVITE with a usable SDP offer is answered at once with 100, 180
and 200. An INVITE without an offer gets our offer in the 200 and must bring
the answer in the ACK; if it does not, the call is ended with BYE. The 200 is
resent until the ACK arrives (RFC 3261 section 13.3.1.4); without an ACK
after 64 * T1 the call is ended with BYE. Responses go back to
the source address of the request (RFC 3581), and requests we start inside a
dialog (BYE) go to the address the INVITE came from, so the agent works
behind the caller's NAT but not through a proxy that expects its Route set
to be followed.

An in-dialog re-INVITE or UPDATE (a session refresh) gets 200 with the SDP
we sent before, and media keeps running untouched.

Media is opened through a callback that returns a MediaSession, so this
module never touches RTP.
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from voiceqa import __version__
from voiceqa.telephony.media import MediaSession, PayloadType, RemoteMedia
from voiceqa.telephony.sip.dialog import ALLOW, Dialog, answer_refresh, via_value
from voiceqa.telephony.sip.errors import SdpError, SipError
from voiceqa.telephony.sip.message import (
    Request,
    Response,
    build_response,
    new_branch,
    new_tag,
    parse_name_addr,
    parse_via,
)
from voiceqa.telephony.sip.sdp import build_answer, build_offer, parse_answer, parse_offer
from voiceqa.telephony.sip.transaction import (
    NonInviteClientTransaction,
    OkRetransmitter,
    server_key,
)
from voiceqa.telephony.sip.transport import Address, UdpTransport, routing_ip

log = logging.getLogger(__name__)

_WILDCARDS = {"", "0.0.0.0"}

OpenMedia = Callable[[str], Awaitable[MediaSession]]


def sdp_body(msg: Request) -> bytes:
    """The body if it is SDP, else empty. A body without Content-Type counts as SDP."""
    kind = msg.headers.get("Content-Type") or "application/sdp"
    if kind.split(";")[0].strip().lower() != "application/sdp":
        return b""
    return msg.body


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 5060
    user: str = "agent"
    payload_types: tuple[PayloadType, ...] = (PayloadType.PCMU, PayloadType.PCMA)
    timer_t1: float = 0.5
    timer_t2: float = 4.0
    user_agent: str = f"VoiceQA-agent/{__version__}"


class ServerCall:
    """One answered call, from the 200 OK until a BYE either way."""

    def __init__(
        self,
        server: "SipServer",
        invite: Request,
        peer: Address,
        dialog: Dialog,
        media: MediaSession,
        remote_media: RemoteMedia | None,
        local_sdp: str,
        contact: str,
    ) -> None:
        self.call_id = dialog.call_id
        self.peer = peer
        self.media = media
        # None until the ACK brings the answer when the INVITE had no offer.
        self.remote_media = remote_media
        self._remote: asyncio.Future[RemoteMedia | None] = (
            asyncio.get_running_loop().create_future()
        )
        if remote_media is not None:
            self._remote.set_result(remote_media)
        self.invite = invite
        self.dialog = dialog
        # The SDP we sent last in this dialog. Session refreshes get it again.
        self.local_sdp = local_sdp
        self.contact = contact
        self.ok: Response | None = None
        self.task: asyncio.Task | None = None
        self._server = server
        self._ended = asyncio.Event()
        self._closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        # 2xx responses waiting for their ACK, by CSeq number.
        self._resending: dict[int, OkRetransmitter] = {}
        self._confirmed = asyncio.Event()
        self._bye_sent = False

    @property
    def ended(self) -> bool:
        return self._ended.is_set()

    async def wait_ended(self) -> None:
        await self._ended.wait()

    async def hangup(self) -> None:
        """Send BYE if the call is still up and wait until media is closed. Safe to call twice."""
        if not self.ended and not self._bye_sent:
            self._bye_sent = True
            # RFC 3261 15.1.1: no BYE before the ACK (or its timeout). The
            # caller may not even know the dialog yet.
            await self._confirmed.wait()
            if not self.ended:
                with contextlib.suppress(SipError):
                    await self._server._send_bye(self)
                self._end()
        await asyncio.shield(self._closed)

    def _start_resending(self, ok: Response) -> None:
        self.ok = ok

        def no_ack() -> None:
            self._confirmed.set()
            log.info("no ACK for call %s, hanging up", self.call_id)
            self._server._spawn(self.hangup())

        self._resend_until_acked(ok, self.peer, no_ack)

    def _resend_until_acked(
        self, ok: Response, peer: Address, on_timeout: Callable[[], None] | None = None
    ) -> None:
        seq = ok.cseq[0]

        def timeout() -> None:
            self._resending.pop(seq, None)
            if on_timeout is not None:
                on_timeout()

        cfg = self._server.config
        resender = OkRetransmitter(
            lambda: self._server._transport_send(ok, peer), cfg.timer_t1, cfg.timer_t2, timeout
        )
        self._resending[seq] = resender
        resender.start()

    def _acked(self, ack: Request) -> None:
        seq = ack.cseq[0]
        resender = self._resending.pop(seq, None)
        if resender is not None:
            resender.stop()
        if seq == self.invite.cseq[0]:
            self._confirmed.set()
            if not self._remote.done():
                self._remote.set_result(self._read_answer(ack))

    def _read_answer(self, ack: Request) -> RemoteMedia | None:
        try:
            return parse_answer(sdp_body(ack), self.media.local.payload_types)
        except SdpError as exc:
            log.info("no usable answer in the ACK for call %s: %s", self.call_id, exc)
            return None

    def _confirm(self) -> None:
        self._confirmed.set()
        for resender in self._resending.values():
            resender.stop()
        self._resending.clear()

    def _end(self) -> None:
        if self.ended:
            return
        self._ended.set()
        self._confirm()
        if not self._remote.done():
            self._remote.set_result(None)
        log.info("call %s ended", self.call_id)
        self._server._forget(self)
        # The handler may be the one hanging up; then it is neither cancelled
        # nor waited for, since it is waiting for us.
        handler = self.task if self.task is not asyncio.current_task() else None
        if handler is not None:
            handler.cancel()
        self._server._spawn(self._close_media(handler))

    async def _close_media(self, handler: asyncio.Task | None) -> None:
        try:
            if handler is not None:
                await asyncio.gather(handler, return_exceptions=True)
            await self.media.close()
        finally:
            if not self._closed.done():
                self._closed.set_result(None)


OnCall = Callable[[ServerCall], Awaitable[None]]


class SipServer:
    def __init__(self, config: ServerConfig, open_media: OpenMedia, on_call: OnCall) -> None:
        self.config = config
        self._open_media = open_media
        self._on_call = on_call
        self._transport: UdpTransport | None = None
        self._calls: dict[str, ServerCall] = {}
        self._pending: set[str] = set()
        # Last response per server transaction, to answer retransmissions.
        self._answered: dict[tuple[str, str], Response] = {}
        self._clients: dict[tuple[str, str], NonInviteClientTransaction] = {}
        self._tasks: set[asyncio.Task] = set()

    @classmethod
    async def open(
        cls, config: ServerConfig, open_media: OpenMedia, on_call: OnCall
    ) -> "SipServer":
        self = cls(config, open_media, on_call)
        self._transport = await UdpTransport.open(config.host, config.port, self._on_message)
        return self

    @property
    def local_addr(self) -> Address:
        assert self._transport is not None
        return self._transport.local_addr

    @property
    def calls(self) -> list[ServerCall]:
        return list(self._calls.values())

    async def close(self) -> None:
        await asyncio.gather(*(call.hangup() for call in self.calls))
        for task in list(self._tasks):
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        for tx in list(self._clients.values()):
            tx.close()
        if self._transport is not None:
            await self._transport.aclose()

    def _host_for(self, peer: Address) -> str:
        if self.config.host in _WILDCARDS:
            return routing_ip(peer)
        return self.config.host

    def _transport_send(self, msg: Request | Response, addr: Address) -> None:
        assert self._transport is not None
        self._transport.send(msg, addr)

    def _spawn(self, coro: Awaitable[None]) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _forget(self, call: ServerCall) -> None:
        if self._calls.get(call.call_id) is call:
            del self._calls[call.call_id]

    def _reply(
        self, req: Request, peer: Address, status: int, reason: str, tag: str | None = None
    ) -> Response:
        resp = build_response(req, status, reason)
        to = req.headers.get("To") or ""
        if status > 100 and parse_name_addr(to).tag is None:
            resp.headers.set("To", f"{to};tag={tag or new_tag()}")
        resp.headers.add("Server", self.config.user_agent)
        self._answer(req, peer, resp)
        return resp

    def _answer(self, req: Request, peer: Address, resp: Response) -> None:
        key = server_key(req)
        if key is not None:
            self._answered[key] = resp
            if resp.status >= 200:
                # Timer J / H: keep the answer around for retransmissions.
                loop = asyncio.get_running_loop()
                loop.call_later(64 * self.config.timer_t1, self._answered.pop, key, None)
        self._transport_send(resp, peer)

    def _via(self, peer: Address) -> str:
        host = self._host_for(peer)
        return via_value(host, self.local_addr[1], new_branch())

    async def _send_bye(self, call: ServerCall) -> None:
        bye = call.dialog.make_request("BYE", self._via(call.peer))
        tx = NonInviteClientTransaction(
            bye,
            lambda r: self._transport_send(r, call.peer),
            self.config.timer_t1,
            self.config.timer_t2,
        )
        key = (tx.branch, "BYE")
        self._clients[key] = tx
        tx.on_terminated = lambda: self._clients.pop(key, None)
        tx.start()
        await tx.final()

    def _on_message(self, msg: Request | Response, peer: Address) -> None:
        if isinstance(msg, Response):
            vias = msg.headers.get_all("Via")
            if vias:
                tx = self._clients.get((parse_via(vias[0]).branch or "", msg.cseq[1]))
                if tx is not None:
                    tx.receive(msg)
            return
        self._on_request(msg, peer)

    def _on_request(self, req: Request, peer: Address) -> None:
        if req.method == "ACK":
            self._on_ack(req)
            return
        key = server_key(req)
        if key is not None and key in self._answered:
            self._transport_send(self._answered[key], peer)
            return
        call = self._calls.get(req.headers.get("Call-ID") or "")
        in_dialog = call is not None and call.dialog.matches(req)
        has_to_tag = parse_name_addr(req.headers.get("To") or "").tag is not None

        if in_dialog and req.method != "CANCEL":
            assert call is not None
            if not call.dialog.accept_seq(req):
                self._reply(req, peer, 500, "Server Internal Error")
                return

        if req.method in ("INVITE", "UPDATE") and in_dialog:
            assert call is not None
            self._refresh(call, req, peer)
        elif req.method == "INVITE":
            if has_to_tag:
                self._reply(req, peer, 481, "Call/Transaction Does Not Exist")
            elif call is not None or req.headers.get("Call-ID") in self._pending:
                self._reply(req, peer, 482, "Loop Detected")
            else:
                self._on_invite(req, peer)
        elif req.method == "BYE":
            if in_dialog:
                assert call is not None
                self._reply(req, peer, 200, "OK")
                call._end()
            else:
                self._reply(req, peer, 481, "Call/Transaction Does Not Exist")
        elif req.method == "CANCEL":
            # We answer at once, so a CANCEL only ever finds a finished INVITE.
            invite_key = (key[0], "INVITE") if key else None
            if invite_key is not None and invite_key in self._answered:
                self._reply(req, peer, 200, "OK")
            else:
                self._reply(req, peer, 481, "Call/Transaction Does Not Exist")
        elif req.method == "UPDATE":
            self._reply(req, peer, 481, "Call/Transaction Does Not Exist")
        elif req.method == "OPTIONS":
            resp = self._reply(req, peer, 200, "OK")
            resp.headers.add("Allow", ALLOW)
        else:
            self._reply(req, peer, 501, "Not Implemented")

    def _refresh(self, call: ServerCall, req: Request, peer: Address) -> None:
        # Media keeps running as it is, whatever the offer says.
        resp = answer_refresh(req, call.local_sdp, self.config.payload_types, call.contact)
        resp.headers.add("Server", self.config.user_agent)
        self._answer(req, peer, resp)
        if req.method == "INVITE" and resp.status == 200:
            call._resend_until_acked(resp, peer)

    def _on_ack(self, ack: Request) -> None:
        # The ACK to the first 200 carries the answer when the INVITE had no
        # offer. The answer to a refresh offer is not read, the session stays.
        call = self._calls.get(ack.headers.get("Call-ID") or "")
        if call is not None and call.dialog.matches(ack):
            call._acked(ack)

    def _on_invite(self, invite: Request, peer: Address) -> None:
        offer = sdp_body(invite)
        remote = None
        if offer:
            try:
                remote = parse_offer(offer, self.config.payload_types)
            except SdpError as exc:
                log.info("rejecting INVITE from %s: %s", peer, exc)
                self._reply(invite, peer, 488, "Not Acceptable Here")
                return
        call_id = invite.headers.get("Call-ID") or ""
        self._pending.add(call_id)
        self._reply(invite, peer, 100, "Trying")
        self._spawn(self._answer_call(invite, peer, remote))

    async def _answer_call(
        self, invite: Request, peer: Address, remote: RemoteMedia | None
    ) -> None:
        """Answer a new INVITE. Without an offer in it, ours goes in the 200 and
        the answer comes in the ACK; media starts once the answer is known."""
        call_id = invite.headers.get("Call-ID") or ""
        tag = new_tag()
        try:
            host = self._host_for(peer)
            try:
                media = await self._open_media(host)
            except Exception:
                log.exception("could not open media for call %s", call_id)
                self._reply(invite, peer, 500, "Server Internal Error", tag)
                return
            if remote is None:
                local_sdp = build_offer(media.local)
            else:
                try:
                    local_sdp = build_answer(media.local, remote.payload_type)
                except ValueError:
                    log.exception("media for call %s cannot do the offered codec", call_id)
                    await media.close()
                    self._reply(invite, peer, 500, "Server Internal Error", tag)
                    return
            self._reply(invite, peer, 180, "Ringing", tag)
            dialog = Dialog.from_invite(invite, tag)
            contact = f"<sip:{self.config.user}@{host}:{self.local_addr[1]}>"
            call = ServerCall(self, invite, peer, dialog, media, remote, local_sdp, contact)
            ok = build_response(invite, 200, "OK")
            ok.headers.set("To", dialog.local_from)
            for rr in invite.headers.get_all("Record-Route"):
                ok.headers.add("Record-Route", rr)
            ok.headers.add("Contact", contact)
            ok.headers.add("Allow", ALLOW)
            ok.headers.add("Server", self.config.user_agent)
            ok.headers.add("Content-Type", "application/sdp")
            ok.body = local_sdp.encode()
            self._calls[call_id] = call
        finally:
            self._pending.discard(call_id)
        self._answer(invite, peer, ok)
        call._start_resending(ok)
        remote = await call._remote
        if remote is None:
            # The ACK had no usable answer, or the call ended before it came.
            if not call.ended:
                await call.hangup()
            return
        call.remote_media = remote
        log.info("answered call %s from %s:%s (%s)", call_id, *peer, remote.payload_type.name)
        await media.start(remote)
        if not call.ended:
            call.task = self._spawn(self._run_call(call))

    async def _run_call(self, call: ServerCall) -> None:
        try:
            await self._on_call(call)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("call handler failed for call %s", call.call_id)
