"""INVITE and non-INVITE client transactions for UDP (RFC 3261 section 17.1),
and the resending of a 2xx to an INVITE on the server side."""

import asyncio
from collections.abc import Callable

from voiceqa.telephony.sip.errors import SipTimeoutError
from voiceqa.telephony.sip.message import Headers, Request, Response, parse_via

Send = Callable[[Request], None]


def ack_for_failure(invite: Request, response: Response) -> Request:
    # RFC 3261 17.1.1.3: same branch as the INVITE, To from the response.
    h = Headers()
    h.add("Via", invite.headers.get_all("Via")[0])
    h.add("Max-Forwards", "70")
    h.add("From", invite.headers.get("From") or "")
    h.add("To", response.headers.get("To") or "")
    h.add("Call-ID", invite.headers.get("Call-ID") or "")
    h.add("CSeq", f"{invite.cseq[0]} ACK")
    for route in invite.headers.get_all("Route"):
        h.add("Route", route)
    return Request("ACK", invite.uri, h)


class _ClientTransaction:
    def __init__(self, request: Request, send: Send, t1: float) -> None:
        self.request = request
        self.branch = parse_via(request.headers.get("Via") or "").branch or ""
        self.on_terminated: Callable[[], None] | None = None
        self.terminated = False
        self._send = send
        self._t1 = t1
        self._loop = asyncio.get_running_loop()
        self._final: asyncio.Future[Response] = self._loop.create_future()
        self._timers: dict[str, asyncio.TimerHandle] = {}

    async def final(self) -> Response:
        return await self._final

    def close(self) -> None:
        if self.terminated:
            return
        self.terminated = True
        self._cancel(*self._timers)
        if not self._final.done():
            self._final.cancel()
        elif not self._final.cancelled():
            self._final.exception()  # mark a timeout as seen if nobody awaited it
        if self.on_terminated is not None:
            self.on_terminated()

    def _schedule(self, name: str, delay: float, fn: Callable, *args: object) -> None:
        self._cancel(name)
        self._timers[name] = self._loop.call_later(delay, fn, *args)

    def _cancel(self, *names: str) -> None:
        for name in list(names):
            handle = self._timers.pop(name, None)
            if handle is not None:
                handle.cancel()

    def _timeout(self) -> None:
        if not self._final.done():
            method = self.request.method
            self._final.set_exception(
                SipTimeoutError(f"no final response to {method} after {64 * self._t1:g} s")
            )
        self.close()


class InviteClientTransaction(_ClientTransaction):
    def __init__(
        self,
        request: Request,
        send: Send,
        t1: float,
        on_retransmitted_2xx: Callable[[Response], None] | None = None,
    ) -> None:
        super().__init__(request, send, t1)
        self.provisional: list[Response] = []
        self._on_retransmitted_2xx = on_retransmitted_2xx
        self._ack: Request | None = None

    def start(self) -> None:
        self._send(self.request)
        self._schedule("A", self._t1, self._timer_a, self._t1)
        self._schedule("B", 64 * self._t1, self._timeout)

    def _timer_a(self, interval: float) -> None:
        self._send(self.request)
        self._schedule("A", 2 * interval, self._timer_a, 2 * interval)

    def receive(self, response: Response) -> None:
        if self.terminated:
            return
        if response.status < 200:
            if not self._final.done():
                self._cancel("A", "B")
                self.provisional.append(response)
            return
        self._cancel("A", "B")
        if response.status < 300:
            if not self._final.done():
                self._final.set_result(response)
                # Stay around to catch 2xx retransmissions (RFC 6026 Accepted state).
                self._schedule("M", 64 * self._t1, self.close)
            elif self._on_retransmitted_2xx is not None:
                self._on_retransmitted_2xx(response)
            return
        if self._ack is None:
            self._ack = ack_for_failure(self.request, response)
        if not self._final.done():
            self._final.set_result(response)
            self._schedule("D", 64 * self._t1, self.close)
        self._send(self._ack)


class NonInviteClientTransaction(_ClientTransaction):
    def __init__(self, request: Request, send: Send, t1: float, t2: float) -> None:
        super().__init__(request, send, t1)
        self._t2 = t2
        self._proceeding = False

    def start(self) -> None:
        self._send(self.request)
        self._schedule("E", self._t1, self._timer_e, self._t1)
        self._schedule("F", 64 * self._t1, self._timeout)

    def _timer_e(self, interval: float) -> None:
        self._send(self.request)
        nxt = self._t2 if self._proceeding else min(2 * interval, self._t2)
        self._schedule("E", nxt, self._timer_e, nxt)

    def receive(self, response: Response) -> None:
        if self.terminated:
            return
        if response.status < 200:
            self._proceeding = True
            return
        if not self._final.done():
            self._final.set_result(response)
        self.close()


def server_key(request: Request) -> tuple[str, str] | None:
    """Branch and method of a received request, to spot retransmissions."""
    vias = request.headers.get_all("Via")
    if not vias:
        return None
    branch = parse_via(vias[0]).branch
    return (branch, request.method) if branch else None


class OkRetransmitter:
    """Resends a final response to an INVITE until the ACK arrives (RFC 3261
    13.3.1.4 for a 2xx, 17.2.1 for the rest). The server also uses it for a 487.

    The interval starts at T1 and doubles up to T2. If no ACK came after
    64 * T1, resending stops and on_timeout is called.
    """

    def __init__(
        self,
        send: Callable[[], None],
        t1: float,
        t2: float,
        on_timeout: Callable[[], None] | None = None,
    ) -> None:
        self._send = send
        self._t1 = t1
        self._t2 = t2
        self._on_timeout = on_timeout
        self._loop = asyncio.get_running_loop()
        self._deadline = 0.0
        self._handle: asyncio.TimerHandle | None = None

    def start(self) -> None:
        self._deadline = self._loop.time() + 64 * self._t1
        self._handle = self._loop.call_later(self._t1, self._resend, 2 * self._t1)

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _resend(self, interval: float) -> None:
        if self._loop.time() >= self._deadline:
            self._handle = None
            if self._on_timeout is not None:
                self._on_timeout()
            return
        self._send()
        nxt = min(2 * interval, self._t2)
        self._handle = self._loop.call_later(interval, self._resend, nxt)
