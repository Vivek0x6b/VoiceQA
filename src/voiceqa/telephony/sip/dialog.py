"""INVITE building and the UAC side of a dialog (RFC 3261 section 12). No I/O here."""

from collections.abc import Iterable
from dataclasses import dataclass

from voiceqa.telephony.sip.message import Headers, Request, Response, parse_name_addr

ALLOW = "INVITE, ACK, BYE, OPTIONS"


def via_value(host: str, port: int, branch: str) -> str:
    return f"SIP/2.0/UDP {host}:{port};branch={branch};rport"


def _name_addr(uri: str, display_name: str | None) -> str:
    if not display_name:
        return f"<{uri}>"
    quoted = display_name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{quoted}" <{uri}>'


def build_invite(
    *,
    target_uri: str,
    from_uri: str,
    from_tag: str,
    display_name: str | None,
    call_id: str,
    cseq: int,
    contact_uri: str,
    via: str,
    sdp: str,
    user_agent: str,
    extra_headers: Iterable[tuple[str, str]] = (),
) -> Request:
    h = Headers()
    h.add("Via", via)
    h.add("Max-Forwards", "70")
    h.add("From", f"{_name_addr(from_uri, display_name)};tag={from_tag}")
    h.add("To", f"<{target_uri}>")
    h.add("Call-ID", call_id)
    h.add("CSeq", f"{cseq} INVITE")
    h.add("Contact", f"<{contact_uri}>")
    h.add("Allow", ALLOW)
    h.add("User-Agent", user_agent)
    for name, value in extra_headers:
        h.add(name, value)
    h.add("Content-Type", "application/sdp")
    return Request("INVITE", target_uri, h, sdp.encode())


@dataclass
class Dialog:
    call_id: str
    local_tag: str
    remote_tag: str
    local_from: str
    remote_to: str
    remote_target: str
    route_set: tuple[str, ...]
    local_seq: int

    @classmethod
    def from_response(cls, invite: Request, response: Response) -> "Dialog":
        local_from = invite.headers.get("From") or ""
        remote_to = response.headers.get("To") or ""
        contacts = response.headers.get_all("Contact")
        target = parse_name_addr(contacts[0]).uri if contacts else invite.uri
        return cls(
            call_id=invite.headers.get("Call-ID") or "",
            local_tag=parse_name_addr(local_from).tag or "",
            remote_tag=parse_name_addr(remote_to).tag or "",
            local_from=local_from,
            remote_to=remote_to,
            remote_target=target,
            # A UAC uses the Record-Route list in reverse order
            route_set=tuple(reversed(response.headers.get_all("Record-Route"))),
            local_seq=invite.cseq[0],
        )

    @classmethod
    def from_invite(cls, invite: Request, local_tag: str) -> "Dialog":
        """The UAS side, created when we answer an INVITE."""
        remote_from = invite.headers.get("From") or ""
        contacts = invite.headers.get_all("Contact")
        target = parse_name_addr(contacts[0] if contacts else remote_from).uri
        return cls(
            call_id=invite.headers.get("Call-ID") or "",
            local_tag=local_tag,
            remote_tag=parse_name_addr(remote_from).tag or "",
            local_from=f"{invite.headers.get('To') or ''};tag={local_tag}",
            remote_to=remote_from,
            remote_target=target,
            route_set=tuple(invite.headers.get_all("Record-Route")),
            local_seq=0,
        )

    def make_request(self, method: str, via: str, cseq: int | None = None) -> Request:
        if cseq is None:
            self.local_seq += 1
            cseq = self.local_seq
        h = Headers()
        h.add("Via", via)
        h.add("Max-Forwards", "70")
        h.add("From", self.local_from)
        h.add("To", self.remote_to)
        h.add("Call-ID", self.call_id)
        h.add("CSeq", f"{cseq} {method}")
        for route in self.route_set:
            h.add("Route", route)
        return Request(method, self.remote_target, h)

    def matches(self, request: Request) -> bool:
        return (
            request.headers.get("Call-ID") == self.call_id
            and parse_name_addr(request.headers.get("To") or "").tag == self.local_tag
            and parse_name_addr(request.headers.get("From") or "").tag == self.remote_tag
        )


def ack_for_2xx(dialog: Dialog, invite: Request, via: str) -> Request:
    ack = dialog.make_request("ACK", via, cseq=invite.cseq[0])
    for name in ("Authorization", "Proxy-Authorization"):
        for value in invite.headers.get_all(name):
            ack.headers.add(name, value)
    return ack
