"""SIP message parsing and building (RFC 3261 section 7). No I/O here."""

import re
import secrets
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from voiceqa.telephony.sip.errors import SipParseError

SIP_VERSION = "SIP/2.0"
BRANCH_COOKIE = "z9hG4bK"

_COMPACT = {
    "v": "via",
    "f": "from",
    "t": "to",
    "i": "call-id",
    "m": "contact",
    "l": "content-length",
    "c": "content-type",
    "s": "subject",
    "k": "supported",
    "e": "content-encoding",
}

# Headers whose values may be joined with commas on one line.
_LIST_HEADERS = {"via", "route", "record-route", "contact"}


def _key(name: str) -> str:
    lower = name.strip().lower()
    return _COMPACT.get(lower, lower)


def split_list(value: str) -> list[str]:
    """Split a comma-joined header value, ignoring commas in quotes or <...>."""
    parts = []
    start = 0
    quoted = False
    angle = False
    i = 0
    while i < len(value):
        ch = value[i]
        if quoted:
            if ch == "\\":
                i += 1
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch == "<":
            angle = True
        elif ch == ">":
            angle = False
        elif ch == "," and not angle:
            parts.append(value[start:i].strip())
            start = i + 1
        i += 1
    parts.append(value[start:].strip())
    return [p for p in parts if p]


class Headers:
    def __init__(self, items: Iterable[tuple[str, str]] = ()) -> None:
        self._items: list[tuple[str, str]] = list(items)

    def get(self, name: str) -> str | None:
        key = _key(name)
        for n, v in self._items:
            if _key(n) == key:
                return v
        return None

    def get_all(self, name: str) -> list[str]:
        key = _key(name)
        values = [v for n, v in self._items if _key(n) == key]
        if key in _LIST_HEADERS:
            return [part for v in values for part in split_list(v)]
        return values

    def add(self, name: str, value: str) -> None:
        self._items.append((name, value))

    def set(self, name: str, value: str) -> None:
        self.remove(name)
        self.add(name, value)

    def remove(self, name: str) -> None:
        key = _key(name)
        self._items = [(n, v) for n, v in self._items if _key(n) != key]

    def copy(self) -> "Headers":
        return Headers(self._items)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(list(self._items))


def _serialize(start_line: str, headers: Headers, body: bytes) -> bytes:
    lines = [start_line]
    lines += [f"{n}: {v}" for n, v in headers if _key(n) != "content-length"]
    lines.append(f"Content-Length: {len(body)}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + body


def _cseq(headers: Headers) -> tuple[int, str]:
    value = headers.get("CSeq")
    try:
        num, method = (value or "").split()
        return int(num), method.upper()
    except ValueError:
        raise SipParseError(f"bad CSeq: {value!r}") from None


@dataclass
class Request:
    method: str
    uri: str
    headers: Headers = field(default_factory=Headers)
    body: bytes = b""

    @property
    def cseq(self) -> tuple[int, str]:
        return _cseq(self.headers)

    def to_bytes(self) -> bytes:
        return _serialize(f"{self.method} {self.uri} {SIP_VERSION}", self.headers, self.body)


@dataclass
class Response:
    status: int
    reason: str
    headers: Headers = field(default_factory=Headers)
    body: bytes = b""

    @property
    def cseq(self) -> tuple[int, str]:
        return _cseq(self.headers)

    def to_bytes(self) -> bytes:
        return _serialize(f"{SIP_VERSION} {self.status} {self.reason}", self.headers, self.body)


def parse_message(data: bytes) -> Request | Response:
    head, sep, rest = data.partition(b"\r\n\r\n")
    if not sep:
        head, sep, rest = data.partition(b"\n\n")
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        raise SipParseError("message is not UTF-8") from None
    lines = text.replace("\r\n", "\n").split("\n")
    start = lines[0].strip()

    headers = Headers()
    for line in lines[1:]:
        if not line.strip():
            continue
        if line[0] in " \t":
            items = list(headers)
            if not items:
                raise SipParseError("continuation line before any header")
            name, value = items[-1]
            headers = Headers(items[:-1] + [(name, f"{value} {line.strip()}")])
            continue
        name, colon, value = line.partition(":")
        if not colon or not name.strip():
            raise SipParseError(f"bad header line: {line!r}")
        headers.add(name.strip(), value.strip())

    length = headers.get("Content-Length")
    if length is None:
        body = rest
    else:
        try:
            n = int(length)
        except ValueError:
            raise SipParseError(f"bad Content-Length: {length!r}") from None
        if n < 0 or n > len(rest):
            raise SipParseError(f"Content-Length {n} but only {len(rest)} body bytes")
        body = rest[:n]

    parts = start.split(" ", 2)
    if len(parts) == 3 and parts[0] == SIP_VERSION:
        code = parts[1]
        if len(code) != 3 or not code.isdigit():
            raise SipParseError(f"bad status line: {start!r}")
        return Response(int(code), parts[2], headers, body)
    if len(parts) == 3 and parts[2] == SIP_VERSION and parts[0].isalpha():
        return Request(parts[0].upper(), parts[1], headers, body)
    raise SipParseError(f"bad start line: {start!r}")


def parse_params(text: str) -> dict[str, str | None]:
    params: dict[str, str | None] = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        name, eq, value = part.partition("=")
        params[name.strip().lower()] = value.strip().strip('"') if eq else None
    return params


@dataclass(frozen=True)
class NameAddr:
    display: str | None
    uri: str
    params: dict[str, str | None]

    @property
    def tag(self) -> str | None:
        return self.params.get("tag")


def parse_name_addr(value: str) -> NameAddr:
    value = value.strip()
    if "<" in value:
        lt = value.index("<")
        gt = value.find(">", lt)
        if gt < 0:
            raise SipParseError(f"unclosed <: {value!r}")
        display = value[:lt].strip().strip('"').replace('\\"', '"') or None
        return NameAddr(display, value[lt + 1 : gt], parse_params(value[gt + 1 :]))
    uri, _, params = value.partition(";")
    return NameAddr(None, uri.strip(), parse_params(params))


@dataclass(frozen=True)
class Via:
    transport: str
    host: str
    port: int | None
    params: dict[str, str | None]

    @property
    def branch(self) -> str | None:
        return self.params.get("branch")


def parse_via(value: str) -> Via:
    sent, _, params = value.partition(";")
    try:
        protocol, sent_by = sent.split(None, 1)
        transport = protocol.split("/")[2].strip().upper()
        host, colon, port = sent_by.strip().rpartition(":")
        if not colon or "]" in port:
            host, port = sent_by.strip(), ""
        return Via(transport, host, int(port) if port else None, parse_params(params))
    except (ValueError, IndexError):
        raise SipParseError(f"bad Via: {value!r}") from None


def new_branch() -> str:
    return BRANCH_COOKIE + secrets.token_hex(8)


def new_tag() -> str:
    return secrets.token_hex(6)


def new_call_id() -> str:
    return secrets.token_hex(12)


def build_response(request: Request, status: int, reason: str) -> Response:
    headers = Headers()
    for name, value in request.headers:
        if _key(name) == "via":
            headers.add("Via", value)
    for name in ("From", "To", "Call-ID", "CSeq"):
        value = request.headers.get(name)
        if value is not None:
            headers.add(name, value)
    return Response(status, reason, headers)
