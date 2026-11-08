"""HTTP digest auth for SIP (RFC 2617, RFC 7616), MD5 only."""

import hashlib
import secrets
from collections.abc import Iterable
from dataclasses import dataclass

from voiceqa.telephony.sip.errors import SipAuthError


@dataclass(frozen=True)
class Challenge:
    realm: str
    nonce: str
    algorithm: str = "MD5"
    qop: tuple[str, ...] = ()
    opaque: str | None = None
    stale: bool = False


def _auth_params(text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t,":
            i += 1
        eq = text.find("=", i)
        if eq < 0:
            break
        name = text[i:eq].strip().lower()
        i = eq + 1
        while i < n and text[i] in " \t":
            i += 1
        if i < n and text[i] == '"':
            i += 1
            chars = []
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                chars.append(text[i])
                i += 1
            i += 1
            value = "".join(chars)
        else:
            end = text.find(",", i)
            end = n if end < 0 else end
            value = text[i:end].strip()
            i = end
        params[name] = value
    return params


def parse_challenge(value: str) -> Challenge:
    scheme, _, rest = value.strip().partition(" ")
    if scheme.lower() != "digest":
        raise SipAuthError(f"not a digest challenge: {value!r}")
    p = _auth_params(rest)
    if "nonce" not in p:
        raise SipAuthError(f"challenge has no nonce: {value!r}")
    qop = tuple(q.strip().lower() for q in p.get("qop", "").split(",") if q.strip())
    return Challenge(
        realm=p.get("realm", ""),
        nonce=p["nonce"],
        algorithm=p.get("algorithm", "MD5").upper(),
        qop=qop,
        opaque=p.get("opaque"),
        stale=p.get("stale", "").lower() == "true",
    )


def choose_challenge(values: Iterable[str]) -> Challenge:
    for value in values:
        try:
            ch = parse_challenge(value)
        except SipAuthError:
            continue
        if ch.algorithm == "MD5":
            return ch
    raise SipAuthError("no usable MD5 digest challenge")


def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def digest_response(
    *,
    username: str,
    realm: str,
    password: str,
    method: str,
    uri: str,
    nonce: str,
    qop: str | None = None,
    nc: str | None = None,
    cnonce: str | None = None,
) -> str:
    ha1 = _md5(f"{username}:{realm}:{password}")
    ha2 = _md5(f"{method}:{uri}")
    if qop:
        return _md5(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}")
    return _md5(f"{ha1}:{nonce}:{ha2}")


def _quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def authorization_value(
    challenge: Challenge,
    *,
    username: str,
    password: str,
    method: str,
    uri: str,
    nc: int = 1,
    cnonce: str | None = None,
) -> str:
    if challenge.algorithm != "MD5":
        raise SipAuthError(f"unsupported digest algorithm {challenge.algorithm}")
    qop = None
    if challenge.qop:
        if "auth" not in challenge.qop:
            raise SipAuthError(f"unsupported qop {','.join(challenge.qop)}")
        qop = "auth"
    nc_text = f"{nc:08x}"
    if qop and cnonce is None:
        cnonce = secrets.token_hex(8)
    response = digest_response(
        username=username,
        realm=challenge.realm,
        password=password,
        method=method,
        uri=uri,
        nonce=challenge.nonce,
        qop=qop,
        nc=nc_text,
        cnonce=cnonce,
    )
    parts = [
        f"username={_quote(username)}",
        f"realm={_quote(challenge.realm)}",
        f"nonce={_quote(challenge.nonce)}",
        f"uri={_quote(uri)}",
        f"response={_quote(response)}",
        "algorithm=MD5",
    ]
    if challenge.opaque is not None:
        parts.append(f"opaque={_quote(challenge.opaque)}")
    if qop:
        parts += [f"qop={qop}", f"nc={nc_text}", f"cnonce={_quote(cnonce or '')}"]
    return "Digest " + ", ".join(parts)
