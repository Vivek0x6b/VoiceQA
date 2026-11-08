import hashlib

import pytest

from voiceqa.telephony.sip.auth import (
    Challenge,
    authorization_value,
    choose_challenge,
    digest_response,
    parse_challenge,
)
from voiceqa.telephony.sip.errors import SipAuthError
from voiceqa.telephony.sip.message import parse_params

# RFC 2617 section 3.5
RFC_CHALLENGE = (
    'Digest realm="testrealm@host.com", qop="auth,auth-int", '
    'nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093", '
    'opaque="5ccc069c403ebaf9f0171e9517f40e41"'
)
RFC_RESPONSE = "6629fae49393a05397450978507c4ef1"


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def fields(value: str) -> dict:
    assert value.startswith("Digest ")
    return parse_params(value[len("Digest ") :].replace(",", ";"))


def test_rfc2617_example():
    assert (
        digest_response(
            username="Mufasa",
            realm="testrealm@host.com",
            password="Circle Of Life",
            method="GET",
            uri="/dir/index.html",
            nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093",
            qop="auth",
            nc="00000001",
            cnonce="0a4f113b",
        )
        == RFC_RESPONSE
    )


def test_parse_rfc_challenge():
    ch = parse_challenge(RFC_CHALLENGE)
    assert ch == Challenge(
        realm="testrealm@host.com",
        nonce="dcd98b7102dd2f0e8b11d0f600bfb0c093",
        algorithm="MD5",
        qop=("auth", "auth-int"),
        opaque="5ccc069c403ebaf9f0171e9517f40e41",
        stale=False,
    )


def test_authorization_value_for_rfc_example():
    value = authorization_value(
        parse_challenge(RFC_CHALLENGE),
        username="Mufasa",
        password="Circle Of Life",
        method="GET",
        uri="/dir/index.html",
        nc=1,
        cnonce="0a4f113b",
    )
    f = fields(value)
    assert f["username"] == "Mufasa"
    assert f["realm"] == "testrealm@host.com"
    assert f["nonce"] == "dcd98b7102dd2f0e8b11d0f600bfb0c093"
    assert f["uri"] == "/dir/index.html"
    assert f["response"] == RFC_RESPONSE
    assert f["opaque"] == "5ccc069c403ebaf9f0171e9517f40e41"
    assert f["qop"] == "auth"
    assert f["nc"] == "00000001"
    assert f["cnonce"] == "0a4f113b"
    assert f["algorithm"] == "MD5"
    # RFC 2617: qop and nc are unquoted tokens in the reply
    assert "qop=auth," in value
    assert "nc=00000001" in value


def test_without_qop():
    ch = parse_challenge('Digest realm="asterisk", nonce="abc123"')
    assert ch.qop == ()
    value = authorization_value(
        ch, username="1001", password="secret", method="INVITE", uri="sip:200@pbx"
    )
    f = fields(value)
    ha1 = md5("1001:asterisk:secret")
    ha2 = md5("INVITE:sip:200@pbx")
    assert f["response"] == md5(f"{ha1}:abc123:{ha2}")
    assert "qop" not in f
    assert "cnonce" not in f
    assert "opaque" not in f


def test_default_cnonce_is_random():
    ch = parse_challenge(RFC_CHALLENGE)
    a = fields(authorization_value(ch, username="u", password="p", method="M", uri="sip:x"))
    b = fields(authorization_value(ch, username="u", password="p", method="M", uri="sip:x"))
    assert a["cnonce"] != b["cnonce"]


def test_parse_stale_and_token_params():
    ch = parse_challenge("digest realm=pbx, nonce=n1, stale=TRUE, algorithm=md5")
    assert ch.realm == "pbx"
    assert ch.nonce == "n1"
    assert ch.stale is True
    assert ch.algorithm == "MD5"


@pytest.mark.parametrize(
    "value",
    ['Basic realm="x"', 'Digest realm="x"', ""],
)
def test_parse_bad_challenge(value):
    with pytest.raises(SipAuthError):
        parse_challenge(value)


def test_unsupported_algorithm_and_qop():
    with pytest.raises(SipAuthError, match="algorithm"):
        authorization_value(
            parse_challenge('Digest realm="r", nonce="n", algorithm=SHA-256'),
            username="u",
            password="p",
            method="INVITE",
            uri="sip:x",
        )
    with pytest.raises(SipAuthError, match="qop"):
        authorization_value(
            parse_challenge('Digest realm="r", nonce="n", qop="auth-int"'),
            username="u",
            password="p",
            method="INVITE",
            uri="sip:x",
        )


def test_choose_challenge_prefers_md5():
    ch = choose_challenge(
        [
            'Digest realm="r", nonce="n1", algorithm=SHA-256',
            "Basic realm=r",
            'Digest realm="r", nonce="n2", algorithm=MD5',
        ]
    )
    assert ch.nonce == "n2"


def test_choose_challenge_none_usable():
    with pytest.raises(SipAuthError):
        choose_challenge(['Digest realm="r", nonce="n", algorithm=SHA-256'])
    with pytest.raises(SipAuthError):
        choose_challenge([])
