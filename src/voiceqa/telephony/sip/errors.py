class SipError(Exception):
    pass


class SipParseError(SipError):
    pass


class SdpError(SipError):
    pass


class SipAuthError(SipError):
    pass


class SipTimeoutError(SipError):
    pass


class SipStatusError(SipError):
    """The far end answered with a final status we can't continue from."""

    def __init__(self, status: int, reason: str) -> None:
        super().__init__(f"{status} {reason}")
        self.status = status
        self.reason = reason
