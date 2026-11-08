# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffold with src layout and hatchling build.
- `voiceqa version` and `voiceqa doctor` commands.
- CI on Windows, macOS and Linux for Python 3.11 to 3.13.
- Current status section in docs/ARCHITECTURE.md, with the full list of M1 issues.
- .gitattributes so text files always use LF line endings.
- G.711 PCMU and PCMA encode and decode in pure NumPy, bit-exact with the old audioop module so it works on Python 3.13.
- 16-bit PCM WAV read and write, stereo to mono downmix, and a sine test tone generator.
- `ruff format --check .` in CI. Failing tests are listed as annotations on the CI run.
- Media interface between SIP and RTP (LocalMedia, RemoteMedia, MediaSession).
- Polyphase windowed-sinc resampler with anti-alias filtering, used to bring audio to 8 kHz for calls. 60 seconds of 48 kHz audio takes about 0.1 s.
- RTP packet build and parse (RFC 3550, handles padding, CSRC and header extensions).
- `RtpSession`: G.711 RTP over UDP paced at 50 packets/s on absolute deadlines, with a recording placed by RTP timestamp so reordered, duplicate and missing packets are handled.
- Minimal SIP UAC over UDP: INVITE with retransmission, MD5 digest auth for 401 and 407, ACK, BYE in both directions, and SDP offer/answer for PCMU and PCMA.
- SIP and RTP socket tests run on both asyncio event loops (selector and, on Windows, proactor).
- Minimal SIP UAS (`SipServer`) that answers calls with 100, 180 and 200, resends the 200 until the ACK arrives, answers resent requests, and handles BYE in both directions, OPTIONS and CANCEL. Offers without a common codec get 488.
- Local test agent and `voiceqa agent` command: answers calls on a local port and echoes the caller's audio or plays a WAV file (any rate, mono or stereo), optionally hanging up after a set time.
- `RtpSession.listen()` to get each received frame as it arrives. Echo mode uses it with a bounded 1 s buffer, so its work and memory per frame do not grow with call length.
