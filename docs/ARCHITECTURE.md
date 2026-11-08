# Architecture

VoiceQA is a local-first QA lab for AI voice agents. It places real SIP calls to an agent, speaks with local TTS, transcribes with local STT (faster-whisper), and writes JSON and HTML reports. No audio, transcript or report leaves the user's machine.

## Current status

- Current milestone: M1 (local SIP call with two-way audio and recording), in progress.
- Finished issues: issue 1 (G.711 codec and WAV I/O), issue 2 (RTP sender and receiver with 20 ms pacing), issue 3 (SIP UAC with INVITE, ACK, BYE, digest auth and SDP offer/answer), issue 4 (local test agent). The project scaffold, CLI (`version`, `doctor`) and CI were done before that.
- M1 issues:
  1. G.711 codec (PCMU/PCMA) and WAV I/O. Done.
  2. RTP sender and receiver with 20 ms pacing. Done.
  3. SIP UAC: INVITE, ACK, BYE, digest auth, SDP offer/answer. Done.
  4. Local test agent: a SIP UAS that answers calls and either echoes the caller's audio or plays a WAV file. Done.
  5. Recording both directions on a shared clock: caller.wav, agent.wav and a stereo mix, plus events.json with timestamps for INVITE sent, ringing, answered, first RTP in, first RTP out and BYE.
  6. End-to-end loopback test in CI: voiceqa calls the local test agent, sends a tone, receives the echo, and checks the audio and the measured round-trip latency.
  7. Session refresh: answer an in-dialog re-INVITE or UPDATE with 200 OK and the current SDP instead of 488. Also handle INVITEs without an SDP offer, which some providers send: put our offer in the 200 OK and take the answer from the ACK.
- Next issue: issue 5 (recording both directions on a shared clock).
- Known problems:
  - An in-dialog re-INVITE (for example a session timer refresh) is answered with 488 by both the UAC and the test agent, which some providers treat as a reason to drop the call. Issue 7 fixes this.
  - The test agent sends its in-dialog requests (BYE) to the address the INVITE came from, not along the Route set, so it does not work behind a proxy that needs Record-Route followed. It answers at once, so CANCEL never stops a call. INVITEs without an SDP offer get 488 from the agent and are not handled by the UAC either; issue 7 covers both.
  - SIP is UDP and IPv4 only, MD5 digest only, no REGISTER or CANCEL, no NAT rewrite of Contact from `received`/`rport`.
  - RTP has no RTCP, no DTMF and no symmetric RTP source check. `recording()` places audio by RTP timestamp only. Issues 5 and 6 need arrival times on a shared clock for the recordings and the round-trip latency.
  - On Windows the Proactor loop does not call `connection_lost` when a UDP transport closes with a send in flight. The SIP transport works around it with `abort()` after two yields.

## Decisions

- Python 3.11+, src layout, package name `voiceqa`, built with hatchling.
- CLI with Typer. Config and data models with Pydantic and PyYAML. Audio math with NumPy.
- Ruff for linting and formatting, pytest for tests.
- CI must pass on Windows, macOS and Linux.
- The core stays Apache-2.0 compatible. PJSIP (GPL) and Piper (GPL-3.0) are not core dependencies. GPL engines can only be added later as optional plugins that users install separately.
- Telephony is our own minimal pure-Python asyncio SIP UAC plus RTP, with G.711 PCMU and PCMA in 20 ms frames.
- The media layer sits behind a small interface so it can be rewritten in Rust (PyO3) later without touching the rest of the code.
- G.711 is our own NumPy code because audioop was removed in Python 3.13. It is bit-exact with the Sun reference g711.c that audioop used, and the tests compare against audioop on Python versions that still have it.
- Audio inside VoiceQA is int16. Mono is a 1-D array, multichannel is (frames, channels). Audio functions are pure (arrays in, arrays out) so they can move to Rust later.
- Resampling uses our own polyphase windowed-sinc filter instead of adding scipy as a dependency.
- RTP pacing uses absolute deadlines (packet n is due at t0 + n * 20 ms). Up to 5 late frames are sent back to back. When the loop is blocked for longer, missed slots are skipped and the RTP timestamp jumps forward, so the far end hears a gap of the right length instead of a burst.
- SIP sends every request to the configured server, used as an outbound proxy. A 2xx with unusable SDP is ACKed and then ended with BYE before the error is raised.
- The SIP UAS (`sip/uas.py`) gets media from a callback that returns a `MediaSession`, so it does not import RTP either. The test agent (`telephony/local/agent.py`) wires it to `RtpSession`. The UAS does not send BYE before the ACK arrives or the ACK wait (64 * T1) runs out, since the caller may not know the dialog yet.
- Echo in the test agent does constant work per frame: `RtpSession.listen()` hands each new decoded frame to a bounded buffer (1 s, oldest dropped when full), which is flushed to `send()` every 20 ms. It never reads the recording.

## Media interface

`src/voiceqa/telephony/media.py` is the contract between SIP and RTP. Neither side imports the other.

1. RTP binds a local UDP socket and reports it as `LocalMedia(host, port, payload_types, ptime_ms)`.
2. SIP puts `LocalMedia` in the SDP offer: `c=` host, `m=audio` port, payload types in preference order (PCMU 0, PCMA 8), `a=ptime:20`.
3. SIP parses the SDP answer into `RemoteMedia(host, port, payload_type, ptime_ms)`. A rejected stream (port 0) or no common codec is a SIP error, not a `RemoteMedia`.
4. RTP is started with `RemoteMedia` and implements the `MediaSession` protocol: `local`, `start(remote)`, `send(samples)`, `drain()`, `recording()`, `close()`.

Audio crossing the boundary is int16 mono at 8 kHz in 160-sample (20 ms) frames. Only 20 ms ptime is supported for now.

## Layout

```
src/voiceqa/
  audio/            codecs, resampling, audio analysis
    g711.py         PCMU and PCMA encode and decode
    wav.py          16-bit PCM WAV read and write
    dsp.py          downmix, resampling, test tones
  telephony/media.py  SIP to RTP contract (LocalMedia, RemoteMedia, MediaSession)
  telephony/rtp/    RTP packet codec, recorder, paced session
  telephony/sip/    SIP messages, SDP, digest auth, transactions, call flow (UAC), UAS
  telephony/local/  local test agent (echo or WAV playback) for loopback tests and offline runs
  engine/           scenario runner and reporting
  cli/              Typer commands
tests/
docs/
examples/scenarios/
```

## Roadmap

- M1: local SIP call with two-way audio and recording.
- M2: local STT and TTS.
- M3: YAML scenario engine.
- M4: audio QA (latency, turn-taking, audio quality metrics).
- M5: dashboard for browsing runs and reports.
- M6: adaptive caller driven by a local LLM.
- M7: provider adapters and bring-your-own-number.
