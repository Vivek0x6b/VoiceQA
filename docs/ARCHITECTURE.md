# Architecture

VoiceQA is a local-first QA lab for AI voice agents. It places real SIP calls to an agent, speaks with local TTS, transcribes with local STT (faster-whisper), and writes JSON and HTML reports. No audio, transcript or report leaves the user's machine.

## Current status

- Current milestone: M1 (local SIP call with two-way audio and recording), in progress.
- Finished issues: issue 1 (G.711 codec and WAV I/O), issue 2 (RTP sender and receiver with 20 ms pacing), issue 3 (SIP UAC with INVITE, ACK, BYE, digest auth and SDP offer/answer), issue 4 (local test agent), issue 5 (recording both directions on a shared clock), issue 6 (`voiceqa call` and `voiceqa calibrate` commands, plus an end-to-end subprocess test in CI and a README quick start). The project scaffold, CLI (`version`, `doctor`) and CI were done before that.
- M1 issues:
  1. G.711 codec (PCMU/PCMA) and WAV I/O. Done.
  2. RTP sender and receiver with 20 ms pacing. Done.
  3. SIP UAC: INVITE, ACK, BYE, digest auth, SDP offer/answer. Done.
  4. Local test agent: a SIP UAS that answers calls and either echoes the caller's audio or plays a WAV file. Done.
  5. Recording both directions on a shared clock: caller.wav, agent.wav and a stereo mix, plus events.json with timestamps for INVITE sent, ringing, answered, first RTP in, first RTP out and BYE. Done.
  6. `voiceqa call` and `voiceqa calibrate` commands, an end-to-end subprocess test in CI, and a README quick start. Done.
  7. Session refresh: answer an in-dialog re-INVITE or UPDATE with 200 OK and the current SDP instead of 488. Also handle INVITEs without an SDP offer, which some providers send: put our offer in the 200 OK and take the answer from the ACK.
- Next issue: issue 7 (session refresh and INVITEs without an SDP offer).
- Known problems:
  - An in-dialog re-INVITE (for example a session timer refresh) is answered with 488 by both the UAC and the test agent, which some providers treat as a reason to drop the call. Issue 7 fixes this.
  - The test agent sends its in-dialog requests (BYE) to the address the INVITE came from, not along the Route set, so it does not work behind a proxy that needs Record-Route followed. It answers at once, so CANCEL never stops a call. INVITEs without an SDP offer get 488 from the agent and are not handled by the UAC either; issue 7 covers both.
  - SIP is UDP and IPv4 only, MD5 digest only, no REGISTER or CANCEL, no NAT rewrite of Contact from `received`/`rport`.
  - RTP has no RTCP, no DTMF and no symmetric RTP source check.
  - If the far end's RTP timestamps jump backwards and stay there, up to 0.5 s of its audio is placed at arrival time and may be dropped as overlapping before the stream is anchored again. Our own event loop stalling for more than 0.5 s has the same effect on the audio queued in the socket during the stall.
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
- Echo in the test agent does constant work per frame: `RtpSession.listen()` hands each new decoded frame to a bounded buffer (1 s, oldest dropped when full), which is flushed to `send()` every 20 ms.
- Media timing and call events use one clock, `media_clock` (`time.perf_counter`). Not `loop.time()`, which ticks in 15.6 ms steps on Windows before Python 3.13. The RTP sender's deadlines run on it too, so a sent frame's time is its exact slot.
- `RtpSession` keeps no audio. `record(sink)` hands every frame sent and received to a `MediaSink` with its time on `media_clock`. Sent frames are timed by their slot deadline. Received frames are placed by RTP timestamp, anchored at the first packet's arrival (`rtp/timeline.py`), so jitter and reordering do not move audio. A packet more than 0.5 s late or 0.1 s early against that anchor is placed at its arrival time instead. Only when that lasts 0.5 s is the stream anchored again, so a burst after a stall does not shift the rest of the call. Duplicates are found in a 64-packet history.
- `CallRecorder` (`telephony/recording.py`) streams each side to its own WAV as frames arrive. Frames wait 100 ms for reordered packets, then are written. A frame within half a frame of the end of the track is appended, a later one is preceded by silence, an earlier one is dropped and counted. Silence for long gaps is written in 1 s blocks. When the call ends both tracks are padded to the same length, mix.wav is built from the two files a second at a time, and events.json is written. Memory stays flat however long the call is. The WAV header is fixed after every write, so the files are valid even if the process dies mid-call (mix.wav and events.json are only written at the end).
- Call events come from `place_call(on_event=...)`, so SIP still knows nothing about recording. events.json times are seconds since the recorder was created, the same zero as sample 0 of every WAV. Events are invite_sent, ringing, answered, call_rejected (final failure status and reason), bye_sent and bye_received.
- The test agent does not record.

## Media interface

`src/voiceqa/telephony/media.py` is the contract between SIP and RTP. Neither side imports the other.

1. RTP binds a local UDP socket and reports it as `LocalMedia(host, port, payload_types, ptime_ms)`.
2. SIP puts `LocalMedia` in the SDP offer: `c=` host, `m=audio` port, payload types in preference order (PCMU 0, PCMA 8), `a=ptime:20`.
3. SIP parses the SDP answer into `RemoteMedia(host, port, payload_type, ptime_ms)`. A rejected stream (port 0) or no common codec is a SIP error, not a `RemoteMedia`.
4. RTP is started with `RemoteMedia` and implements the `MediaSession` protocol: `local`, `start(remote)`, `send(samples)`, `drain()`, `record(sink)`, `close()`. A `MediaSink` has `sent(at, samples)` and `received(at, arrived, samples)`, with times on `media_clock`.

Audio crossing the boundary is int16 mono at 8 kHz in 160-sample (20 ms) frames. Only 20 ms ptime is supported for now.

## Layout

```
src/voiceqa/
  audio/            codecs, resampling, audio analysis
    g711.py         PCMU and PCMA encode and decode
    wav.py          16-bit PCM WAV read and write
    dsp.py          downmix, resampling, test tones
    analysis.py     first loud frame in a WAV, pulse onset and round-trip timing
  telephony/media.py  SIP to RTP contract (LocalMedia, RemoteMedia, MediaSession, MediaSink)
  telephony/recording.py  call recorder: caller.wav, agent.wav, mix.wav, events.json
  telephony/rtp/    RTP packet codec, receive timeline, paced session
  telephony/sip/    SIP messages, SDP, digest auth, transactions, call flow (UAC), UAS
  telephony/local/  local test agent (echo or WAV playback) for loopback tests and offline runs
  engine/           scenario runner and reporting
    call.py         places one call, sends audio, records both sides (voiceqa call)
    calibrate.py    measures VoiceQA's own round-trip floor against a local echo agent
  cli/              Typer commands
  __main__.py       lets `python -m voiceqa` run the CLI
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
