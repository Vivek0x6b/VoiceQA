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
- Call recording on a shared clock: `CallRecorder` writes caller.wav, agent.wav, a stereo mix.wav and events.json (INVITE sent, ringing, answered, first RTP in and out, BYE). Audio is streamed to disk as it arrives, gaps are filled with silence, and both tracks end at the same length.
- `place_call(on_event=...)` reports call events. `RtpSession.record(sink)` hands every sent and received frame to a sink with its time.
- `place_call` reports a `call_rejected` event (status and reason) when the final response is a failure, so a rejected or unauthorized call shows up in events.json.
- `WavWriter` for writing a WAV as audio comes in, and `read_wav_blocks` for reading one in pieces.
- `voiceqa call` command: places one SIP call, sends a WAV file or a default test tone once answered, records both sides, and prints a plain text summary (answered, duration, frames sent and received, first-audio latency, recording folder). Reports an unresolvable SIP server or an answer with no usable codec as an unanswered call with a short reason, the same as a rejected call, and a non-numeric port in `--server` as a bad-options error (exit 2). Ctrl+C during a call also prints the recording folder. Exit code reflects whether the call was answered.
- `first_loud_frame` in `voiceqa.audio.analysis`: finds the first 20 ms frame above a given RMS threshold in a WAV file, reading it in blocks. Used to compute first-audio latency for `voiceqa call`.
- `voiceqa calibrate` command: calls a local echo agent with a train of short tone pulses instead of a real conversation, measures each pulse's round-trip delay, and writes `calibration.json` (median, p95, min, max, pulses sent and matched) into the recording folder. This is VoiceQA's own round-trip floor, meant to be subtracted from latencies measured against a real voice agent. Exits 1 if fewer than half the pulses came back, and prints the call's status when the echo agent did not answer.
- `pulse_round_trip_delays` in `voiceqa.audio.analysis`: finds pulse onsets in a caller and an agent recording and pairs them up into round-trip delays, used by `voiceqa calibrate`.
- `src/voiceqa/__main__.py`, so `python -m voiceqa` runs the CLI without the console script being on PATH.
- End-to-end test: runs `voiceqa agent` and `voiceqa call` as real subprocesses through `python -m voiceqa`, and checks the exit code, the printed summary, the recorded files, the tone frequency and onset timing in agent.wav, and the order of events in events.json. Runs on Windows, macOS and Linux in CI.
- README quick start section showing `voiceqa agent`, `voiceqa call` and `voiceqa calibrate` with real output and what lands in the recording folder.

### Changed

- `RtpSession` no longer keeps the received audio in memory, and `recording()` is gone. Received frames are placed by RTP timestamp on the local clock, and a short burst of late packets no longer shifts the rest of the call.
- Media timing uses `time.perf_counter` instead of the event loop clock, which only ticks every 15.6 ms on Windows before Python 3.13.
- Session refreshes no longer get 488. The UAC and the test agent answer an in-dialog re-INVITE or UPDATE with 200 and the SDP they sent before, resend that 200 until the ACK arrives, and keep media running. A re-INVITE without SDP gets the same SDP as an offer, and the answer in the ACK is ignored. An offer with no codec in common still gets 488, and the call stays up. A request with an older CSeq than the last one from the far end gets 500.
- Our INVITE, our 200 to an INVITE, and refresh responses carry `Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, UPDATE`.
- The UAC answers resent requests from the far end with the same response instead of handling them again.
- The test agent accepts an INVITE without an SDP offer instead of sending 488. It puts its offer in the 200, takes the answer from the ACK, and only then starts media. An ACK without a usable answer ends the call with BYE.
