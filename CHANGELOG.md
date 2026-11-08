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
- `voiceqa call` command: places one SIP call, sends a WAV file or a default test tone once answered, records both sides, and prints a plain text summary (answered, duration, frames sent and received, first-audio latency, recording folder). Exit code reflects whether the call was answered.
- `first_loud_frame` in `voiceqa.audio.analysis`: finds the first 20 ms frame above a given RMS threshold in a WAV file, reading it in blocks. Used to compute first-audio latency for `voiceqa call`.
- `voiceqa calibrate` command: calls a local echo agent with a train of short tone pulses instead of a real conversation, measures each pulse's round-trip delay, and writes `calibration.json` (median, p95, min, max, pulses sent and matched) into the recording folder. Pulses sent is the number the pulse train was built with, not a count of onsets found in the recording. This is VoiceQA's own round-trip floor, meant to be subtracted from latencies measured against a real voice agent. Exits 1 if fewer than half the pulses came back, and prints the call's reason when the echo agent did not answer.
- `pulse_round_trip_delays` in `voiceqa.audio.analysis`: finds pulse onsets in a caller and an agent recording and pairs them up into round-trip delays, used by `voiceqa calibrate`. An onset right at the start of the recording, with no quiet before it to measure, is still counted.

### Changed

- `RtpSession` no longer keeps the received audio in memory, and `recording()` is gone. Received frames are placed by RTP timestamp on the local clock, and a short burst of late packets no longer shifts the rest of the call.
- Media timing uses `time.perf_counter` instead of the event loop clock, which only ticks every 15.6 ms on Windows before Python 3.13.
- `voiceqa call` no longer crashes with a traceback when the SIP server host cannot be resolved or the far end answers with no usable codec; both are now reported as an unanswered call with a short reason, same as a rejected call. A non-numeric port in `--server` (for example `host:abc`) is now a plain bad-options error (exit 2) instead of an unhandled exception. Ctrl+C during a call now also prints the recording folder, not just "call interrupted".
