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
- faster-whisper speech-to-text backend (`create_stt("faster-whisper", ...)`), installed with the optional `[stt]` extra. Audio is resampled to 16 kHz with our own resampler. Defaults: CPU, int8, `base.en`, word timestamps and VAD filter on. A model that is not on disk is an error naming the pull command, so nothing is downloaded during a call.
- `voiceqa models pull stt <model>` and `voiceqa models list`. Models live in the Hugging Face cache, or in `--models-dir` / `VOICEQA_MODELS_DIR`.
- `voiceqa doctor` shows whether the `[stt]` extra is installed, the models folder in use and each cached model with its size.
- `voiceqa bench stt [--stt-model M] [--models-dir D] [--json]`: model load time, real-time factor and per-utterance latency on this CPU, measured on the test clip after one warm-up run. Informational only.
- Test speech fixture: 18 s of a public-domain LibriVox reading of The Gift of the Magi at 8 kHz, with utterance times, the book's text and a SOURCE.md. It also ships in the wheel for `voiceqa bench stt`.
- Real-model tests (pytest marker `model`) run tiny.en against the fixture and check the word error rate. CI caches tiny.en, runs them on all nine jobs and adds the bench numbers to the job summary.
- `voiceqa.speech.wer`: word error rate (word-level edit distance over reference words) with text normalised for case and punctuation, apostrophes inside words kept and numbers left as written. The real-model tests use it instead of their own copy.
- `SttWorker` (`voiceqa.speech.worker`): runs an STT engine in a child process started with "spawn" on every OS. The engine is built in the child from its registry name and plain keyword arguments and loaded once. `transcribe()` is async and never blocks the event loop, `submit()` returns a future, `close()` lets queued jobs finish within a timeout and then terminates the child. A child that dies or an engine that fails to load fails the pending jobs with `SttWorkerError` instead of hanging.
- `FakeStt` takes `delay` (seconds per job) and `error` (raise on every job) for testing slow and failing jobs.
- `voiceqa transcribe RUN_DIR [--stt-model M] [--models-dir D]`: transcribes caller.wav and agent.wav in the STT worker and writes transcript.json, with each segment tagged `caller` or `agent`, timed on the same zero as events.json and the WAVs, and sorted by start time. Long tracks are read in blocks and sent in chunks of up to ten minutes, cut at a quiet frame. A missing `[stt]` extra or model is an error naming the fix (exit 1), and a folder without the two WAVs is exit 2.
- `voiceqa call --transcribe [--stt-model M] [--models-dir D]`: writes transcript.json once an answered call has ended. If the transcript cannot be made, the error and its fix go to stderr and the exit code stays the call's.
- `SapiTts` (Windows, through PowerShell and System.Speech) and `SayTts` (macOS `say`) text to speech engines, as `--tts sapi` and `--tts say`. The text never goes into a command line, so quotes, `$` or backticks in it cannot run anything.
- An unknown `--voice` with the `say` engine is an error naming `voiceqa say --list-voices`, instead of `say` quietly using its default voice. Voice names match exactly.
- When `say` exits 0 with almost no audio (under 0.1 s, or silent) for text with letters or digits in it, the `say` engine runs it once more and raises an error if the second try is no better, so a call never sends silence and nothing empty is cached.
- `PluginTts`: runs a separately installed `voiceqa-tts-<name>` program as `--tts plugin:<name>`, following the plugin contract (JSON on stdin, WAV at `--out`). It has a timeout that kills the whole process tree, and shows the plugin's stderr when it fails.
- Per-user cache of synthesized speech, keyed by engine, voice and text, in `%LOCALAPPDATA%/voiceqa/cache/tts`, `~/Library/Caches/voiceqa/tts` or `~/.cache/voiceqa/tts` (`VOICEQA_CACHE_DIR` moves it). Written atomically, so two runs never read a half-written file.
- `voiceqa say TEXT -o OUT.wav [--tts NAME] [--voice V] [--prompts INDEX]` to try a voice, and `voiceqa say --list-voices`.
- `voiceqa call --say TEXT` speaks the text once the call is answered, instead of the tone or a WAV. The speech is made before dialing, so a TTS failure never costs a call. The default engine is sapi on Windows, say on macOS and wav prompts elsewhere.
- Speech loopback end-to-end test: `voiceqa agent` (echo) and `voiceqa call --say ... --transcribe --stt-model tiny.en` run as subprocesses, and the caller and agent transcripts must match the spoken text within a word error rate bound. It speaks with sapi on Windows, say on macOS and a WAV prompt cut from the LibriVox fixture on Linux, and runs with the other real-model tests in CI.
- README: speech setup (the `[stt]` extra, `voiceqa models pull`, model choice and sizes, `voiceqa bench stt`), text to speech with the OS engines and WAV prompts, `voiceqa say`, `voiceqa call --say --transcribe`, `voiceqa transcribe`, the TTS plugin contract, and why Piper and Kokoro can only be plugins.
- docs/ARCHITECTURE.md: M2 is done. New "Speech interface" section next to "Media interface", the loopback test and help markup decisions, and M3 (YAML scenario engine) as the next milestone, which takes over live per-turn STT with end-of-turn detection.
- docs/ARCHITECTURE.md: M2 issue 4 is done, with the TTS engine, plugin runner and cache decisions and their known problems.
- docs/ARCHITECTURE.md: M2 issue 3 is done; the worker process, transcript.json and WER decisions are written down, along with how far off faster-whisper's word start times are.
- `voiceqa.speech` package: `Speech` (int16 mono samples and rate), the `TextToSpeech` and `SpeechToText` protocols, and the `Word`, `Segment`, `Transcript` models that give transcript.json one schema.
- `WavPromptTts`: maps text to a WAV file from a YAML index (text to file path, relative to the index file's folder) or from an in-memory mapping. Text is matched after stripping whitespace, case sensitive. Unknown text is a `ValueError` naming it, never silence. Keeps the file's own sample rate and downmixes stereo to mono.
- `FakeTts` (tone bursts whose length follows the text) and `FakeStt` (a scripted `Transcript`, given as a `Transcript` or as plain data) for tests that need a TTS or STT engine without real speech.
- `create_tts(name, **kwargs)` and `create_stt(name, **kwargs)`: look up a TTS or STT engine by name (`wav`, `fake`, `sapi`, `say`, `faster-whisper`, `plugin:<name>`) from plain, picklable keyword arguments only. An unknown name raises `ValueError` listing the known engines.

### Changed

- `RtpSession` no longer keeps the received audio in memory, and `recording()` is gone. Received frames are placed by RTP timestamp on the local clock, and a short burst of late packets no longer shifts the rest of the call.
- Media timing uses `time.perf_counter` instead of the event loop clock, which only ticks every 15.6 ms on Windows before Python 3.13.
- Session refreshes no longer get 488. The UAC and the test agent answer an in-dialog re-INVITE or UPDATE with 200 and the SDP they sent before, resend that 200 until the ACK arrives, and keep media running. A re-INVITE without SDP gets the same SDP as an offer, and the answer in the ACK is ignored. An offer with no codec in common still gets 488, and the call stays up. A request with an older CSeq than the last one from the far end gets 500.
- Our INVITE, our 200 to an INVITE, and refresh responses carry `Allow: INVITE, ACK, BYE, CANCEL, OPTIONS, UPDATE`.
- The UAC answers resent requests from the far end with the same response instead of handling them again.
- The test agent accepts an INVITE without an SDP offer instead of sending 488. It puts its offer in the 200, takes the answer from the ACK, and only then starts media. An ACK without a usable answer ends the call with BYE.
- `voiceqa agent --answer-delay SECONDS` makes the test agent ring that long before it answers.
- The test agent honours CANCEL before it answers: 200 to the CANCEL, 487 to the INVITE (resent until its ACK), media closed, no call. A CANCEL after the answer still gets 200 and changes nothing.
- The test agent no longer starts media when a BYE arrives right after the ACK, before media was started.
- docs/ARCHITECTURE.md marks M1 as done and names M2 (local STT and TTS) as the current milestone.
- docs/ARCHITECTURE.md lists the M2 issues and the decisions behind them: faster-whisper as an optional extra, WAV prompts plus OS speech engines for TTS, and neural TTS engines only as separately installed plugins for licence reasons.
- `run_call` resolves the SIP server once and hands the address to `place_call` (new `server` argument and public `resolve_server`), instead of looking it up a second time. An unresolvable server now reads "cannot resolve SIP server HOST: ..." in the call summary.
- `create_tts("sapi")`, `create_tts("say")` and `create_tts("plugin:<name>")` now build engines instead of raising `NotImplementedError`. An OS engine on the wrong OS raises `TtsError`. `create_stt("plugin:<name>")` still raises `NotImplementedError`, since there are no STT plugins.

### Fixed

- `test_reinvite_mid_call_does_not_interrupt_the_echo` counted silence frames slipped into the echo and failed on one CI job under load. It now checks energy only, matching the already-accepted contract that the agent may slip a silence frame into the echo when its queue runs dry for a slot (see `test_recording_of_an_echo_call_lines_up_on_one_clock`).
- The real-model STT tests load tiny.en with `cpu_threads=1` instead of the faster-whisper default, so the test process keeps fewer CTranslate2 worker threads alive after the model tests finish (that thread pool has no release call, so it stays for the rest of the pytest run).
- `voiceqa call --help` lost the word "[stt]" (it read "needs the  extra") because Rich took it for a style tag. It is escaped now, and a test checks every bracketed word in every command's help.
- `voiceqa call --transcribe` dialed before checking the `[stt]` extra and the model, so a missing one was only reported after a real call had already gone out. Both are now checked before dialing; a missing extra or model exits 1 and nothing is dialed.
- `--stt-model` and `--models-dir` on `voiceqa call` were silently ignored without `--transcribe`. They now exit 2 when given without it.
- The `[stt]` extra install hint (`pip install voiceqa[stt]`) named a PyPI package VoiceQA is not published as, and the unquoted brackets fail in zsh. It now reads `pip install -e ".[stt]"` from a checkout.
- `voiceqa say` and `voiceqa call --say` could crash with a traceback instead of a clean error when a WAV prompt file named in the index was missing, or when the output WAV could not be written (for example `-o` into a folder that does not exist). Both now exit 1 with a plain message.
- `voiceqa transcribe` crashed with a traceback on a corrupt or non-16-bit WAV instead of exiting 1 with a plain message naming the file.
