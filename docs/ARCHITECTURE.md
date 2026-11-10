# Architecture

VoiceQA is a local-first QA lab for AI voice agents. It places real SIP calls to an agent, speaks with local TTS, transcribes with local STT (faster-whisper), and writes JSON and HTML reports. No audio, transcript or report leaves the user's machine.

## Current status

- Current milestone: M2 (local STT and TTS). Issues: 1 speech interfaces, fakes and WAV prompts; 2 faster-whisper STT backend; 3 call transcripts and `voiceqa transcribe`; 4 OS TTS engines and the plugin runner; 5 loopback check and docs.
- Finished M2 issues: issue 1, speech interfaces (`Speech`, `TextToSpeech`, `SpeechToText`), the `Word`/`Segment`/`Transcript` models, `WavPromptTts`, `FakeTts`/`FakeStt`, and the `create_tts`/`create_stt` registry. Issue 2, the faster-whisper STT backend behind the `[stt]` extra, `voiceqa models pull/list`, model info in `voiceqa doctor`, the LibriVox test clip, real-model tests in CI with a cached tiny.en, and `voiceqa bench stt`. Issue 3, the STT worker process (`SttWorker`), transcript.json for a run folder, `voiceqa transcribe`, `voiceqa call --transcribe` and the WER helper. The M1 carry-over where `run_call` resolved the SIP server a second time is fixed in the same issue.
- Next issue: M2 issue 4, OS TTS engines and the plugin runner.
- M2 known problems: the test clip's reference text is the book's text and was not checked by ear. Both tiny.en and base.en hear "Stella, the in slender" for its first words, "Della, being slender", so the reader may differ from the book there. The real-model WER bounds allow for it. CTranslate2 keeps its worker thread pool alive for the life of the process once a model is loaded (confirmed locally: loading tiny.en adds about twenty OS threads that `del` and `gc.collect()` do not remove, and `unload_model()` does not touch them either). The real-model tests load tiny.en with `cpu_threads=1` to keep that footprint small, since this is real, measured leftover state in the pytest process, not just a theory. Transcription for calls runs in the worker child, so those threads stay there and go away with it. faster-whisper's word start times are not precise enough to measure latency with: on the test clip, tiny.en puts the first word 0.25 to 0.3 s before the speech after leading silence, and at 0.0 when the speech starts within the first few frames (measured 2026-11-09). Shifts between two placements of the same speech come back exact to within a few hundredths of a second. M4 latency should use energy onsets (`first_loud_frame`), not STT word times. A chunk boundary inside a long track (one every ten minutes) can still cut a word when the quietest frame in the last two seconds is mid-word. The worker child exits on its own within about a second if its parent dies without closing it; there is no test for that path.
- M1 (local SIP call with two-way audio and recording) is done. All seven issues are finished and CI passes on Windows, macOS and Linux for Python 3.11 to 3.13. What M1 gives: `voiceqa agent` (a local echo or WAV playback agent), `voiceqa call` (one SIP call with recording of both sides and events.json) and `voiceqa calibrate` (VoiceQA's own round-trip floor). Its limits are listed under known problems below.
- Finished M1 issues: issue 1 (G.711 codec and WAV I/O), issue 2 (RTP sender and receiver with 20 ms pacing), issue 3 (SIP UAC with INVITE, ACK, BYE, digest auth and SDP offer/answer), issue 4 (local test agent), issue 5 (recording both directions on a shared clock), issue 6 (`voiceqa call` and `voiceqa calibrate` commands, plus an end-to-end subprocess test in CI and a README quick start), issue 7 (session refresh, INVITEs without an SDP offer, and CANCEL before the agent answers). The project scaffold, CLI (`version`, `doctor`) and CI were done before that.
- M1 issues:
  1. G.711 codec (PCMU/PCMA) and WAV I/O. Done.
  2. RTP sender and receiver with 20 ms pacing. Done.
  3. SIP UAC: INVITE, ACK, BYE, digest auth, SDP offer/answer. Done.
  4. Local test agent: a SIP UAS that answers calls and either echoes the caller's audio or plays a WAV file. Done.
  5. Recording both directions on a shared clock: caller.wav, agent.wav and a stereo mix, plus events.json with timestamps for INVITE sent, ringing, answered, first RTP in, first RTP out and BYE. Done.
  6. `voiceqa call` and `voiceqa calibrate` commands, an end-to-end subprocess test in CI, and a README quick start. Done.
  7. Session refresh: answer an in-dialog re-INVITE or UPDATE with 200 OK and the current SDP instead of 488. Also handle INVITEs without an SDP offer, which some providers send: put our offer in the 200 OK and take the answer from the ACK. Done, along with CANCEL handling in the test agent and `voiceqa agent --answer-delay`.
- Known problems:
  - A session refresh (re-INVITE or UPDATE) is answered with 200 and the SDP we sent before, and media keeps running as it was. A re-INVITE that changes the far end's address, port or codec still gets our current SDP, and our media is not re-pointed to the new address. Session timers (RFC 4028) are not supported: a refresh with Session-Expires gets a plain 200.
  - A 200 to a refresh that never gets its ACK is only resent until 64 * T1, then dropped. No BYE is sent, though RFC 3261 says the UAS should send one.
  - The test agent treats an INVITE with a multipart body (for example SDP plus ISUP) as an INVITE without an offer. It puts its own offer in the 200 and expects the answer in the ACK.
  - The test agent sends its in-dialog requests (BYE) to the address the INVITE came from, not along the Route set, so it does not work behind a proxy that needs Record-Route followed.
  - SIP is UDP and IPv4 only, MD5 digest only, no REGISTER, no NAT rewrite of Contact from `received`/`rport`. The UAC cannot send CANCEL.
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
- A new INVITE goes through one answer path in the UAS: open media, build our SDP (an answer to the offer, or our own offer when the INVITE has none), send 180, wait `answer_delay` seconds (0 by default), send 200. A CANCEL that matches the INVITE's branch before the 200 gets 200, and the INVITE gets 487, resent on timer G until its ACK or 64 * T1. The media is closed and no call is created. A CANCEL after the 200 gets 200 and nothing else (RFC 3261 9.2). Without an offer in the INVITE, media starts only when the ACK brings a usable answer, and not at all if the call ended first.
- Session refreshes (in-dialog re-INVITE or UPDATE) get 200 with the SDP we sent last, with the same o= version (RFC 6337). Media is never closed, restarted or re-pointed by a refresh. Only an offer with no codec in common gets 488, and the call stays up.
- Echo in the test agent does constant work per frame: `RtpSession.listen()` hands each new decoded frame to a bounded buffer (1 s, oldest dropped when full), which is flushed to `send()` every 20 ms.
- Media timing and call events use one clock, `media_clock` (`time.perf_counter`). Not `loop.time()`, which ticks in 15.6 ms steps on Windows before Python 3.13. The RTP sender's deadlines run on it too, so a sent frame's time is its exact slot.
- `RtpSession` keeps no audio. `record(sink)` hands every frame sent and received to a `MediaSink` with its time on `media_clock`. Sent frames are timed by their slot deadline. Received frames are placed by RTP timestamp, anchored at the first packet's arrival (`rtp/timeline.py`), so jitter and reordering do not move audio. A packet more than 0.5 s late or 0.1 s early against that anchor is placed at its arrival time instead. Only when that lasts 0.5 s is the stream anchored again, so a burst after a stall does not shift the rest of the call. Duplicates are found in a 64-packet history.
- `CallRecorder` (`telephony/recording.py`) streams each side to its own WAV as frames arrive. Frames wait 100 ms for reordered packets, then are written. A frame within half a frame of the end of the track is appended, a later one is preceded by silence, an earlier one is dropped and counted. Silence for long gaps is written in 1 s blocks. When the call ends both tracks are padded to the same length, mix.wav is built from the two files a second at a time, and events.json is written. Memory stays flat however long the call is. The WAV header is fixed after every write, so the files are valid even if the process dies mid-call (mix.wav and events.json are only written at the end).
- Call events come from `place_call(on_event=...)`, so SIP still knows nothing about recording. events.json times are seconds since the recorder was created, the same zero as sample 0 of every WAV. Events are invite_sent, ringing, answered, call_rejected (final failure status and reason), bye_sent and bye_received.
- The test agent does not record.
- Speech engines sit behind two small interfaces: text to speech returns int16 mono samples with their rate, speech to text returns a transcript of timed segments and words. Engines are picked by name, so the CLI and later YAML scenarios choose them with a string.
- `create_tts(name, **kwargs)` and `create_stt(name, **kwargs)` build an engine from its registry name plus plain, picklable keyword arguments only (str, int, float, paths, dicts), because the STT worker (a later M2 issue) builds engines inside a spawned child process, where anything else could not cross the pickle boundary. `FakeStt`'s scripted transcript is accepted as a `Transcript` or as plain data for the same reason.
- `WavPromptTts` matches text after stripping leading and trailing whitespace, case sensitive. A YAML index's file paths are relative to the index file's own folder, not a separately given one.
- The subprocess TTS plugin contract (`voiceqa-tts-<name>` on PATH, JSON on stdin, a WAV file at `--out`) is written down in `voiceqa.speech.registry`'s docstring now, so the plugin runner issue implements exactly that interface instead of inventing one.
- STT is faster-whisper, installed with the optional `[stt]` extra so the core install stays small. Default model `base.en` (148 MB), changed with `--stt-model`. `small.en` (486 MB) is better on 8 kHz phone audio but slower on CPU. Models live in the Hugging Face cache unless `--models-dir` says otherwise, and are only downloaded when the user asks or on first use with a message.
- TTS in M2 is WAV prompts on every OS, plus the OS engine on Windows (SAPI) and macOS (`say`). Linux uses WAV prompts only. No espeak-ng.
- Neural TTS engines (Kokoro, Piper) can only be added as separately installed plugins that VoiceQA runs as a subprocess and never imports. Piper is GPL-3.0. Kokoro itself is Apache-2.0 and its English pipeline runs without espeak-ng (it uses it only for words missing from its dictionary, and drops those words without it), but installing it pulls in `phonemizer-fork` (GPLv3+) and `espeakng-loader`, which ships the espeak-ng library (GPL-3.0). Checked on 2026-11-09 against kokoro 0.9.4 and misaki 0.9.4.
- CI caches the `tiny.en` model (78 MB) and runs the real-model STT tests on all three OSes.
- Test speech is a public-domain LibriVox clip, downsampled to 8 kHz, with its source and licence noted next to it in the repo.
- Model sizes (whole Hugging Face repo, checked 2026-11-09): tiny.en 78.1 MB, base.en 147.8 MB, small.en 486.1 MB.
- `FasterWhisperStt` loads its model in the constructor and takes only plain keyword arguments (`model`, `models_dir`, `device`, `compute_type`, `language`, `beam_size`, `word_timestamps`, `vad_filter`, `cpu_threads`), so the issue 3 worker can build it in a spawned child through `create_stt`. It never downloads: a model that is not on disk raises `ModelNotAvailableError` naming `voiceqa models pull stt <model>`. Downloads happen only through `pull_stt_model`, which the CLI calls in `voiceqa models pull` and, on first use, in `voiceqa bench stt`, after printing what it downloads and where.
- The models folder uses the Hugging Face hub cache layout in every case, so `--models-dir` and `VOICEQA_MODELS_DIR` are passed to faster-whisper as its download root. Order: `--models-dir`, then `VOICEQA_MODELS_DIR`, then the Hugging Face cache (worked out like huggingface_hub does: `HF_HUB_CACHE`, `HF_HOME/hub`, `XDG_CACHE_HOME/huggingface/hub`, `~/.cache/huggingface/hub`). `speech/model_cache.py` reads this without importing faster-whisper, so `voiceqa doctor` and `voiceqa models list` work without the extra.
- A model's size on disk is the sum of the files its snapshot links resolve to, each counted once. huggingface_hub 2.x links blobs again into a shared `blobs/` store at the cache root, so the repo folder alone does not hold the model bytes. CI therefore caches the whole models folder (it only holds tiny.en), not one repo folder.
- The `[stt]` extra pulls in faster-whisper (MIT), ctranslate2 (MIT), onnxruntime (MIT), PyAV (BSD-3-Clause, wheels bundle LGPL FFmpeg, dynamically linked), tokenizers and huggingface_hub (Apache-2.0) and their small dependencies. No GPL package. Checked 2026-11-09 with faster-whisper 1.2.1 on Python 3.11.
- `voiceqa bench stt` runs in the CLI process, not in the issue 3 worker: it measures the model, not the IPC. The test clip ships in the wheel (`voiceqa/data/speech`) so the bench also works outside a source checkout.
- Real-model tests carry the `model` marker. They skip with the reason when the extra or tiny.en is missing, except when `VOICEQA_REQUIRE_MODEL_TESTS` is set (CI sets it), where a missing model fails.
- STT for calls runs in a separate worker process (`speech/worker.py`, `SttWorker`), never in the process or event loop that runs media. The child is started with the "spawn" start method on every OS, so what we test is what Windows does. The worker takes an engine spec (registry name plus plain keyword arguments) and builds the engine in the child with `create_stt`; an engine instance is never pickled. The model is loaded once per worker. Jobs (job id, int16 samples, rate) go over a multiprocessing queue, results come back over a pipe as plain data, and a reader thread in the parent resolves each job's future. `transcribe()` awaits that future, so the event loop keeps running while a job runs. A child that dies, or an engine that fails to load, fails every pending job with `SttWorkerError` instead of hanging. `close()` sends a sentinel, lets queued jobs finish within a timeout (10 s by default), then terminates and, if needed, kills the child. Leaving the `with` block on an error stops the child at once. The child ignores Ctrl+C; the parent decides when it stops. Keeping the engine in the child also keeps CTranslate2's permanent thread pool out of the caller's process. M3 reuses this worker for live per-turn transcription during calls.
- transcript.json uses the `Transcript` schema. Each track is transcribed on its own and segments are tagged `caller` or `agent`, timed in seconds from sample 0 of the WAVs (the zero of events.json), and sorted by start time. `language` is the tracks' languages, comma-separated if they differ. Tracks are read in blocks and sent in chunks of up to ten minutes, each cut at the quietest 20 ms frame in its last two seconds, so a long call is never one array or one job. `engine/transcribe.py` takes any object with an async `transcribe()`, in practice an `SttWorker`.
- `voiceqa transcribe` and `voiceqa call --transcribe` never download a model. A missing `[stt]` extra or model is an error naming `pip install voiceqa[stt]` or `voiceqa models pull stt <model>`, checked before the worker starts. `voiceqa transcribe` exits 1 for that and 2 for a run folder without caller.wav or agent.wav. `voiceqa call --transcribe` transcribes only after the call has ended and the recording is closed, and only if the call was answered. If the transcript cannot be made, the call summary and recording stand, the error goes to stderr, and the exit code is still the call's (0 answered, 1 not). A script that needs the transcript should check that transcript.json exists.
- The WER helper (`speech/wer.py`) is our own word-level Levenshtein distance divided by the reference word count. Text is case folded, punctuation dropped, hyphens split words, inner apostrophes kept, and numbers stay as written ("20" and "twenty" differ). An empty reference raises `ValueError`.
- `run_call` resolves the SIP server once with `resolve_server` (from the SIP layer) and passes the address to `place_call(server=...)`, which only resolves on its own when it is not given one.

## STT CPU baseline (informational)

Measured with `voiceqa bench stt` on 2026-11-09: Windows 11, Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz (6 cores, 12 threads), Python 3.11.3, faster-whisper 1.2.1, ctranslate2 4.8.5, CPU int8, on the 18.1 s test clip (4 utterances). Real-time factor is processing time divided by audio length. These are for comparison only, nothing gates on them, and they move by 10 to 40 percent between runs.

| Model | Model load | Real-time factor | Utterance median | p95 | max |
|-------|-----------|------------------|------------------|-----|-----|
| tiny.en | 0.51 s | 0.084 | 600 ms | 857 ms | 899 ms |
| base.en | 0.70 s | 0.139 | 1009 ms | 1064 ms | 1072 ms |

Word error rate on the same clip (book text as reference): tiny.en 0.128, base.en 0.064.

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
    transcribe.py   transcript.json for a run folder, both tracks through an STT worker
  speech/           text to speech and speech to text interfaces
    interfaces.py   Speech, TextToSpeech, SpeechToText
    models.py       Word, Segment, Transcript (transcript.json schema)
    fakes.py        FakeTts, FakeStt for tests
    wav_prompts.py  WavPromptTts: text to WAV file, from a YAML index or a mapping
    registry.py     create_tts, create_stt by name, plus the subprocess plugin contract
    whisper.py      FasterWhisperStt (needs the [stt] extra)
    model_cache.py  models folder, cached model listing and size, model download
    bench.py        STT speed check on the test clip (voiceqa bench stt)
    worker.py       SttWorker: STT engine in a spawned child process
    wer.py          word error rate
  cli/              Typer commands
  __main__.py       lets `python -m voiceqa` run the CLI
tests/
  data/speech/      LibriVox test clip (8 kHz), utterances.json, SOURCE.md
docs/
examples/scenarios/
```

## Roadmap

- M1: local SIP call with two-way audio and recording. Done.
- M2: local STT and TTS.
- M3: YAML scenario engine.
- M4: audio QA (latency, turn-taking, audio quality metrics).
- M5: dashboard for browsing runs and reports.
- M6: adaptive caller driven by a local LLM.
- M7: provider adapters and bring-your-own-number.
