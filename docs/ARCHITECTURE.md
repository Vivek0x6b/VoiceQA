# Architecture

VoiceQA is a local-first QA lab for AI voice agents. It places real SIP calls to an agent, speaks with local TTS, transcribes with local STT (faster-whisper), and writes JSON and HTML reports. No audio, transcript or report leaves the user's machine.

## Current status

- Current milestone: M1 (local SIP call with two-way audio and recording), in progress.
- Finished issues: issue 1, G.711 codec (PCMU/PCMA) and WAV I/O. The project scaffold, CLI (`version`, `doctor`) and CI were done before that.
- Next issue: M1 issue 2.
- Known problems: none.

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

## Layout

```
src/voiceqa/
  audio/            codecs, resampling, audio analysis
    g711.py         PCMU and PCMA encode and decode
    wav.py          16-bit PCM WAV read and write
    dsp.py          downmix, resampling, test tones
  telephony/sip/    SIP UAC and RTP
  telephony/local/  local loopback transport for tests and offline runs
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
