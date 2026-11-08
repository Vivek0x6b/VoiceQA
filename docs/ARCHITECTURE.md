# Architecture

VoiceQA is a local-first QA lab for AI voice agents. It places real SIP calls to an agent, speaks with local TTS, transcribes with local STT (faster-whisper), and writes JSON and HTML reports. No audio, transcript or report leaves the user's machine.

## Decisions

- Python 3.11+, src layout, package name `voiceqa`, built with hatchling.
- CLI with Typer. Config and data models with Pydantic and PyYAML. Audio math with NumPy.
- Ruff for linting and formatting, pytest for tests.
- CI must pass on Windows, macOS and Linux.
- The core stays Apache-2.0 compatible. PJSIP (GPL) and Piper (GPL-3.0) are not core dependencies. GPL engines can only be added later as optional plugins that users install separately.
- Telephony is our own minimal pure-Python asyncio SIP UAC plus RTP, with G.711 PCMU and PCMA in 20 ms frames.
- The media layer sits behind a small interface so it can be rewritten in Rust (PyO3) later without touching the rest of the code.

## Layout

```
src/voiceqa/
  audio/            codecs, resampling, audio analysis
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
