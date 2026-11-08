# Changelog

All notable changes to this project are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Project scaffold with src layout and hatchling build.
- `voiceqa version` and `voiceqa doctor` commands.
- CI on Windows, macOS and Linux for Python 3.11 to 3.13.
- Current status section in docs/ARCHITECTURE.md.
- .gitattributes so text files always use LF line endings.
- G.711 PCMU and PCMA encode and decode in pure NumPy, bit-exact with the old audioop module so it works on Python 3.13.
- 16-bit PCM WAV read and write, stereo to mono downmix, and a sine test tone generator.
- `ruff format --check .` in CI.
- Media interface between SIP and RTP (LocalMedia, RemoteMedia, MediaSession).
- Polyphase windowed-sinc resampler with anti-alias filtering, used to bring audio to 8 kHz for calls. 60 seconds of 48 kHz audio takes about 0.1 s.
