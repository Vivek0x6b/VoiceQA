# VoiceQA

Open-source QA lab for AI voice agents. VoiceQA places real SIP phone calls to your agent, talks to it with local text-to-speech, transcribes the replies with local speech-to-text, and writes JSON and HTML reports covering the transcript, recordings, latency, turn-taking and audio quality.

Everything runs on your machine. Nothing is sent to the cloud. The only cost is your own VoIP/SIP provider for the phone line.

## Status

Pre-alpha. Only the project scaffold and a basic CLI exist so far. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the roadmap.

## Install

Requires Python 3.11 or newer.

```
git clone https://github.com/Vivek0x6b/VoiceQA.git
cd VoiceQA
python -m venv .venv
```

Activate the virtual environment:

```
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install and check your setup:

```
pip install -e ".[dev]"
voiceqa version
voiceqa doctor
```

`voiceqa doctor` prints your Python version and whether ffmpeg is on your PATH.

## Responsible use

Only test voice agents you own or have permission to test. See [SECURITY.md](SECURITY.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
