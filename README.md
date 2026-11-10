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

`voiceqa doctor` prints your Python version, whether ffmpeg is on your PATH, whether speech-to-text is installed, and which STT models are on disk.

Speech-to-text (faster-whisper) is an optional extra. Install it and download a model ahead of time:

```
pip install -e ".[dev,stt]"
voiceqa models pull stt base.en
voiceqa bench stt
```

Models go to the Hugging Face cache unless you pass `--models-dir` or set `VOICEQA_MODELS_DIR`. `base.en` (148 MB) is the default. `small.en` (486 MB) is clearly better on 8 kHz phone audio but needs more CPU time. `voiceqa bench stt` shows how fast a model runs on your CPU.

## Quick start

Start the local test agent in one terminal. It answers calls and echoes back whatever it hears.

```
voiceqa agent
```

```
agent listening on 127.0.0.1:5060 (echo), Ctrl+C to stop
```

In another terminal, place a call to it. With no `--play` option, VoiceQA sends a 1 second test tone once the call is answered.

```
voiceqa call sip:agent@127.0.0.1:5060
```

```
answered      yes
duration      9.98 s
frames sent   500
frames recv   500
first audio   0.027 s after answer
recording     runs\20261108-185309
```

The recording folder holds:

- caller.wav: what VoiceQA sent, as it went on the wire.
- agent.wav: what the agent sent back.
- mix.wav: both sides as one stereo file, caller left and agent right.
- events.json: call events (INVITE sent, ringing, answered, first RTP in and out, BYE) with timestamps.

To measure VoiceQA's own round-trip floor, with no real agent needed:

```
voiceqa calibrate
```

```
median        40.6 ms
p95           40.6 ms
matched       20/20
recording     runs\20261108-185319-calibrate
```

This runs against a local echo agent started in-process and writes the same recording files plus calibration.json (median, p95, min, max, and how many pulses were sent and matched). It is VoiceQA's own sending, local UDP and receiving time, not anything a real agent does: subtract it from latencies measured against one.

## Responsible use

Only test voice agents you own or have permission to test. See [SECURITY.md](SECURITY.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
