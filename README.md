# VoiceQA

Open-source QA lab for AI voice agents. VoiceQA places real SIP phone calls to your agent, talks to it with local text-to-speech, transcribes the replies with local speech-to-text, and writes JSON and HTML reports covering the transcript, recordings, latency, turn-taking and audio quality.

Everything runs on your machine. Nothing is sent to the cloud. The only cost is your own VoIP/SIP provider for the phone line.

## Status

Pre-alpha. VoiceQA can place a SIP call, speak text into it, record both sides and transcribe them. Scenarios, audio checks and reports come next. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the roadmap.

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

## Speech setup

### Speech-to-text

Speech-to-text uses faster-whisper and is an optional extra, `[stt]`. Install it and download a model ahead of time. VoiceQA never downloads a model in the middle of a call.

```
pip install -e ".[dev,stt]"
voiceqa models pull stt base.en
voiceqa models list
```

```
models dir:    C:\Users\you\.cache\huggingface\hub (Hugging Face cache)
base.en        147.8 MB  C:\Users\you\.cache\huggingface\hub\models--Systran--faster-whisper-base.en
```

Models go to the Hugging Face cache unless you pass `--models-dir` or set `VOICEQA_MODELS_DIR`. Which model to pick:

| model | size | notes |
|---|---|---|
| tiny.en | 78.1 MB | fastest, makes more mistakes; used by the tests |
| base.en | 147.8 MB | the default |
| small.en | 486.1 MB | clearly better on 8 kHz phone audio, needs more CPU time |

To see how fast a model runs on your CPU, on an 18 s test clip that ships with VoiceQA:

```
voiceqa bench stt --stt-model base.en
```

```
engine            faster-whisper base.en
audio             18.1 s, 4 utterances
model load        0.72 s
real-time factor  0.133 (2.40 s for the clip)
utterance median  945 ms
utterance p95     998 ms
utterance max     1003 ms
```

The numbers are for information only. A real-time factor of 0.133 means 1 s of audio takes 0.133 s to transcribe.

### Text-to-speech

No extra is needed. VoiceQA uses the speech engine your OS already has: SAPI on Windows (`--tts sapi`) and `say` on macOS (`--tts say`). Linux has no such engine, so there you use WAV prompts (`--tts wav`): a YAML file that maps each line of text to a WAV file, with paths relative to the YAML file.

```
# prompts.yaml
"Hello, this is a test of the local voice lab.": hello.wav
"I would like to book a table for two.": booking.wav
```

Try a voice with `voiceqa say`:

```
voiceqa say "Hello, this is a test of the local voice lab." -o hello.wav
```

```
engine        sapi
audio         3.71 s at 22050 Hz
saved         hello.wav
```

`voiceqa say --list-voices` prints the voice names for `--voice`. Speech from the OS engines and from plugins is cached per user, so the same text and voice is instant the second time. Set `VOICEQA_CACHE_DIR` to move the cache.

### Outside TTS engines (plugins)

Any other engine can be plugged in as a separate program. VoiceQA never imports its code. The contract:

- An executable on your PATH named `voiceqa-tts-<name>`.
- It reads JSON from stdin: `{"text": "...", "voice": "..."}` (voice may be null).
- It writes a WAV file to the path given by its `--out` argument.
- Exit 0 means success. On failure it exits non-zero with a message on stderr, which VoiceQA shows.

Use it with `--tts plugin:<name>`. VoiceQA runs it with a timeout.

Piper and Kokoro are good neural engines, but they can only be used this way, never shipped with VoiceQA. Piper is GPL-3.0, and installing Kokoro pulls in phonemizer-fork (GPLv3+) and the espeak-ng library (through espeakng-loader). VoiceQA is Apache 2.0, so GPL code stays out of it and its extras.

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

To speak text instead of the tone and transcribe both sides once the call ends (needs the `[stt]` extra and a downloaded model):

```
voiceqa call sip:agent@127.0.0.1:5060 --say "Hello, this is a test of the local voice lab. Please repeat after me." --duration 8 --transcribe
```

```
tts           sapi, 5.93 s of speech
answered      yes
duration      7.98 s
frames sent   400
frames recv   400
first audio   0.171 s after answer
recording     runs\20261110-041154
transcript    runs\20261110-041154\transcript.json
```

The speech is made before dialing, so a TTS problem never costs a call. `--tts`, `--voice` and `--prompts` pick the engine as for `voiceqa say`. transcript.json holds each segment with its words, tagged `caller` or `agent`, with times in seconds on the same zero as events.json and the WAV files.

To transcribe a recording again, for example with another model:

```
voiceqa transcribe runs\20261110-041154 --stt-model base.en
```

```
transcript    runs\20261110-041154\transcript.json
segments      4
    0.00  caller  Hello, this is a test of the local voice lab.
    0.00  agent   Hello, this is a test of the local voice lab.
    3.50  caller  Please repeat after me.
    3.60  agent   Please repeat after me.
```

Start times from faster-whisper can be a few tenths of a second early, so the agent's first line shows 0.00 here although the echo came back 0.17 s later.

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
