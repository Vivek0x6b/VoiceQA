# VoiceQA

A free, offline test framework for AI voice agents. VoiceQA places real SIP phone calls to your agent, talks to it with local text-to-speech, transcribes the replies with local speech-to-text, and checks what came back over the wire: the transcript, recordings, latency, turn-taking and audio quality.

Everything runs on your machine. Nothing is sent to the cloud. The only cost is your own VoIP/SIP provider for the phone line.

## Why VoiceQA

- No API keys. Nothing to sign up for, and VoiceQA never needs one.
- No cloud. Speech-to-text, text-to-speech and every check run locally. No audio, transcript or result leaves your machine.
- It measures the real audio. VoiceQA is a SIP caller, so it hears what a phone caller would hear, with the codec, the network and the agent's real timing.
- Same input, same result. Checks are plain rules (text matches, timings, audio measurements), so a scenario run twice against the same agent behaviour gives the same verdict.
- The only possible cost is your own phone number or SIP trunk, paid to your provider.

Checks on meaning, such as whether the agent followed its instructions or made something up, are planned as an optional add-on that runs a local model (for example through Ollama). Those results will be marked as probabilistic and kept apart from the rule-based ones.

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

To run the tests: `pytest` runs everything. `pytest -m "not realtime"` leaves out the tests that place live calls, time audio or start subprocesses, which is the set CI requires to pass; CI runs the `realtime` tests in a separate job with reruns and reports their failures as warnings.

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

If the model is not on disk yet, `voiceqa bench stt` downloads it first, printing where it is going before it starts.

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

`voiceqa say --list-voices` prints the voice names for `--voice`. On a Mac with no default voice set, `say` falls back to an installed voice such as Samantha and prints a warning; set a default voice in System Settings or pass `--voice` to choose. Speech from the OS engines and from plugins is cached per user, so the same text and voice is instant the second time. Set `VOICEQA_CACHE_DIR` to move the cache.

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

### Scripted test agent

`voiceqa agent --script replies.yaml` stands in for a real voice agent. It plays a greeting when it answers, then one reply each time the caller stops talking, in order:

```yaml
greeting: {say: "Hello, how can I help?"}
replies:
  - {say: "Sure.", delay_ms: 400}
  - {say: "Bye now."}
then: hangup
```

Each line is either `wav` (a WAV file, relative to the YAML file, any rate, for example `{wav: replies/name.wav}`) or `say` (text, spoken with `--tts`, `--voice` and `--prompts` as for `voiceqa say`). `delay_ms` (default 0) counts from the end of the caller's last word, or from the answer for the greeting. The caller's turn ends after 700 ms of silence (`--end-silence-ms`), so a delay shorter than that is the same as no delay: in the run below the caller stopped at 3.0 s and reply 1 went out at 3.70 s. After the last reply, `then: hangup` (the default) ends the call and `then: silence` keeps it open. Every call starts from the top.

```
voiceqa agent --script replies.yaml --tts fake
```

```
tts           fake, 3 say line(s) ready
agent listening on 127.0.0.1:5060 (script replies.yaml), Ctrl+C to stop
2026-11-11 17:16:22,010 answered call fa1903962645b281cb3d774c from 127.0.0.1:52334 (PCMU)
2026-11-11 17:16:22,042 call fa1903962645b281cb3d774c: greeting (say) at 0.04 s
2026-11-11 17:16:25,696 call fa1903962645b281cb3d774c: reply 1 (say) at 3.70 s
2026-11-11 17:16:28,707 call fa1903962645b281cb3d774c: reply 2 (say) at 6.72 s
2026-11-11 17:16:29,134 call fa1903962645b281cb3d774c ended
```

Every say line is made before the agent starts listening, so a bad script or a TTS failure stops it at once (exit 1) with the line it was about. The turn detection uses the energy VAD by default, so it needs no extra; `--vad silero` needs the `[stt]` extra.

## Run a scenario

A scenario is a YAML file of steps: what the caller says, what the agent should answer, and where to go next. `voiceqa run` places the call, says each line, transcribes each agent turn with faster-whisper (needs the `[stt]` extra and a downloaded model) and checks it.

```yaml
name: table booking
variables: {people: four}
defaults: {timeout: 15s, end_silence_ms: 1200}
steps:
  - label: greet
    expect: {any_of: ["how can I help", "how may I help"]}
  - label: ask
    say: "I would like to book a table for tonight."
    expect: {contains: how many, max_latency_ms: 3000}
  - label: people
    say: "We are {people} people, at seven."
    expect: {fuzzy: {text: "you are booked for {people} people", min: 0.7}}
    complete: true
assert: [call_connected, agent_responded, task_completed]
```

Against the scripted agent from the section above, with three SAPI say lines that match:

```
voiceqa run booking.yaml --target sip:agent@127.0.0.1:5070 --junit junit.xml
```

```
tts           sapi, 2 line(s) ready
scenario  table booking (booking.yaml)
PASS    1  greet             heard "Hello, thanks for calling. How can I help you today?"
PASS    2  ask               heard "Sure. For how many people? //"; latency 1000 ms
PASS    3  people            heard "Great, you are booked for four people at 7 tonight."; latency 1000 ms
assert    call_connected    pass  the call was answered
assert    agent_responded   pass  3 agent turn(s) heard
assert    task_completed    pass  a complete step passed
result    PASSED, 3 of 3 steps passed
run       runs\20261112-042434Z-table-booking
```

The run folder holds the usual recording files, transcript.json (agent turns from the live transcripts, caller lines with the text that was said) and result.json (every step with what was said and heard, matcher results, latency, attempts and the path taken, plus the call assertions). `--junit` writes one testcase per step and per assertion, for CI.

How a run goes:

- A step with only `expect` listens first, for a greeting. A step with `say` and `expect` says the line, then waits up to `timeout` for the agent's next turn. A step with only `say` does not wait.
- `say` is text (spoken with `--tts`, `--voice`, `--prompts` as for `voiceqa say`) or `{wav: file.wav, text: "what it says"}`. Without `text:` the transcript and results show the file name in brackets.
- A failed expect is said and listened to again `retries` times. Then `on_fail: goto <label>` goes elsewhere (the step shows as ROUTED and does not fail the run), and without `on_fail` the run stops as failed. `on_match: goto <label>` jumps, `goto end` stops. More than `max_steps` steps (default 50) stops a goto loop as a failure.
- Latency is from the end of the last voiced frame of the line we said to the start of the agent's speech, so silence at the end of a WAV or TTS clip does not count.
- `regex` is case-sensitive unless the pattern starts with `(?i)`. The other text matchers ignore case and punctuation.
- `fuzzy` compares the expected text with every window of the same number of words in the agent's turn, so a different word split from STT ("to night" for "tonight") lowers the score. Leave some room in `min`.
- An agent turn ends after `end_silence_ms` of silence (default 700). Agents that pause between sentences get split into two turns: in the run above, SAPI's pause after "Sure." split the reply until `end_silence_ms` was raised to 1200. It can be set per step too.

`voiceqa run booking.yaml --check` only checks the file and lists every problem. `--var name=value` sets a variable. Before dialing, the extra and the model are checked, the model is loaded and every line is synthesized, so a setup problem never costs a call. Exit codes: 0 everything passed, 1 a step or assertion failed (an unanswered call included), 2 bad input or an invalid scenario, 3 could not run (TTS, model, VAD or network failure).

### Run history

Every `voiceqa run` that places a call adds a row to a local SQLite file, runs.sqlite3, in the VoiceQA data folder: `%LOCALAPPDATA%\voiceqa` on Windows, `~/Library/Application Support/voiceqa` on macOS, and `$XDG_DATA_HOME/voiceqa` (or `~/.local/share/voiceqa`) elsewhere. Set `VOICEQA_DATA_DIR` to keep it somewhere else. Failed runs and runs that could not finish get a row too. If the file cannot be written, the run prints a warning and keeps its exit code.

`voiceqa runs list` shows the newest runs first. `--limit N` (default 20), `--scenario NAME` and `--status passed|failed|error` narrow it down. After three runs against the scripted agent, two of the booking scenario above and one that expects "cancelled":

```
id  started               status  exit  steps  duration  scenario
 3  2026-11-13T02:27:05Z  failed     1    1/2    14.4 s  cancel booking
 2  2026-11-13T02:26:25Z  passed     0    3/3    22.7 s  table booking
 1  2026-11-13T02:25:59Z  passed     0    3/3    22.6 s  table booking
```

`voiceqa runs show <id>` prints the same summary the run printed. It comes from the stored result.json, so it still works after the run folder is deleted (this is `runs show 3` with the runs folder gone):

```
scenario  cancel booking (cancel.yaml)
PASS    1  greet             heard "Hello, thanks for calling. How can I help you today?"
FAIL    2  cancel            contains: did not hear "cancelled"; heard "tour. For how many people?"; latency 1380 ms
assert    call_connected    pass  the call was answered
stopped   step 2 (cancel) failed
result    FAILED, 1 of 2 steps passed
run       runs\20261113-022703Z-cancel-booking
```

An unknown id exits 2. If the history cannot be read (for example it was written by a newer VoiceQA), both commands exit 1 and say why.

## Responsible use

Only test voice agents you own or have permission to test. See [SECURITY.md](SECURITY.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
