import asyncio
import logging
import os
import platform
import re
import shutil
import wave
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml

from voiceqa import __version__
from voiceqa.audio.dsp import generate_tone, resample
from voiceqa.audio.wav import write_wav
from voiceqa.engine.calibrate import CalibrationResult, run_calibration
from voiceqa.engine.call import CallResult, run_call
from voiceqa.engine.transcribe import TRACKS, transcribe_run, write_transcript
from voiceqa.speech import model_cache
from voiceqa.speech.bench import default_clip_dir, load_bench_clip, run_stt_bench
from voiceqa.speech.interfaces import Speech, TextToSpeech
from voiceqa.speech.model_cache import (
    KNOWN_SIZES_MB,
    STT_EXTRA_HINT,
    ModelNotAvailableError,
    SttExtraMissingError,
    cached_models,
    find_model,
    models_dir_source,
    pull_stt_model,
    repo_id,
    resolve_models_dir,
)
from voiceqa.speech.models import Transcript
from voiceqa.speech.registry import create_tts, default_tts_name
from voiceqa.speech.tts_cache import CachedTts
from voiceqa.speech.tts_process import TtsError
from voiceqa.speech.worker import SttWorker, SttWorkerError
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent, load_prompt
from voiceqa.telephony.media import SAMPLE_RATE, PayloadType

app = typer.Typer(help="Local-first QA lab for AI voice agents.", no_args_is_help=True)
models_app = typer.Typer(help="List and download speech models.", no_args_is_help=True)
bench_app = typer.Typer(help="Measure speech engine speed on this machine.", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(bench_app, name="bench")

ModelsDirOption = Annotated[
    Path | None,
    typer.Option(
        "--models-dir",
        help="Model folder (Hugging Face cache layout). Default: VOICEQA_MODELS_DIR, "
        "else the Hugging Face cache.",
    ),
]

DEFAULT_SIP_PORT = 5060
DEFAULT_TONE_HZ = 440
DEFAULT_TONE_SECONDS = 1.0

# A bare target URI: optional scheme and user part, then a host and an
# optional port, for example "sip:agent@127.0.0.1:5070" or "agent".
_TARGET_RE = re.compile(r"^sips?:(?:[^@]*@)?([^:;>\s]+)(?::(\d+))?", re.IGNORECASE)


class Codec(StrEnum):
    pcmu = "pcmu"
    pcma = "pcma"


_PAYLOAD_TYPES = {Codec.pcmu: PayloadType.PCMU, Codec.pcma: PayloadType.PCMA}


@app.command()
def version() -> None:
    """Print the VoiceQA version."""
    typer.echo(f"voiceqa {__version__}")


def _faster_whisper_version() -> str:
    from importlib.metadata import version as dist_version

    return dist_version("faster-whisper")


def _size(size_bytes: int) -> str:
    return f"{size_bytes / 1_000_000:.1f} MB"


def _echo_models_dir(models_dir: Path | None) -> None:
    _doctor_line(
        "models dir:", f"{resolve_models_dir(models_dir)} ({models_dir_source(models_dir)})"
    )


def _echo_cached_models(models_dir: Path | None) -> None:
    found = cached_models(models_dir)
    if not found:
        _doctor_line("stt models:", "none cached")
    for m in found:
        _doctor_line("stt model:", f"{m.name:<10} {_size(m.size_bytes):>10}  {m.path}")


@app.command()
def doctor(models_dir: ModelsDirOption = None) -> None:
    """Check the local environment."""
    typer.echo(f"python: {platform.python_version()}")
    ffmpeg = shutil.which("ffmpeg")
    typer.echo(f"ffmpeg: found ({ffmpeg})" if ffmpeg else "ffmpeg: not found")
    if model_cache.stt_extra_installed():
        _doctor_line("stt extra:", f"installed (faster-whisper {_faster_whisper_version()})")
    else:
        _doctor_line("stt extra:", f"not installed ({model_cache.STT_EXTRA_HINT})")
    _echo_models_dir(models_dir)
    _echo_cached_models(models_dir)


class ModelKind(StrEnum):
    stt = "stt"


@models_app.command("list")
def models_list(models_dir: ModelsDirOption = None) -> None:
    """Show cached faster-whisper models with their size and path."""
    _echo_models_dir(models_dir)
    found = cached_models(models_dir)
    if not found:
        typer.echo("no faster-whisper models cached")
    for m in found:
        typer.echo(f"{m.name:<12} {_size(m.size_bytes):>10}  {m.path}")


def _check_model_name(model: str) -> None:
    try:
        repo_id(model)
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None


def _download_message(model: str, models_dir: Path | None) -> str:
    size = f" ({KNOWN_SIZES_MB[model]} MB)" if model in KNOWN_SIZES_MB else ""
    return (
        f"downloading faster-whisper model {model}{size} from Hugging Face "
        f"into {resolve_models_dir(models_dir)}"
    )


def _pull(model: str, models_dir: Path | None) -> Path:
    try:
        return pull_stt_model(model, resolve_models_dir(models_dir))
    except (SttExtraMissingError, ModelNotAvailableError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None


@models_app.command("pull")
def models_pull(
    kind: Annotated[ModelKind, typer.Argument(help="Model kind. Only stt for now.")],
    model: Annotated[str, typer.Argument(help="Model name, for example base.en.")],
    models_dir: ModelsDirOption = None,
) -> None:
    """Download a model now, so nothing is downloaded later during a run."""
    _check_model_name(model)
    existing = find_model(model, resolve_models_dir(models_dir))
    if existing is not None:
        typer.echo(f"{model} already cached at {existing}")
        return
    if not model_cache.stt_extra_installed():
        typer.echo(
            f"error: downloading STT models needs the [stt] extra: {model_cache.STT_EXTRA_HINT}",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(_download_message(model, models_dir))
    path = _pull(model, models_dir)
    typer.echo(f"{model} saved at {path}")


def _bench_engine(model: str, models_dir: Path):
    from voiceqa.speech.whisper import FasterWhisperStt

    return FasterWhisperStt(model=model, models_dir=models_dir)


@bench_app.command("stt")
def bench_stt(
    stt_model: Annotated[str, typer.Option("--stt-model", help="faster-whisper model.")] = (
        "base.en"
    ),
    models_dir: ModelsDirOption = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")] = False,
) -> None:
    """Time STT on this CPU with the LibriVox test clip (informational only).

    Reports model load time, real-time factor for the whole clip (processing
    time / audio length) and per-utterance latency. One warm-up run is not
    counted.
    """
    _check_model_name(stt_model)
    if not model_cache.stt_extra_installed():
        typer.echo(
            f"error: bench stt needs the [stt] extra: {model_cache.STT_EXTRA_HINT}", err=True
        )
        raise typer.Exit(1)
    directory = resolve_models_dir(models_dir)
    if find_model(stt_model, directory) is None:
        typer.echo(_download_message(stt_model, models_dir), err=True)
        _pull(stt_model, models_dir)

    clip = load_bench_clip(default_clip_dir())
    result = run_stt_bench(
        lambda: _bench_engine(stt_model, directory),
        clip,
        engine_name="faster-whisper",
        model=stt_model,
    )
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
        return
    _bench_line("engine", f"{result.engine} {result.model}")
    _bench_line("audio", f"{result.audio_seconds:.1f} s, {result.utterances} utterances")
    _bench_line("model load", f"{result.load_seconds:.2f} s")
    _bench_line("real-time factor", f"{result.rtf:.3f} ({result.clip_seconds:.2f} s for the clip)")
    _bench_line("utterance median", f"{result.utterance_median_ms:.0f} ms")
    _bench_line("utterance p95", f"{result.utterance_p95_ms:.0f} ms")
    _bench_line("utterance max", f"{result.utterance_max_ms:.0f} ms")


def _stt_worker(model: str, models_dir: Path | None) -> SttWorker:
    """A (not yet started) faster-whisper worker, after checking up front
    that the extra and the model are there, so the error names the fix."""
    if not model_cache.stt_extra_installed():
        raise SttExtraMissingError(f"transcription needs the [stt] extra: {STT_EXTRA_HINT}")
    directory = resolve_models_dir(models_dir)
    if find_model(model, directory) is None:
        raise ModelNotAvailableError(
            f"faster-whisper model {model} is not in {directory}. "
            f"Download it first: voiceqa models pull stt {model}"
        )
    return SttWorker("faster-whisper", model=model, models_dir=str(directory))


def _run_transcription(run_dir: Path, worker: SttWorker) -> tuple[Path, Transcript]:
    with worker:
        transcript = asyncio.run(transcribe_run(run_dir, worker))
    return write_transcript(run_dir, transcript), transcript


def _make_transcript(run_dir: Path, model: str, models_dir: Path | None) -> tuple[Path, Transcript]:
    return _run_transcription(run_dir, _stt_worker(model, models_dir))


_TRANSCRIPT_ERRORS = (SttExtraMissingError, ModelNotAvailableError, SttWorkerError)

DEFAULT_STT_MODEL = "base.en"

SttModelOption = Annotated[
    str, typer.Option("--stt-model", help="faster-whisper model for the transcript.")
]
# None by default: --stt-model only applies together with --transcribe, and the
# effective default (DEFAULT_STT_MODEL) is filled in only when that is set.
CallSttModelOption = Annotated[
    str | None,
    typer.Option(
        "--stt-model",
        help="faster-whisper model for the transcript. Only used with --transcribe.",
    ),
]


@app.command()
def transcribe(
    run_dir: Annotated[
        Path, typer.Argument(help="Run folder with caller.wav and agent.wav, from voiceqa call.")
    ],
    stt_model: SttModelOption = "base.en",
    models_dir: ModelsDirOption = None,
) -> None:
    """Transcribe a recorded call into transcript.json in its run folder.

    Each side is transcribed on its own and every segment is tagged caller
    or agent. Times are seconds on the same zero as events.json and the
    WAVs. The model must already be downloaded (voiceqa models pull stt).
    """
    _check_model_name(stt_model)
    for _, name in TRACKS:
        if not (run_dir / name).is_file():
            typer.echo(f"error: no {name} in {run_dir}", err=True)
            raise typer.Exit(2)
    try:
        path, transcript = _make_transcript(run_dir, stt_model, models_dir)
    except _TRANSCRIPT_ERRORS as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    except (wave.Error, ValueError, EOFError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        typer.echo("transcription interrupted", err=True)
        raise typer.Exit(1) from None
    _line("transcript", str(path))
    _line("segments", str(len(transcript.segments)))
    for s in transcript.segments:
        typer.echo(f"{s.start:8.2f}  {s.speaker:<6}  {s.text}")


TtsOption = Annotated[
    str | None,
    typer.Option(
        "--tts",
        help="TTS engine: sapi, say, wav, fake or plugin:<name>. "
        "Default: sapi on Windows, say on macOS, wav elsewhere.",
    ),
]
VoiceOption = Annotated[str | None, typer.Option("--voice", help="Voice name for the engine.")]
PromptsOption = Annotated[
    Path | None,
    typer.Option("--prompts", help="YAML index of text to WAV files, for --tts wav."),
]

# Engines whose output is not worth caching: WAV prompts already are files,
# and the fake engine is instant.
_UNCACHED_TTS = ("wav", "fake")


def _bad_option(message: str) -> typer.Exit:
    typer.echo(f"error: {message}", err=True)
    return typer.Exit(2)


def _tts_engine(tts: str | None, prompts: Path | None, wav_hint: str) -> TextToSpeech:
    """Build the TTS engine the options ask for. Problems with the options
    themselves (unknown engine, an OS engine on the wrong OS, wav without an
    index) exit 2."""
    name = tts or default_tts_name()
    if name == "wav":
        if prompts is None:
            if tts is None:
                raise _bad_option(
                    "the default TTS engine here is wav prompts (no OS speech engine on "
                    f"this OS). {wav_hint}"
                )
            raise _bad_option(f"--tts wav needs an index of prompts. {wav_hint}")
        try:
            return create_tts("wav", index=prompts)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise _bad_option(f"cannot read prompts {prompts}: {exc}") from None
    if prompts is not None:
        raise _bad_option("--prompts is only used with --tts wav")
    try:
        engine = create_tts(name)
    except (ValueError, TtsError) as exc:
        raise _bad_option(str(exc)) from None
    return engine if name in _UNCACHED_TTS else CachedTts(engine)


def _synthesize(engine: TextToSpeech, text: str, voice: str | None) -> Speech:
    try:
        speech = engine.synthesize(text, voice)
    except (TtsError, ValueError, OSError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    if len(speech.samples) == 0:
        typer.echo(f"error: TTS engine {engine.name} returned no audio", err=True)
        raise typer.Exit(1)
    return speech


def _check_text(text: str) -> None:
    if not text.strip():
        raise _bad_option("the text to speak is empty")


@app.command()
def say(
    text: Annotated[str | None, typer.Argument(help="Text to speak.")] = None,
    out: Annotated[Path | None, typer.Option("-o", "--out", help="WAV file to write.")] = None,
    tts: TtsOption = None,
    voice: VoiceOption = None,
    prompts: PromptsOption = None,
    list_voices: Annotated[
        bool, typer.Option("--list-voices", help="Print the engine's voice names and exit.")
    ] = False,
) -> None:
    """Speak TEXT with a TTS engine into a WAV file, to try a voice.

    The WAV keeps the engine's own sample rate. Results from OS engines and
    plugins are cached per user (VOICEQA_CACHE_DIR to move the cache), so
    asking again for the same text and voice is instant.
    """
    hint = "Pass --prompts INDEX.yaml with a WAV for this text."
    if list_voices:
        engine = _tts_engine(tts, prompts, hint)
        try:
            names = engine.voices()
        except TtsError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
        for name in names:
            typer.echo(name)
        return
    if text is None or out is None:
        raise _bad_option("give the TEXT to speak and -o OUT.wav (or --list-voices)")
    _check_text(text)
    engine = _tts_engine(tts, prompts, hint)
    speech = _synthesize(engine, text, voice)
    try:
        write_wav(out, speech.samples, speech.rate)
    except OSError as exc:
        typer.echo(f"error: cannot write {out}: {exc}", err=True)
        raise typer.Exit(1) from None
    _line("engine", engine.name)
    _line("audio", f"{len(speech.samples) / speech.rate:.2f} s at {speech.rate} Hz")
    _line("saved", str(out))


@app.command()
def agent(
    host: Annotated[str, typer.Option(help="Address to listen on for SIP and RTP.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="SIP port (UDP).")] = 5060,
    play: Annotated[
        Path | None, typer.Option(help="WAV file to play to each caller. Default: echo.")
    ] = None,
    hangup_after: Annotated[
        float | None, typer.Option(help="Hang up this many seconds after answering.")
    ] = None,
    answer_delay: Annotated[
        float, typer.Option(min=0.0, help="Ring this many seconds before answering.")
    ] = 0.0,
) -> None:
    """Run a local test agent that answers calls and echoes or plays a WAV."""
    config = AgentConfig(
        host=host, port=port, play=play, hangup_after=hangup_after, answer_delay=answer_delay
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        asyncio.run(_serve_agent(config))
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        pass


async def _serve_agent(config: AgentConfig) -> None:
    local = await LocalAgent.start(config)
    host, port = local.local_addr
    mode = f"playing {config.play}" if config.play else "echo"
    typer.echo(f"agent listening on {host}:{port} ({mode}), Ctrl+C to stop")
    try:
        await asyncio.Event().wait()
    finally:
        await local.close()


def _split_host_port(value: str, default_port: int) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep:
        return value, default_port
    try:
        return host, int(port)
    except ValueError:
        raise ValueError(f"bad --server value {value!r}: port {port!r} is not a number") from None


def _target_host_port(target: str) -> tuple[str, int] | None:
    """Host and port from a full SIP URI target, or None for a bare user part."""
    match = _TARGET_RE.match(target)
    if match is None:
        return None
    host, port = match.group(1), match.group(2)
    return host, int(port) if port else DEFAULT_SIP_PORT


def _positive(value: float) -> float:
    if value <= 0:
        raise typer.BadParameter("must be greater than 0")
    return value


def _line(label: str, value: str, width: int = 14) -> None:
    typer.echo(f"{label:<{width}}{value}")


def _doctor_line(label: str, value: str) -> None:
    _line(label, value, width=15)


def _bench_line(label: str, value: str) -> None:
    _line(label, value, width=18)


def _not_answered_detail(status: int | None, reason: str | None) -> str:
    """ "status reason", or just reason if there is no status (for example a
    resolve failure). Shared by voiceqa call and voiceqa calibrate, which
    both get these from a CallResult."""
    return f"{status} {reason}" if status else (reason or "")


def _print_summary(result: CallResult) -> None:
    if result.answered:
        _line("answered", "yes")
        _line("duration", f"{result.duration:.2f} s")
        _line("frames sent", str(result.frames_sent))
        _line("frames recv", str(result.frames_received))
        if result.first_audio_latency is None:
            _line("first audio", "none")
        else:
            _line("first audio", f"{result.first_audio_latency:.3f} s after answer")
    else:
        _line("answered", f"no ({_not_answered_detail(result.status, result.reason)})")
    _line("recording", str(result.record_dir))


@app.command()
def call(
    target: Annotated[str, typer.Argument(help="Call target: a SIP URI or a bare user part.")],
    server: Annotated[
        str | None,
        typer.Option(
            help="SIP server HOST[:PORT], used as outbound proxy. Required if TARGET has no host."
        ),
    ] = None,
    user: Annotated[str, typer.Option(help="SIP username.")] = "voiceqa",
    play: Annotated[
        Path | None,
        typer.Option(help="WAV file to send after the call is answered. Default: a test tone."),
    ] = None,
    record_dir: Annotated[
        Path | None,
        typer.Option("--record-dir", help="Where to write the recording. Default: runs/<time>."),
    ] = None,
    duration: Annotated[
        float,
        typer.Option(help="Hang up this many seconds after the answer.", callback=_positive),
    ] = 10.0,
    codec: Annotated[Codec, typer.Option(help="Payload type to offer.")] = Codec.pcmu,
    # Help text goes through Rich markup, which would drop a bare "[stt]".
    transcribe: Annotated[
        bool,
        typer.Option(
            "--transcribe",
            help="Check the \\[stt] extra and model, then write transcript.json "
            "once the call ends.",
        ),
    ] = False,
    stt_model: CallSttModelOption = None,
    models_dir: ModelsDirOption = None,
    say_text: Annotated[
        str | None,
        typer.Option("--say", help="Speak this text once answered, instead of the tone or WAV."),
    ] = None,
    tts: TtsOption = None,
    voice: VoiceOption = None,
    prompts: PromptsOption = None,
) -> None:
    """Place one call, send audio, and record both sides.

    The SIP password, if the server needs one, is read from the
    VOICEQA_SIP_PASSWORD environment variable, never from a command line
    option, so it never ends up in shell history.

    With --say, the text is synthesized before dialing, so a TTS problem
    never costs a call: it exits 2 for bad TTS options and 1 when the
    engine fails, without calling.

    With --transcribe, the \\[stt] extra and the model are checked before
    dialing: if either is missing, nothing is dialed and the error names
    the fix (exit 1). --stt-model and --models-dir only apply together
    with --transcribe. Once an answered call ends, the recording is
    transcribed; if that fails, the error goes to stderr and the exit
    code is still the call's.
    """
    if say_text is not None and play is not None:
        raise _bad_option("use either --say or --play, not both")
    if say_text is None and (tts is not None or voice is not None or prompts is not None):
        raise _bad_option("--tts, --voice and --prompts only apply together with --say")
    if say_text is not None:
        _check_text(say_text)
    if not transcribe and (stt_model is not None or models_dir is not None):
        raise _bad_option("--stt-model and --models-dir only apply together with --transcribe")
    stt_worker = None
    if transcribe:
        model = stt_model or DEFAULT_STT_MODEL
        _check_model_name(model)
        try:
            stt_worker = _stt_worker(model, models_dir)
        except _TRANSCRIPT_ERRORS as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(1) from None
    if "@" in target and not target.startswith(("sip:", "sips:")):
        typer.echo(
            "error: TARGET must not contain '@' unless it is a full sip: URI "
            "(use TARGET without '@' together with --server)",
            err=True,
        )
        raise typer.Exit(2)

    if server is not None:
        try:
            server_host, server_port = _split_host_port(server, DEFAULT_SIP_PORT)
        except ValueError as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None
    else:
        parsed = _target_host_port(target)
        if parsed is None:
            typer.echo("error: --server is required when TARGET is not a full SIP URI", err=True)
            raise typer.Exit(2)
        server_host, server_port = parsed

    if say_text is not None:
        hint = "Pass --prompts INDEX.yaml with a WAV for this text, or --play a WAV file."
        engine = _tts_engine(tts, prompts, hint)
        speech = _synthesize(engine, say_text, voice)
        audio = resample(speech.samples, speech.rate, SAMPLE_RATE)
        _line("tts", f"{engine.name}, {len(speech.samples) / speech.rate:.2f} s of speech")
    else:
        try:
            audio = (
                load_prompt(play)
                if play is not None
                else generate_tone(DEFAULT_TONE_HZ, DEFAULT_TONE_SECONDS)
            )
        except (FileNotFoundError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(2) from None

    directory = record_dir or Path("runs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    password = os.environ.get("VOICEQA_SIP_PASSWORD", "")

    try:
        result = asyncio.run(
            run_call(
                target=target,
                server_host=server_host,
                server_port=server_port,
                username=user,
                password=password,
                audio=audio,
                record_dir=directory,
                duration=duration,
                codec=_PAYLOAD_TYPES[codec],
            )
        )
    except KeyboardInterrupt:
        typer.echo("call interrupted", err=True)
        _line("recording", str(directory))
        raise typer.Exit(1) from None

    _print_summary(result)
    if transcribe:
        _transcribe_after_call(result, stt_worker)
    raise typer.Exit(0 if result.answered else 1)


def _transcribe_after_call(result: CallResult, worker: SttWorker) -> None:
    if not result.answered:
        _line("transcript", "skipped, the call was not answered")
        return
    try:
        path, _ = _run_transcription(result.record_dir, worker)
    except _TRANSCRIPT_ERRORS as exc:
        typer.echo(f"error: no transcript: {exc}", err=True)
        _line("transcript", "none")
        return
    except KeyboardInterrupt:
        typer.echo("transcription interrupted", err=True)
        raise typer.Exit(1) from None
    _line("transcript", str(path))


def _print_calibration_summary(result: CalibrationResult) -> None:
    if not result.answered:
        _line("call", f"no ({_not_answered_detail(result.status, result.reason)})")
    _line("median", f"{result.median_ms:.1f} ms" if result.median_ms is not None else "none")
    _line("p95", f"{result.p95_ms:.1f} ms" if result.p95_ms is not None else "none")
    _line("matched", f"{result.pulses_matched}/{result.pulses_sent}")
    _line("recording", str(result.record_dir))


@app.command()
def calibrate(
    seconds: Annotated[
        float, typer.Option(help="How long to send test pulses for.", callback=_positive)
    ] = 10.0,
    record_dir: Annotated[
        Path | None,
        typer.Option(
            "--record-dir",
            help="Where to write the recording. Default: runs/<time>-calibrate.",
        ),
    ] = None,
) -> None:
    """Measure VoiceQA's own round-trip latency against a local echo agent.

    Starts a local echo agent in-process, sends it a train of short tone
    pulses instead of a real conversation, and times how long each pulse
    takes to come back. This is VoiceQA's own floor, not how fast a real
    voice agent thinks: subtract it from latencies measured against one.
    """
    directory = record_dir or Path("runs") / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-calibrate"

    try:
        result = asyncio.run(run_calibration(directory, seconds=seconds))
    except KeyboardInterrupt:
        typer.echo("calibration interrupted", err=True)
        _line("recording", str(directory))
        raise typer.Exit(1) from None

    _print_calibration_summary(result)
    enough_matched = result.pulses_sent > 0 and result.pulses_matched * 2 >= result.pulses_sent
    raise typer.Exit(0 if enough_matched else 1)
