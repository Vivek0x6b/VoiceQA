import asyncio
import logging
import os
import platform
import re
import shutil
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from voiceqa import __version__
from voiceqa.audio.dsp import generate_tone
from voiceqa.engine.call import CallResult, run_call
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent, load_prompt
from voiceqa.telephony.media import PayloadType

app = typer.Typer(help="Local-first QA lab for AI voice agents.", no_args_is_help=True)

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


@app.command()
def doctor() -> None:
    """Check the local environment."""
    typer.echo(f"python: {platform.python_version()}")
    ffmpeg = shutil.which("ffmpeg")
    typer.echo(f"ffmpeg: found ({ffmpeg})" if ffmpeg else "ffmpeg: not found")


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
) -> None:
    """Run a local test agent that answers calls and echoes or plays a WAV."""
    config = AgentConfig(host=host, port=port, play=play, hangup_after=hangup_after)
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
    return (host, int(port)) if sep else (value, default_port)


def _target_host_port(target: str) -> tuple[str, int] | None:
    """Host and port from a full SIP URI target, or None for a bare user part."""
    match = _TARGET_RE.match(target)
    if match is None:
        return None
    host, port = match.group(1), match.group(2)
    return host, int(port) if port else DEFAULT_SIP_PORT


def _print_summary(result: CallResult) -> None:
    def line(label: str, value: str) -> None:
        typer.echo(f"{label:<14}{value}")

    if result.answered:
        line("answered", "yes")
        line("duration", f"{result.duration:.2f} s")
        line("frames sent", str(result.frames_sent))
        line("frames recv", str(result.frames_received))
        if result.first_audio_latency is None:
            line("first audio", "none")
        else:
            line("first audio", f"{result.first_audio_latency:.3f} s after answer")
    else:
        detail = f"{result.status} {result.reason}" if result.status else result.reason
        line("answered", f"no ({detail})")
    line("recording", str(result.record_dir))


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
        float, typer.Option(help="Hang up this many seconds after the answer.")
    ] = 10.0,
    codec: Annotated[Codec, typer.Option(help="Payload type to offer.")] = Codec.pcmu,
) -> None:
    """Place one call, send audio, and record both sides.

    The SIP password, if the server needs one, is read from the
    VOICEQA_SIP_PASSWORD environment variable, never from a command line
    option, so it never ends up in shell history.
    """
    if server is not None:
        server_host, server_port = _split_host_port(server, DEFAULT_SIP_PORT)
    else:
        parsed = _target_host_port(target)
        if parsed is None:
            typer.echo("error: --server is required when TARGET is not a full SIP URI", err=True)
            raise typer.Exit(2)
        server_host, server_port = parsed

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
        raise typer.Exit(1) from None

    _print_summary(result)
    raise typer.Exit(0 if result.answered else 1)
