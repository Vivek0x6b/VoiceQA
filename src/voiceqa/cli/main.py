import asyncio
import logging
import platform
import shutil
from pathlib import Path
from typing import Annotated

import typer

from voiceqa import __version__
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent

app = typer.Typer(help="Local-first QA lab for AI voice agents.", no_args_is_help=True)


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
