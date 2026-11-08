import platform
import shutil

import typer

from voiceqa import __version__

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
