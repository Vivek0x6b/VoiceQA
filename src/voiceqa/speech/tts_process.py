"""Running an external TTS program (an OS engine or a plugin) and reading its WAV.

Every external engine runs through run_tts_process: stdin bytes in, stdout
and stderr captured, a timeout, and on timeout the whole process tree is
killed. Killing just the direct child is not enough on Windows, where a
plugin is often a .cmd wrapper around the real program: the grandchild
would keep the pipes open and the wait would hang until it finished.
"""

import os
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from voiceqa.audio.dsp import to_mono
from voiceqa.audio.wav import read_wav

from .interfaces import Speech

DEFAULT_TIMEOUT = 60.0
_STDERR_LIMIT = 2000


class TtsError(RuntimeError):
    """A TTS engine could not produce speech: missing program, failure or timeout."""


def _stderr_text(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace").strip()
    if len(text) > _STDERR_LIMIT:
        text = "..." + text[-_STDERR_LIMIT:]
    return text


def _kill_tree(proc: subprocess.Popen) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_tts_process(
    args: Sequence[str],
    *,
    label: str,
    stdin: bytes = b"",
    timeout: float = DEFAULT_TIMEOUT,
    env: Mapping[str, str] | None = None,
) -> bytes:
    """Run args, feed it stdin, and return its stdout.

    A non-zero exit or a timeout raises TtsError naming `label` and showing
    the program's stderr.
    """
    if sys.platform == "win32":
        group = {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        group = {"start_new_session": True}
    try:
        proc = subprocess.Popen(
            list(args),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=dict(env) if env is not None else None,
            **group,
        )
    except OSError as exc:
        raise TtsError(f"{label}: cannot start {args[0]}: {exc}") from None
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            _, err = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            err = b""
        detail = _stderr_text(err)
        raise TtsError(
            f"{label} timed out after {timeout:g} s" + (f": {detail}" if detail else "")
        ) from None
    except BaseException:
        _kill_tree(proc)
        raise
    if proc.returncode != 0:
        detail = _stderr_text(err) or "no message on stderr"
        raise TtsError(f"{label} failed (exit code {proc.returncode}): {detail}")
    return out


def read_speech_wav(path: Path, label: str) -> Speech:
    if not path.is_file() or path.stat().st_size == 0:
        raise TtsError(f"{label} exited without writing {path}")
    try:
        samples, rate = read_wav(path)
    except (ValueError, EOFError) as exc:
        raise TtsError(f"{label} wrote an unreadable WAV: {exc}") from None
    return Speech(samples=to_mono(samples), rate=rate)
