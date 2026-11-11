"""Replies for the scripted test agent (`voiceqa agent --script`).

A script is a YAML file:

    greeting: {wav: hello.wav}          # optional, played on answer
    replies:                            # one per caller turn, in order
      - {say: "Sure, for how many?", delay_ms: 600}
      - {wav: replies/name.wav}
    then: hangup                        # or silence; default hangup

Each line is either `wav` (a WAV file, relative to the YAML file) or `say`
(text for a TTS engine), with an optional `delay_ms` (0 or more): for a
reply, from the end of the caller's turn; for the greeting, from the answer.
WAV lines are read when the script is read. Say lines are synthesized by
`Script.with_speech()`, so every line has its audio before the agent
listens. All audio ends up int16 mono at 8 kHz.
"""

import dataclasses
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from voiceqa.audio.dsp import resample, to_mono
from voiceqa.audio.wav import read_wav
from voiceqa.speech.interfaces import Speech
from voiceqa.speech.tts_process import TtsError
from voiceqa.telephony.media import SAMPLE_RATE

THEN_HANGUP = "hangup"
THEN_SILENCE = "silence"

_TOP_KEYS = ("greeting", "replies", "then")
_LINE_KEYS = ("wav", "say", "delay_ms")


class ScriptError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class Line:
    """The greeting or one reply. `audio` is None for a say line until it
    is synthesized."""

    name: str
    kind: str
    text: str
    delay_ms: float
    audio: np.ndarray | None


@dataclass(frozen=True, eq=False)
class Script:
    path: Path
    greeting: Line | None
    replies: tuple[Line, ...]
    then: str

    @property
    def lines(self) -> list[Line]:
        return ([self.greeting] if self.greeting is not None else []) + list(self.replies)

    @property
    def needs_tts(self) -> bool:
        return any(line.audio is None for line in self.lines)

    def with_speech(self, synthesize: Callable[[str], Speech]) -> "Script":
        """A copy with every say line synthesized by `synthesize(text)` and
        resampled to 8 kHz. A failure raises ScriptError naming the line."""

        def ready(line: Line | None) -> Line | None:
            if line is None or line.audio is not None:
                return line
            try:
                speech = synthesize(line.text)
            except (TtsError, ValueError, OSError) as exc:
                raise self._error(line.name, f"TTS failed for {line.text!r}: {exc}") from None
            if len(speech.samples) == 0:
                raise self._error(line.name, f"TTS returned no audio for {line.text!r}")
            audio = resample(speech.samples, speech.rate, SAMPLE_RATE)
            return dataclasses.replace(line, audio=audio)

        return dataclasses.replace(
            self,
            greeting=ready(self.greeting),
            replies=tuple(ready(line) for line in self.replies),
        )

    def _error(self, where: str, problem: str) -> ScriptError:
        return ScriptError(f"{self.path}: {where}: {problem}")


def read_script(path: str | Path) -> Script:
    """Read and check a script, loading its WAV lines. Raises ScriptError
    naming the file, the line and the problem."""
    path = Path(path)
    if not path.is_file():
        raise ScriptError(f"no such script file: {path}")

    def error(where: str, problem: str) -> ScriptError:
        return ScriptError(f"{path}: {where}: {problem}")

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScriptError(f"{path}: not valid YAML: {exc}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise ScriptError(f"{path}: cannot read it: {exc}") from None
    if not isinstance(data, dict):
        raise ScriptError(f"{path}: must be a mapping with replies (and optionally greeting, then)")
    for key in data:
        if key not in _TOP_KEYS:
            raise error("script", f"unknown key {key!r}, expected one of {', '.join(_TOP_KEYS)}")

    greeting = None
    if data.get("greeting") is not None:
        greeting = _read_line(data["greeting"], "greeting", path.parent, error)
    replies = data.get("replies")
    if replies is None or replies == []:
        raise error("replies", "needs at least one reply")
    if not isinstance(replies, list):
        raise error("replies", "must be a list")
    lines = tuple(
        _read_line(item, f"reply {i}", path.parent, error) for i, item in enumerate(replies, 1)
    )
    then = data.get("then", THEN_HANGUP)
    if then not in (THEN_HANGUP, THEN_SILENCE):
        raise error("then", f"must be {THEN_HANGUP} or {THEN_SILENCE}, got {then!r}")
    return Script(path=path, greeting=greeting, replies=lines, then=then)


def _read_line(
    item: object, name: str, folder: Path, error: Callable[[str, str], ScriptError]
) -> Line:
    if not isinstance(item, dict):
        raise error(name, "must be a mapping with wav or say")
    for key in item:
        if key not in _LINE_KEYS:
            raise error(name, f"unknown key {key!r}, expected one of {', '.join(_LINE_KEYS)}")
    if "wav" in item and "say" in item:
        raise error(name, "has both wav and say, give one")
    if "wav" not in item and "say" not in item:
        raise error(name, "needs wav or say")

    delay = item.get("delay_ms", 0)
    if isinstance(delay, bool) or not isinstance(delay, int | float) or math.isnan(delay):
        raise error(name, f"delay_ms must be a number, got {delay!r}")
    if delay < 0:
        raise error(name, f"delay_ms must be 0 or more, got {delay}")

    if "say" in item:
        text = item["say"]
        if not isinstance(text, str) or not text.strip():
            raise error(name, f"say must be text, got {text!r}")
        return Line(name=name, kind="say", text=text, delay_ms=delay, audio=None)

    wav = item["wav"]
    if not isinstance(wav, str) or not wav.strip():
        raise error(name, f"wav must be a file path, got {wav!r}")
    file = folder / wav
    if not file.is_file():
        raise error(name, f"no such WAV file: {file}")
    try:
        samples, rate = read_wav(file)
    except (OSError, ValueError, EOFError) as exc:
        raise error(name, f"cannot read {wav}: {exc}") from None
    audio = resample(to_mono(samples), rate, SAMPLE_RATE)
    return Line(name=name, kind="wav", text=wav, delay_ms=delay, audio=audio)
