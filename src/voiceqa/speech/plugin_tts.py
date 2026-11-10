"""Runs a separately installed TTS plugin as a subprocess.

Implements the contract in voiceqa.speech.registry: the executable
`voiceqa-tts-<name>` on PATH gets `{"text": ..., "voice": ...}` as JSON on
stdin and writes a WAV to the path after `--out`. VoiceQA never imports the
plugin, so its licence (Piper is GPL-3.0, for example) does not reach ours.
"""

import json
import re
import shutil
import tempfile
from pathlib import Path

from .interfaces import Speech
from .tts_process import DEFAULT_TIMEOUT, TtsError, read_speech_wav, run_tts_process

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EXECUTABLE_PREFIX = "voiceqa-tts-"


class PluginTts:
    def __init__(self, plugin: str, timeout: float = DEFAULT_TIMEOUT) -> None:
        if not _NAME_RE.match(plugin):
            raise ValueError(
                f"bad plugin name {plugin!r}: use letters, digits, '.', '_' and '-' only"
            )
        self._plugin = plugin
        self._timeout = timeout

    @property
    def name(self) -> str:
        return f"plugin:{self._plugin}"

    @property
    def executable(self) -> str:
        return EXECUTABLE_PREFIX + self._plugin

    def voices(self) -> list[str]:
        """The plugin contract has no way to list voices."""
        return []

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        path = shutil.which(self.executable)
        if path is None:
            raise TtsError(
                f"TTS plugin {self._plugin!r} not found: no {self.executable} on PATH. "
                "Install the plugin or add its folder to PATH."
            )
        request = json.dumps({"text": text, "voice": voice}).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="voiceqa-tts-") as tmp:
            out = Path(tmp) / "out.wav"
            run_tts_process(
                [path, "--out", str(out)],
                label=self.executable,
                stdin=request,
                timeout=self._timeout,
            )
            return read_speech_wav(out, self.executable)
