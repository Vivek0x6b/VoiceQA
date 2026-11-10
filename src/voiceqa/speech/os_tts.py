"""The speech engines that come with the OS: SAPI on Windows, `say` on macOS.

SapiTts runs Windows PowerShell with System.Speech instead of using COM, so
core needs no pywin32. The PowerShell script is a constant passed with
-EncodedCommand. The text never goes into a command line: it is written to
a UTF-8 file in a temp folder, and the script gets that file's path (and
the voice name and output path) through environment variables. Quotes, `$`
or backticks in the text therefore cannot run anything.

SayTts runs `say` with the text in a file (-f), so text starting with "-"
is never read as an option, and asks for 16-bit little-endian WAV.

Both return the engine's own sample rate (22050 Hz); calls resample later.
"""

import base64
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from .interfaces import Speech
from .tts_process import DEFAULT_TIMEOUT, TtsError, read_speech_wav, run_tts_process

OUTPUT_RATE = 22050

# Errors are written straight to the console's stderr and the script exits
# 1: PowerShell's own error output would arrive as CLIXML, since its output
# is redirected.
_SAPI_WRAPPER = """
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try {
    [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
BODY
    } finally {
        $synth.Dispose()
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
"""

_SAPI_SPEAK = _SAPI_WRAPPER.replace(
    "BODY",
    f"""
        if ($env:VOICEQA_TTS_VOICE) {{ $synth.SelectVoice($env:VOICEQA_TTS_VOICE) }}
        $text = [System.IO.File]::ReadAllText(
            $env:VOICEQA_TTS_TEXT, [System.Text.Encoding]::UTF8)
        $format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
            {OUTPUT_RATE},
            [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
            [System.Speech.AudioFormat.AudioChannel]::Mono)
        $synth.SetOutputToWaveFile($env:VOICEQA_TTS_OUT, $format)
        $synth.Speak($text)
        $synth.SetOutputToNull()
""",
)

_SAPI_VOICES = _SAPI_WRAPPER.replace(
    "BODY",
    """
        foreach ($v in $synth.GetInstalledVoices()) {
            if ($v.Enabled) { [Console]::WriteLine($v.VoiceInfo.Name) }
        }
""",
)

# One line of `say -v ?`: the name (which may contain spaces), two or more
# spaces, the locale, then "# sample text".
_SAY_VOICE_RE = re.compile(r"^(.+?)\s{2,}\S+\s+#")


def _powershell() -> str:
    found = shutil.which("powershell")
    if found:
        return found
    root = os.environ.get("SystemRoot", r"C:\Windows")
    return str(Path(root) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe")


def _encoded(script: str) -> str:
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def _powershell_args(script: str) -> list[str]:
    return [_powershell(), "-NoProfile", "-NonInteractive", "-EncodedCommand", _encoded(script)]


class SapiTts:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        if sys.platform != "win32":
            raise TtsError("the sapi TTS engine needs Windows")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "sapi"

    def voices(self) -> list[str]:
        out = run_tts_process(
            _powershell_args(_SAPI_VOICES), label="sapi voice list", timeout=self._timeout
        )
        lines = out.decode("utf-8", "replace").splitlines()
        return [line.strip() for line in lines if line.strip()]

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        with tempfile.TemporaryDirectory(prefix="voiceqa-sapi-") as tmp:
            text_path = Path(tmp) / "text.txt"
            text_path.write_text(text, encoding="utf-8")
            out = Path(tmp) / "out.wav"
            env = dict(os.environ)
            env["VOICEQA_TTS_TEXT"] = str(text_path)
            env["VOICEQA_TTS_OUT"] = str(out)
            env["VOICEQA_TTS_VOICE"] = voice or ""
            run_tts_process(
                _powershell_args(_SAPI_SPEAK), label="sapi", timeout=self._timeout, env=env
            )
            return read_speech_wav(out, "sapi")


class SayTts:
    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        if sys.platform != "darwin":
            raise TtsError("the say TTS engine needs macOS")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "say"

    def voices(self) -> list[str]:
        out = run_tts_process(["say", "-v", "?"], label="say voice list", timeout=self._timeout)
        names = []
        for line in out.decode("utf-8", "replace").splitlines():
            match = _SAY_VOICE_RE.match(line)
            if match:
                names.append(match.group(1).strip())
        return names

    def synthesize(self, text: str, voice: str | None = None) -> Speech:
        with tempfile.TemporaryDirectory(prefix="voiceqa-say-") as tmp:
            text_path = Path(tmp) / "text.txt"
            text_path.write_text(text, encoding="utf-8")
            out = Path(tmp) / "out.wav"
            args = [
                "say",
                "-o",
                str(out),
                "--file-format=WAVE",
                f"--data-format=LEI16@{OUTPUT_RATE}",
                "-f",
                str(text_path),
            ]
            if voice:
                args[1:1] = ["-v", voice]
            run_tts_process(args, label="say", timeout=self._timeout)
            return read_speech_wav(out, "say")
