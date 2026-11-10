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
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

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


# `say` sometimes exits 0 having written only a few milliseconds of audio
# for a whole sentence (seen once on a macOS CI runner: 0.005 s). Text with
# at least one letter or digit must give at least MIN_SPEECH_SECONDS of
# audio that is not all silence (peak at or above SILENT_PEAK), otherwise
# the output counts as a failed synthesis and is retried once.
MIN_SPEECH_SECONDS = 0.1
SILENT_PEAK = 64


def _peak(speech: Speech) -> int:
    return int(np.abs(speech.samples.astype(np.int32)).max()) if len(speech.samples) else 0


def _missing_audio(text: str, speech: Speech) -> bool:
    if not any(ch.isalnum() for ch in text):
        return False
    too_short = len(speech.samples) < MIN_SPEECH_SECONDS * speech.rate
    return too_short or _peak(speech) < SILENT_PEAK


def parse_say_voices(output: str) -> list[str]:
    """Voice names from `say -v ?`.

    Each line is "<name> <locale> # <sample text>". Names may contain spaces
    and long ones fill the column, leaving a single space before the locale,
    so the line is split on " #" and the last word (the locale) dropped
    instead of relying on column widths.
    """
    names = []
    for line in output.splitlines():
        head, sep, _ = line.partition(" #")
        parts = head.strip().rsplit(None, 1)
        if sep and len(parts) == 2:
            names.append(parts[0])
    return names


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
        return parse_say_voices(out.decode("utf-8", "replace"))

    def _check_voice(self, voice: str) -> None:
        """`say -v <unknown>` speaks with the default voice and exits 0, so the
        name is checked against `say -v ?` first. The match is exact, the same
        as the cache key, so one voice never gets two cache entries."""
        names = self.voices()
        if voice in names:
            return
        message = f"say has no voice {voice!r}"
        same_but_case = [n for n in names if n.casefold() == voice.casefold()]
        if same_but_case:
            message += f", did you mean {same_but_case[0]!r}?"
        raise TtsError(f"{message}. See the voices with: voiceqa say --list-voices --tts say")

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
                self._check_voice(voice)
                args[1:1] = ["-v", voice]
            speech = None
            for _ in range(2):
                out.unlink(missing_ok=True)
                run_tts_process(args, label="say", timeout=self._timeout)
                speech = read_speech_wav(out, "say")
                if not _missing_audio(text, speech):
                    return speech
            seconds = len(speech.samples) / speech.rate
            raise TtsError(
                f"say exited 0 but wrote no audio twice in a row ({seconds:.3f} s, "
                f"peak {_peak(speech)}) for {len(text)} characters of text"
            )
