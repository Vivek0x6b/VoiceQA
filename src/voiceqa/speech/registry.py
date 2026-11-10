"""Looks up TTS and STT engines by name.

TTS: "wav" (WavPromptTts), "fake" (FakeTts), "sapi" (SapiTts, Windows),
"say" (SayTts, macOS) and "plugin:<name>" (PluginTts, see the contract
below). STT: "fake" (FakeStt) and "faster-whisper" (FasterWhisperStt,
imported only when asked for, since it needs the [stt] extra).
default_tts_name() is the engine to use when none is given: sapi on
Windows, say on macOS, wav elsewhere.

Every constructor takes only plain, picklable keyword arguments (str, int,
float, paths, dicts), so an engine can be built inside a spawned worker
process, where anything else could not cross the pickle boundary.

Subprocess TTS plugin contract: an executable on PATH named
`voiceqa-tts-<name>` reads JSON from stdin, `{"text": ..., "voice": ...}`,
and writes a WAV file to the path given by its `--out` argument. Exit 0
means success; a non-zero exit with a message on stderr means failure.
VoiceQA never imports the plugin's code, only runs it as a subprocess
(voiceqa.speech.plugin_tts), with a timeout, and shows its stderr when it
fails.
"""

import sys

from .fakes import FakeStt, FakeTts
from .interfaces import SpeechToText, TextToSpeech
from .os_tts import SapiTts, SayTts
from .plugin_tts import PluginTts
from .wav_prompts import WavPromptTts

PLUGIN_PREFIX = "plugin:"

_TTS_ENGINES: dict[str, type[TextToSpeech]] = {
    "wav": WavPromptTts,
    "fake": FakeTts,
    "sapi": SapiTts,
    "say": SayTts,
}

_STT_ENGINES: dict[str, type[SpeechToText]] = {"fake": FakeStt}


def default_tts_name() -> str:
    if sys.platform == "win32":
        return "sapi"
    if sys.platform == "darwin":
        return "say"
    return "wav"


def _faster_whisper(**kwargs: object) -> SpeechToText:
    from .whisper import FasterWhisperStt

    return FasterWhisperStt(**kwargs)


def create_tts(name: str, **kwargs: object) -> TextToSpeech:
    if name.startswith(PLUGIN_PREFIX):
        return PluginTts(plugin=name[len(PLUGIN_PREFIX) :], **kwargs)
    try:
        engine = _TTS_ENGINES[name]
    except KeyError:
        known = ", ".join([*sorted(_TTS_ENGINES), "plugin:<name>"])
        raise ValueError(f"unknown TTS engine {name!r}, known engines: {known}") from None
    return engine(**kwargs)


def create_stt(name: str, **kwargs: object) -> SpeechToText:
    if name.startswith(PLUGIN_PREFIX):
        raise NotImplementedError(f"{name!r}: STT plugins are not supported, only TTS plugins")
    if name == "faster-whisper":
        return _faster_whisper(**kwargs)
    try:
        engine = _STT_ENGINES[name]
    except KeyError:
        known = ", ".join(sorted([*_STT_ENGINES, "faster-whisper"]))
        raise ValueError(f"unknown STT engine {name!r}, known engines: {known}") from None
    return engine(**kwargs)
