"""Looks up TTS and STT engines by name.

Known now: "wav" (WavPromptTts) and "fake" (FakeTts / FakeStt). "sapi" and
"say" (OS TTS engines) and "faster-whisper" (STT) are reserved for later
tasks and raise NotImplementedError until then, rather than being unknown
names. A name of the form "plugin:<name>" parses (see the subprocess
contract below) but also raises NotImplementedError until the plugin
runner lands.

Every constructor takes only plain, picklable keyword arguments (str, int,
float, paths, dicts): a later task builds engines inside a spawned worker
process, where anything else could not cross the pickle boundary.

Subprocess TTS plugin contract (for the plugin runner task): an executable
on PATH named `voiceqa-tts-<name>` reads JSON from stdin,
`{"text": ..., "voice": ...}`, and writes a WAV file to the path given by
its `--out` argument. Exit 0 means success; a non-zero exit with a message
on stderr means failure. VoiceQA never imports the plugin's code, only runs
it as a subprocess.
"""

from .fakes import FakeStt, FakeTts
from .interfaces import SpeechToText, TextToSpeech
from .wav_prompts import WavPromptTts

_TTS_ENGINES: dict[str, type[TextToSpeech]] = {"wav": WavPromptTts, "fake": FakeTts}
_RESERVED_TTS = ("sapi", "say")

_STT_ENGINES: dict[str, type[SpeechToText]] = {"fake": FakeStt}
_RESERVED_STT = ("faster-whisper",)


def _reject_plugin_for_now(name: str) -> None:
    if name.startswith("plugin:"):
        raise NotImplementedError(
            f"{name!r} is a plugin engine; plugin support is not implemented "
            "until the subprocess plugin runner task"
        )


def create_tts(name: str, **kwargs: object) -> TextToSpeech:
    _reject_plugin_for_now(name)
    if name in _RESERVED_TTS:
        raise NotImplementedError(f"TTS engine {name!r} is not implemented yet")
    try:
        engine = _TTS_ENGINES[name]
    except KeyError:
        known = ", ".join(sorted(_TTS_ENGINES))
        raise ValueError(f"unknown TTS engine {name!r}, known engines: {known}") from None
    return engine(**kwargs)


def create_stt(name: str, **kwargs: object) -> SpeechToText:
    _reject_plugin_for_now(name)
    if name in _RESERVED_STT:
        raise NotImplementedError(f"STT engine {name!r} is not implemented yet")
    try:
        engine = _STT_ENGINES[name]
    except KeyError:
        known = ", ".join(sorted(_STT_ENGINES))
        raise ValueError(f"unknown STT engine {name!r}, known engines: {known}") from None
    return engine(**kwargs)
