"""Text-to-speech and speech-to-text interfaces, engines, fakes and WAV prompts.

See `voiceqa.speech.registry` for how engines are looked up by name and the
subprocess plugin contract TTS plugins implement.
"""

from .fakes import FakeStt, FakeTts
from .interfaces import Speech, SpeechToText, TextToSpeech
from .models import Segment, Transcript, Word
from .os_tts import SapiTts, SayTts
from .plugin_tts import PluginTts
from .registry import create_stt, create_tts, default_tts_name
from .tts_process import TtsError
from .wav_prompts import WavPromptTts

__all__ = [
    "FakeStt",
    "FakeTts",
    "PluginTts",
    "SapiTts",
    "SayTts",
    "Segment",
    "Speech",
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
    "TtsError",
    "WavPromptTts",
    "Word",
    "create_stt",
    "create_tts",
    "default_tts_name",
]
