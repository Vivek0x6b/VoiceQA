"""Text-to-speech and speech-to-text interfaces, fakes and WAV prompts.

See `voiceqa.speech.registry` for how engines are looked up by name and the
subprocess plugin contract later TTS plugins implement.
"""

from .fakes import FakeStt, FakeTts
from .interfaces import Speech, SpeechToText, TextToSpeech
from .models import Segment, Transcript, Word
from .registry import create_stt, create_tts
from .wav_prompts import WavPromptTts

__all__ = [
    "FakeStt",
    "FakeTts",
    "Segment",
    "Speech",
    "SpeechToText",
    "TextToSpeech",
    "Transcript",
    "WavPromptTts",
    "Word",
    "create_stt",
    "create_tts",
]
