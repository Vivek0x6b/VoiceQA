"""Real OS engine tests. SAPI runs only on Windows and `say` only on macOS;
the CI runners for those OSes have them. Output is checked for length and
level, never for what was said."""

import sys

import numpy as np
import pytest

from voiceqa.speech.os_tts import OUTPUT_RATE, SapiTts, SayTts, parse_say_voices
from voiceqa.speech.registry import create_tts
from voiceqa.speech.tts_process import TtsError

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="SAPI is Windows only")
macos_only = pytest.mark.skipif(sys.platform != "darwin", reason="say is macOS only")

SENTENCE = "Hello, this is a VoiceQA test call."

# Text that would run commands if it were pasted into a PowerShell or shell
# command line: quotes, $ expressions, backticks, a subexpression, a
# statement separator and a newline. Each would create a marker file.
INJECTION = (
    "it's \"quoted\" $env:USERNAME `n $(New-Item -ItemType File -Path '{d}/a') "
    "'; New-Item -ItemType File -Path '{d}/b'; ' \"; New-Item -Path '{d}/c'; \"\n"
    "$(touch {d}/d) `touch {d}/e` ; touch {d}/f\n- dash line"
)


def assert_sounds_like_speech(speech, text):
    seconds = len(speech.samples) / speech.rate
    assert speech.samples.dtype == np.int16
    # Generous bounds: about 0.03 to 1 s per character covers any voice.
    assert 0.03 * len(text) < seconds < 1.0 * len(text) + 2
    rms = np.sqrt(np.mean(speech.samples.astype(np.float64) ** 2))
    assert rms > 300, f"rms {rms:.0f}"
    assert np.abs(speech.samples).max() > 3000


@windows_only
def test_sapi_speaks_a_sentence():
    tts = create_tts("sapi")
    assert isinstance(tts, SapiTts)
    speech = tts.synthesize(SENTENCE)
    assert speech.rate == OUTPUT_RATE
    assert_sounds_like_speech(speech, SENTENCE)


@windows_only
def test_sapi_text_cannot_inject_commands(tmp_path):
    text = INJECTION.format(d=tmp_path.as_posix())
    speech = SapiTts().synthesize(text)
    assert len(speech.samples) > 0
    assert list(tmp_path.iterdir()) == []


@windows_only
def test_sapi_lists_voices_and_uses_one():
    tts = SapiTts()
    voices = tts.voices()
    assert voices, "no SAPI voices installed"
    speech = tts.synthesize("Testing one two three.", voice=voices[0])
    assert_sounds_like_speech(speech, "Testing one two three.")


@windows_only
def test_sapi_unknown_voice_is_an_error_with_the_reason():
    with pytest.raises(TtsError, match="sapi failed") as exc_info:
        SapiTts().synthesize("hello", voice="No Such Voice 123")
    # The message is PowerShell's exception text, not its CLIXML error stream.
    assert "CLIXML" not in str(exc_info.value)
    assert "SelectVoice" in str(exc_info.value)


@pytest.mark.skipif(sys.platform == "win32", reason="checks the error off Windows")
def test_sapi_off_windows_says_it_needs_windows():
    with pytest.raises(TtsError, match="needs Windows"):
        SapiTts()


def test_parse_say_voices_handles_every_column_width():
    output = (
        "Albert              en_US    # Hello! My name is Albert.\n"
        "Bad News            en_US    # Hello! My name is Bad News.\n"
        "\n"
        "Eddy (English (UK)) en_GB    # Hello! My name is Eddy.\n"
        "Grandma (German (Germany)) de_DE    # Hallo! Ich heisse Grandma.\n"
        "   \n"
        "Samantha            en_US    # Hello, my name is Samantha. # really\n"
        "no hash here\n"
    )
    assert parse_say_voices(output) == [
        "Albert",
        "Bad News",
        "Eddy (English (UK))",
        "Grandma (German (Germany))",
        "Samantha",
    ]


SAY_VOICES_OUTPUT = (
    b"Albert              en_US    # Hello! My name is Albert.\n"
    b"Eddy (English (UK)) en_GB    # Hello! My name is Eddy.\n"
    b"Samantha            en_US    # Hello, my name is Samantha.\n"
)


@pytest.fixture
def fake_say(monkeypatch):
    """SayTts on any OS, with `say` replaced by a fake that records its
    arguments, answers `-v ?` with SAY_VOICES_OUTPUT and writes a short WAV."""
    from voiceqa.audio.dsp import generate_tone
    from voiceqa.audio.wav import write_wav
    from voiceqa.speech import os_tts

    calls = []

    def run(args, *, label, stdin=b"", timeout=0, env=None):
        calls.append(list(args))
        if args[1:] == ["-v", "?"]:
            return SAY_VOICES_OUTPUT
        out = args[args.index("-o") + 1]
        write_wav(out, generate_tone(300, 0.2, sample_rate=OUTPUT_RATE), OUTPUT_RATE)
        return b""

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(os_tts, "run_tts_process", run)
    return calls


def test_say_unknown_voice_is_rejected_before_speaking(fake_say):
    with pytest.raises(TtsError) as exc_info:
        SayTts().synthesize("hello", voice="No Such Voice")
    message = str(exc_info.value)
    assert "No Such Voice" in message
    assert "voiceqa say --list-voices" in message
    # Only the voice list was asked for; nothing was synthesized.
    assert fake_say == [["say", "-v", "?"]]


def test_say_voice_match_is_exact_and_suggests_the_right_case(fake_say):
    with pytest.raises(TtsError, match="did you mean 'Samantha'"):
        SayTts().synthesize("hello", voice="samantha")


def test_say_known_voice_is_passed_to_say(fake_say):
    speech = SayTts().synthesize("hello", voice="Eddy (English (UK))")
    assert speech.rate == OUTPUT_RATE
    args = fake_say[-1]
    assert args[args.index("-v") + 1] == "Eddy (English (UK))"


def test_say_without_a_voice_skips_the_voice_list(fake_say):
    SayTts().synthesize("hello")
    assert len(fake_say) == 1
    assert "-v" not in fake_say[0]


@macos_only
def test_say_speaks_a_sentence():
    tts = create_tts("say")
    assert isinstance(tts, SayTts)
    speech = tts.synthesize(SENTENCE)
    assert speech.rate == OUTPUT_RATE
    assert_sounds_like_speech(speech, SENTENCE)


@macos_only
def test_say_text_cannot_inject_commands(tmp_path):
    text = INJECTION.format(d=tmp_path.as_posix())
    speech = SayTts().synthesize(text)
    assert len(speech.samples) > 0
    assert list(tmp_path.iterdir()) == []


@macos_only
def test_say_lists_voices_and_uses_one():
    tts = SayTts()
    voices = tts.voices()
    assert voices, "no say voices installed"
    voice = "Samantha" if "Samantha" in voices else voices[0]
    speech = tts.synthesize("Testing one two three.", voice=voice)
    assert speech.rate == OUTPUT_RATE
    assert len(speech.samples) > 0


@macos_only
def test_say_unknown_voice_is_an_error():
    with pytest.raises(TtsError, match="say"):
        SayTts().synthesize("hello", voice="No Such Voice 123")


@pytest.mark.skipif(sys.platform == "darwin", reason="checks the error off macOS")
def test_say_off_macos_says_it_needs_macos():
    with pytest.raises(TtsError, match="needs macOS"):
        SayTts()
