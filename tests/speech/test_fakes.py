import numpy as np

from voiceqa.speech import FakeStt, FakeTts, Segment, Transcript, Word


def test_fake_tts_name_and_voices():
    tts = FakeTts()
    assert tts.name == "fake"
    assert tts.voices() == ["fake"]


def test_fake_tts_length_follows_text():
    tts = FakeTts(seconds_per_char=0.1, min_seconds=0.0)
    short = tts.synthesize("hi")
    long = tts.synthesize("hello there, this is longer")
    assert len(long.samples) > len(short.samples)
    assert short.samples.dtype == np.int16
    assert short.rate == 8000


def test_fake_tts_has_a_floor_for_empty_text():
    tts = FakeTts(min_seconds=0.2)
    speech = tts.synthesize("")
    assert len(speech.samples) == round(0.2 * 8000)


def test_fake_stt_returns_scripted_transcript_from_a_plain_dict():
    scripted = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello", "words": []}],
        "engine": "fake",
        "model": "scripted",
        "language": "en",
    }
    stt = FakeStt(scripted)
    result = stt.transcribe(np.zeros(8000, dtype=np.int16), 8000)
    assert result == Transcript.model_validate(scripted)


def test_fake_stt_returns_scripted_transcript_from_a_transcript_instance():
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                text="hi",
                words=[Word(start=0.0, end=0.3, text="hi", probability=0.8)],
            )
        ],
        engine="fake",
        model="scripted",
        language="en",
    )
    stt = FakeStt(transcript)
    assert stt.transcribe(np.zeros(1, dtype=np.int16), 16000) is transcript


def test_fake_stt_ignores_the_given_audio():
    transcript = Transcript(engine="fake", model="scripted", language="en")
    stt = FakeStt(transcript)
    assert stt.transcribe(np.zeros(5, dtype=np.int16), 8000) == stt.transcribe(
        np.ones(500, dtype=np.int16), 44100
    )
