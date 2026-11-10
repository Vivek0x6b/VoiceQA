from voiceqa.speech import Segment, Transcript, Word


def test_word_round_trip():
    word = Word(start=0.0, end=0.5, text="hi", probability=0.9)
    again = Word.model_validate_json(word.model_dump_json())
    assert again == word


def test_segment_defaults():
    segment = Segment(start=0.0, end=1.0, text="hello world")
    assert segment.words == []
    assert segment.speaker is None


def test_transcript_round_trip():
    transcript = Transcript(
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                text="hello",
                words=[Word(start=0.0, end=0.4, text="hello", probability=0.95)],
                speaker="caller",
            )
        ],
        engine="fake",
        model="scripted",
        language="en",
    )
    again = Transcript.model_validate_json(transcript.model_dump_json())
    assert again == transcript


def test_transcript_defaults_to_no_segments():
    transcript = Transcript(engine="fake", model="scripted", language="en")
    assert transcript.segments == []
