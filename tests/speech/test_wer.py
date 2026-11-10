import pytest

from voiceqa.speech.wer import normalize_words, wer


def test_identical_text_has_no_errors():
    assert wer("the cat sat", "the cat sat") == 0.0


def test_one_substitution():
    assert wer("the cat sat", "the bat sat") == pytest.approx(1 / 3)


def test_one_deletion():
    assert wer("the cat sat", "the sat") == pytest.approx(1 / 3)


def test_insertions_count_against_the_reference_length():
    assert wer("the cat", "the cat sat down") == pytest.approx(2 / 2)
    assert wer("hi", "oh hi there you") == pytest.approx(3.0)


def test_empty_hypothesis_is_all_deletions():
    assert wer("a b c d", "") == pytest.approx(1.0)


def test_mixed_edits_take_the_cheapest_alignment():
    # i want [to] book [a->the] table for [two->to]: 1 deletion, 2 substitutions.
    assert wer("i want to book a table for two", "i want book the table for to") == pytest.approx(
        3 / 8
    )


def test_case_and_punctuation_are_ignored():
    assert wer("Hello, World! How are you?", "hello world how are you") == 0.0


def test_numbers_stay_as_written():
    assert wer("20 dollars", "twenty dollars") == pytest.approx(1 / 2)
    assert normalize_words("Pay 1,000 now, or 3.5 later.") == [
        "pay",
        "1,000",
        "now",
        "or",
        "3.5",
        "later",
    ]


def test_apostrophes_inside_words_are_kept():
    assert normalize_words("Don't 'quote' me") == ["don't", "quote", "me"]
    assert normalize_words("don" + chr(0x2019) + "t") == ["don't"]
    assert wer("don't stop", "dont stop") == pytest.approx(1 / 2)


def test_hyphens_split_words():
    assert normalize_words("twenty-one") == ["twenty", "one"]


def test_empty_reference_is_an_error():
    with pytest.raises(ValueError, match="reference"):
        wer("", "anything")
    with pytest.raises(ValueError, match="reference"):
        wer(" ,. ", "")
