import pytest

from voiceqa.scenario import Expect, Fuzzy
from voiceqa.scenario.match import MatcherResult, fuzzy_similarity, match_expect

RSQUO = chr(0x2019)


def only(expect: Expect, heard: str, latency_ms: float | None = None) -> MatcherResult:
    result = match_expect(expect, heard, latency_ms)
    (matcher,) = result.results
    assert result.passed is matcher.passed
    assert result.heard == heard
    return matcher


# contains


def test_contains_passes_on_the_phrase():
    r = only(Expect(contains=("how can I help",)), "Hello! How can I help you today?")
    assert r == MatcherResult(
        matcher="contains", passed=True, expected="how can I help", reason='heard "how can i help"'
    )


def test_contains_fails_and_says_what_was_missing():
    r = only(Expect(contains=("name",)), "What time would you like?")
    assert r == MatcherResult(
        matcher="contains", passed=False, expected="name", reason='did not hear "name"'
    )


@pytest.mark.parametrize(
    ("phrase", "heard"),
    [
        ("NAME", "and your name?"),
        ("your name", "And YOUR, name?"),
        ("I'd like", "i" + RSQUO + "d LIKE that"),
        ("give-name", "please give name"),
        ("give name", "please give-name"),
        ("table for 4", "A table for 4."),
    ],
)
def test_contains_compares_normalised_words(phrase, heard):
    assert only(Expect(contains=(phrase,)), heard).passed


@pytest.mark.parametrize(
    ("phrase", "heard"),
    [
        ("name", "your username"),
        ("name", "names please"),
        ("your name", "your full name"),
        ("table for 4", "table for four"),
    ],
)
def test_contains_needs_whole_words_in_order(phrase, heard):
    assert not only(Expect(contains=(phrase,)), heard).passed


def test_contains_with_several_phrases_checks_each():
    result = match_expect(Expect(contains=("table", "seven")), "a table at eight")
    assert [(r.expected, r.passed) for r in result.results] == [("table", True), ("seven", False)]
    assert not result.passed


# any_of


def test_any_of_passes_on_any_phrase():
    r = only(Expect(any_of=("how can I help", "how may I help")), "Hi, how may I help?")
    assert r == MatcherResult(
        matcher="any_of",
        passed=True,
        expected='"how can I help" | "how may I help"',
        reason='heard "how may i help"',
    )


def test_any_of_fails_when_none_is_heard():
    r = only(Expect(any_of=("how can I help", "how may I help")), "Hello?")
    assert r.passed is False
    assert r.reason == "heard none of them"


# not_contains


def test_not_contains_passes_when_absent():
    r = only(Expect(not_contains=("sorry",)), "Sure, no problem.")
    assert r == MatcherResult(
        matcher="not_contains", passed=True, expected="sorry", reason='did not hear "sorry"'
    )


def test_not_contains_fails_when_heard_after_normalising():
    r = only(Expect(not_contains=("we are closed",)), "Sorry, WE ARE closed.")
    assert r.passed is False
    assert r.reason == 'heard "we are closed"'


# regex


def test_regex_sees_the_raw_transcript():
    heard = "Thank you, Alex."
    assert only(Expect(regex=r"Thank you, Alex\."), heard).passed
    # The same words would match after normalising, but regex does not normalise.
    assert not only(Expect(regex="thank you alex"), heard).passed
    assert only(Expect(contains=("thank you alex",)), heard).passed


def test_regex_reports_what_matched():
    r = only(Expect(regex="(?i)thank(s| you),? alex"), "Great. Thanks Alex, see you.")
    assert r == MatcherResult(
        matcher="regex",
        passed=True,
        expected="(?i)thank(s| you),? alex",
        reason='matched "Thanks Alex"',
    )


def test_regex_fail_reason():
    r = only(Expect(regex="^yes"), "well, yes")
    assert r.passed is False
    assert r.reason == "no match in what was heard"


# fuzzy


def test_fuzzy_hand_computed_best_window():
    # Expected "see you tonight" (3 words). Heard normalises to
    # great see you to night then, so the 3-word windows score:
    #   great see you   -> insert great, delete tonight: 2 edits, 1/3
    #   see you to      -> tonight to to: 1 edit, 2/3
    #   you to night    -> 3 edits, 0
    #   to night then   -> 3 edits, 0
    score, window = fuzzy_similarity("see you tonight", "Great, see you to night then.")
    assert score == pytest.approx(2 / 3)
    assert window == "see you to"


def test_fuzzy_fails_below_min_and_passes_at_it():
    heard = "Great, see you to night then."
    r = only(Expect(fuzzy=Fuzzy(text="see you tonight", min=0.8)), heard)
    assert (r.matcher, r.passed, r.expected) == ("fuzzy", False, "see you tonight")
    assert r.reason == 'best window "see you to" scores 0.67, needs 0.80'
    assert r.score == pytest.approx(2 / 3)
    assert only(Expect(fuzzy=Fuzzy(text="see you tonight", min=0.66)), heard).passed


@pytest.mark.parametrize(
    "heard",
    [
        "see you tonight, bye now then",
        "okay so see you tonight bye",
        "right, thanks, bye, see you tonight",
    ],
)
def test_fuzzy_finds_an_exact_phrase_anywhere_in_a_long_turn(heard):
    # Against the whole turn these would score 0 (three or more extra words).
    assert fuzzy_similarity("see you tonight", heard)[0] == 1.0


def test_fuzzy_normalises_both_sides():
    assert fuzzy_similarity("See you, TONIGHT!", "see you tonight")[0] == 1.0


def test_fuzzy_against_a_shorter_turn_uses_the_whole_turn():
    assert fuzzy_similarity("see you tonight", "see you") == (pytest.approx(2 / 3), "see you")


def test_fuzzy_against_nothing_scores_zero():
    assert fuzzy_similarity("see you tonight", "") == (0.0, "")
    assert fuzzy_similarity("see you tonight", " ... ") == (0.0, "")


def test_fuzzy_with_no_word_in_common_scores_zero():
    assert fuzzy_similarity("a b", "x") == (0.0, "x")
    assert fuzzy_similarity("a b", "x y z") == (0.0, "x y")


# combined, latency


def test_every_matcher_must_pass_and_the_failure_is_named():
    expect = Expect(
        any_of=("how can I help",),
        not_contains=("sorry",),
        regex="help",
    )
    result = match_expect(expect, "Sorry, how can I help?")
    assert [(r.matcher, r.passed) for r in result.results] == [
        ("any_of", True),
        ("not_contains", False),
        ("regex", True),
    ]
    assert result.passed is False
    assert [r.matcher for r in result.failures] == ["not_contains"]
    assert result.summary() == 'not_contains: heard "sorry"'


def test_summary_of_a_pass_is_empty():
    result = match_expect(Expect(contains=("help",)), "help")
    assert result.passed and result.failures == () and result.summary() == ""


def test_no_matchers_passes_on_any_turn():
    result = match_expect(Expect(), "anything at all")
    assert result.passed and result.results == ()


def test_max_latency():
    expect = Expect(max_latency_ms=3000)
    assert only(expect, "hi", 2999.5) == MatcherResult(
        matcher="max_latency_ms",
        passed=True,
        expected="at most 3000 ms",
        reason="latency 3000 ms",
        score=2999.5,
    )
    r = only(expect, "hi", 3200)
    assert r.passed is False and r.reason == "latency 3200 ms is over 3000 ms"
    r = only(expect, "hi", None)
    assert r.passed is False and r.reason == "no latency measured"
    assert only(expect, "hi", 3000).passed


def test_match_result_dumps_to_json():
    result = match_expect(Expect(contains=("x",)), "y")
    assert result.model_dump(mode="json") == {
        "passed": False,
        "heard": "y",
        "results": [
            {
                "matcher": "contains",
                "passed": False,
                "expected": "x",
                "reason": 'did not hear "x"',
                "score": None,
            }
        ],
    }
