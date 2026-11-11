import re

import pytest

from voiceqa.scenario.schema import parse_duration, parse_goto
from voiceqa.scenario.variables import UnknownVariableError, substitute


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("10s", 10.0),
        ("500ms", 0.5),
        ("1.5s", 1.5),
        (8, 8.0),
        (2.25, 2.25),
        ("3", 3.0),
        ("0.75", 0.75),
    ],
)
def test_durations_are_seconds(value, seconds):
    assert parse_duration(value) == seconds


@pytest.mark.parametrize("value", ["10m", "s", "1.5 s", "ten", "", "1e3s", "-2s", "nan", "inf"])
def test_bad_duration_text_is_refused(value):
    with pytest.raises(ValueError, match=r"bad duration .*, use seconds like 10, 1\.5s or 500ms"):
        parse_duration(value)


@pytest.mark.parametrize("value", [0, "0s", "0ms", -1, -0.5])
def test_duration_must_be_above_zero(value):
    with pytest.raises(ValueError, match="must be above 0"):
        parse_duration(value)


@pytest.mark.parametrize("value", [True, None, [1], float("nan"), float("inf")])
def test_duration_of_the_wrong_type_is_refused(value):
    with pytest.raises(ValueError, match="bad duration"):
        parse_duration(value)


def test_goto_gives_the_target_label():
    assert parse_goto("goto human") == "human"
    assert parse_goto("goto end") == "end"
    assert parse_goto("  goto   give-name ") == "give-name"


@pytest.mark.parametrize("value", ["human", "goto", "go to human", "goto a b", "GOTO x", ""])
def test_goto_needs_the_goto_form(value):
    with pytest.raises(ValueError, match=re.escape("must be 'goto <label>', got " + repr(value))):
        parse_goto(value)


VARS = {"name": "Alex", "party": "4"}


def test_variables_are_substituted():
    assert substitute("a table for {party}, {name}", VARS) == "a table for 4, Alex"


def test_double_braces_escape():
    assert substitute("{{name}} is {name}", VARS) == "{name} is Alex"
    assert substitute("{{{name}", VARS) == "{Alex"
    assert substitute("{{", VARS) == "{"


def test_braces_around_a_non_name_are_left_alone():
    # Regex quantifiers do not need escaping.
    assert substitute(r"\d{2,3} or {1} or { name }", VARS) == r"\d{2,3} or {1} or { name }"


def test_values_are_not_substituted_again():
    assert substitute("{a}", {"a": "{b}", "b": "x"}) == "{b}"


def test_escape_is_applied_to_values_only():
    assert substitute(r"hi,? {name}\.", {"name": "J.R."}, escape=re.escape) == r"hi,? J\.R\.\."


def test_unknown_variable_names_every_missing_name():
    with pytest.raises(UnknownVariableError) as caught:
        substitute("{nam} and {name} and {who}", VARS)
    assert caught.value.names == ("nam", "who")
    assert str(caught.value) == "unknown variable {nam}, {who} (defined: name, party)"


def test_unknown_variable_with_none_defined():
    with pytest.raises(
        UnknownVariableError, match=r"unknown variable \{x\} \(no variables defined\)"
    ):
        substitute("{x}", {})
