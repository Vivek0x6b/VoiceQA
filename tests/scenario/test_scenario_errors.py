"""One test per validation error, each checking the exact message."""

import textwrap

import pytest

from voiceqa.scenario import ScenarioError, check_scenario, load_scenario

# (test id, YAML body, expected problem). Bodies are inserted under a valid
# `name:` line unless they start with "!", which means the whole file.
CASES = [
    (
        "unknown-goto-label",
        """
        steps:
          - {label: ask, say: hi, expect: {contains: name}, on_fail: goto humn}
          - {label: human, say: help}
        """,
        "step 1 (ask): on_fail: no step labelled 'humn'",
    ),
    (
        "unknown-on-match-label",
        """
        steps:
          - {label: ask, say: hi, on_match: goto nope}
        """,
        "step 1 (ask): on_match: no step labelled 'nope'",
    ),
    (
        "goto-without-goto",
        """
        steps:
          - {label: ask, say: hi, expect: {contains: x}, on_fail: human}
        """,
        "step 1 (ask): on_fail: must be 'goto <label>', got 'human'",
    ),
    (
        "duplicate-label",
        """
        steps:
          - {label: ask, say: hi}
          - {label: other, say: hi}
          - {label: ask, say: again}
        """,
        "step 3 (ask): duplicate label 'ask', also step 1",
    ),
    (
        "duplicate-of-a-default-label",
        """
        steps:
          - {say: hi}
          - {label: step-1, say: again}
        """,
        "step 2 (step-1): duplicate label 'step-1', also step 1",
    ),
    (
        "default-label-taken-by-an-earlier-step",
        """
        steps:
          - {label: step-2, say: hi}
          - {say: again}
        """,
        "step 2: duplicate label 'step-2', also step 1",
    ),
    (
        "say-empty-after-variables",
        """
        variables: {name: ""}
        steps:
          - {label: a, say: "{name}"}
        """,
        "step 1 (a): say: is empty after filling in variables",
    ),
    (
        "wav-text-empty-after-variables",
        """
        variables: {name: " "}
        steps:
          - {label: a, say: {wav: none.wav, text: "{name}"}}
        """,
        "step 1 (a): say.text: is empty after filling in variables",
    ),
    (
        "wav-text-blank",
        """
        steps:
          - {label: a, say: {wav: none.wav, text: "  "}}
        """,
        "step 1 (a): say.text: must not be empty",
    ),
    (
        "any-of-a-single-string",
        """
        steps:
          - {label: a, expect: {any_of: hello}}
        """,
        "step 1 (a): expect.any_of: must be a list of phrases",
    ),
    (
        "step-end-silence-zero",
        """
        steps:
          - {label: a, say: hi, end_silence_ms: 0}
        """,
        "step 1 (a): end_silence_ms: must be above 0, got 0",
    ),
    (
        "reserved-label-end",
        """
        steps:
          - {label: end, say: hi}
        """,
        "step 1 (end): label 'end' is reserved",
    ),
    (
        "label-with-spaces",
        """
        steps:
          - {label: give name, say: hi}
        """,
        "step 1 (give name): label: must be one word with no spaces, got 'give name'",
    ),
    (
        "neither-say-nor-expect",
        """
        steps:
          - {label: greeting, say: hi}
          - {label: empty}
        """,
        "step 2 (empty): needs say or expect",
    ),
    (
        "unlabelled-step-is-named-by-index",
        """
        steps:
          - {say: hi}
          - {complete: true}
        """,
        "step 2: needs say or expect",
    ),
    (
        "empty-say-text",
        """
        steps:
          - {label: a, say: "  "}
        """,
        "step 1 (a): say: must not be empty",
    ),
    (
        "say-of-the-wrong-type",
        """
        steps:
          - {label: a, say: 5}
        """,
        "step 1 (a): say: must be text or {wav: path}",
    ),
    (
        "say-wav-unknown-key",
        """
        steps:
          - {label: a, say: {wav: a.wav, volume: 2}}
        """,
        "step 1 (a): say: unknown key 'volume'",
    ),
    (
        "say-wav-missing-path",
        """
        steps:
          - {label: a, say: {}}
        """,
        "step 1 (a): say: missing key 'wav'",
    ),
    (
        "missing-wav-file",
        """
        steps:
          - {label: a, say: {wav: prompts/none.wav}}
        """,
        "step 1 (a): say.wav: no such WAV file: {dir}/prompts/none.wav",
    ),
    (
        "bad-regex",
        """
        steps:
          - {label: a, expect: {regex: "thank(s"}}
        """,
        "step 1 (a): expect.regex: bad regex 'thank(s': "
        "missing ), unterminated subpattern at position 5",
    ),
    (
        "bad-duration-in-expect",
        """
        steps:
          - {label: a, expect: {contains: x, timeout: 10m}}
        """,
        "step 1 (a): expect.timeout: bad duration '10m', use seconds like 10, 1.5s or 500ms",
    ),
    (
        "bad-duration-in-defaults",
        """
        defaults: {timeout: soon}
        steps:
          - {say: hi}
        """,
        "defaults.timeout: bad duration 'soon', use seconds like 10, 1.5s or 500ms",
    ),
    (
        "zero-duration",
        """
        defaults: {timeout: 0s}
        steps:
          - {say: hi}
        """,
        "defaults.timeout: duration must be above 0, got '0s'",
    ),
    (
        "negative-step-retries",
        """
        steps:
          - {label: a, expect: {contains: x}, retries: -1}
        """,
        "step 1 (a): retries: must be 0 or more, got -1",
    ),
    (
        "negative-default-retries",
        """
        defaults: {retries: -2}
        steps:
          - {say: hi}
        """,
        "defaults.retries: must be 0 or more, got -2",
    ),
    (
        "max-steps-below-one",
        """
        max_steps: 0
        steps:
          - {say: hi}
        """,
        "max_steps: must be 1 or more, got 0",
    ),
    (
        "task-completed-without-complete-step",
        """
        steps:
          - {say: hi, expect: {contains: x}}
        assert: [call_connected, task_completed]
        """,
        "assert: task_completed needs a step with complete: true",
    ),
    (
        "unknown-assertion",
        """
        steps:
          - {say: hi}
        assert: [call_connected, polite]
        """,
        "assert: unknown assertion 'polite', expected call_connected, agent_responded "
        "or task_completed",
    ),
    (
        "unknown-top-level-key",
        """
        descripton: typo
        steps:
          - {say: hi}
        """,
        "scenario: unknown key 'descripton'",
    ),
    (
        "unknown-step-key",
        """
        steps:
          - {label: a, sya: hi}
        """,
        "step 1 (a): unknown key 'sya'",
    ),
    (
        "unknown-expect-key",
        """
        steps:
          - {label: a, expect: {contain: x}}
        """,
        "step 1 (a): expect: unknown key 'contain'",
    ),
    (
        "unknown-fuzzy-key",
        """
        steps:
          - {label: a, expect: {fuzzy: {text: x, minimum: 0.5}}}
        """,
        "step 1 (a): expect.fuzzy: unknown key 'minimum'",
    ),
    (
        "unknown-defaults-key",
        """
        defaults: {timeuot: 5}
        steps:
          - {say: hi}
        """,
        "defaults: unknown key 'timeuot'",
    ),
    (
        "unknown-variable",
        """
        variables: {name: Alex}
        steps:
          - {label: a, say: "hi {nmae}"}
        """,
        "step 1 (a): say: unknown variable {nmae} (defined: name)",
    ),
    (
        "unknown-variable-in-expect",
        """
        steps:
          - {label: a, expect: {fuzzy: {text: "bye {who}"}}}
        """,
        "step 1 (a): expect.fuzzy.text: unknown variable {who} (no variables defined)",
    ),
    (
        "bad-variable-name",
        """
        variables: {first-name: Alex}
        steps:
          - {say: hi}
        """,
        "variables: bad name 'first-name', use letters, digits and underscores",
    ),
    (
        "variable-of-the-wrong-type",
        """
        variables: {name: [a, b]}
        steps:
          - {say: hi}
        """,
        "variables.name: must be text or a number, got ['a', 'b']",
    ),
    (
        "missing-name",
        """!
        steps:
          - {say: hi}
        """,
        "scenario: missing key 'name'",
    ),
    (
        "missing-steps",
        """
        variables: {}
        """,
        "scenario: missing key 'steps'",
    ),
    (
        "empty-steps",
        """
        steps: []
        """,
        "steps: needs at least one step",
    ),
    (
        "empty-any-of",
        """
        steps:
          - {label: a, expect: {any_of: []}}
        """,
        "step 1 (a): expect.any_of: needs at least one phrase",
    ),
    (
        "text-with-no-words",
        """
        steps:
          - {label: a, expect: {contains: "?!"}}
        """,
        "step 1 (a): expect.contains: '?!' has no words after normalising",
    ),
    (
        "any-of-item-with-no-words",
        """
        steps:
          - {label: a, expect: {any_of: [hello, "..."]}}
        """,
        "step 1 (a): expect.any_of: '...' has no words after normalising",
    ),
    (
        "fuzzy-text-with-no-words",
        """
        steps:
          - {label: a, expect: {fuzzy: {text: "-"}}}
        """,
        "step 1 (a): expect.fuzzy.text: '-' has no words after normalising",
    ),
    (
        "fuzzy-min-out-of-range",
        """
        steps:
          - {label: a, expect: {fuzzy: {text: hi, min: 1.5}}}
        """,
        "step 1 (a): expect.fuzzy.min: must be above 0 and at most 1, got 1.5",
    ),
    (
        "max-latency-not-positive",
        """
        steps:
          - {label: a, expect: {contains: hi, max_latency_ms: 0}}
        """,
        "step 1 (a): expect.max_latency_ms: must be above 0, got 0",
    ),
    (
        "end-silence-not-positive",
        """
        defaults: {end_silence_ms: -5}
        steps:
          - {say: hi}
        """,
        "defaults.end_silence_ms: must be above 0, got -5",
    ),
    (
        "on-fail-without-expect",
        """
        steps:
          - {label: a, say: hi, on_fail: goto b}
          - {label: b, say: bye}
        """,
        "step 1 (a): on_fail needs an expect",
    ),
    (
        "retries-without-expect",
        """
        steps:
          - {label: a, say: hi, retries: 2}
        """,
        "step 1 (a): retries needs an expect",
    ),
    (
        "wrong-type-in-a-list",
        """
        steps:
          - {label: a, expect: {any_of: [hello, 5]}}
        """,
        "step 1 (a): expect.any_of[1]: Input should be a valid string",
    ),
    (
        "not-a-mapping",
        """!
        - just
        - a list
        """,
        "scenario: must be a mapping with name and steps",
    ),
]


@pytest.mark.parametrize(("body", "problem"), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_validation_error(write_scenario, tmp_path, body, problem):
    text = textwrap.dedent(body[1:] if body.startswith("!") else body)
    if not body.startswith("!"):
        text = "name: t\n" + text
    path = write_scenario(text)
    wav = (tmp_path / "prompts" / "none.wav").resolve()
    problem = problem.replace("{dir}/prompts/none.wav", str(wav))
    problems = check_scenario(path)
    assert problem in problems, problems
    with pytest.raises(ScenarioError) as caught:
        load_scenario(path)
    assert problem in caught.value.problems


def test_shape_problems_are_all_reported_at_once(write_scenario):
    path = write_scenario(
        "name: t\nmax_steps: 0\nsteps:\n  - {label: a, retries: -1, say: hi}\n  - {sya: x}\n"
    )
    assert check_scenario(path) == [
        "max_steps: must be 1 or more, got 0",
        "step 1 (a): retries: must be 0 or more, got -1",
        "step 2: unknown key 'sya'",
    ]


def test_override_with_a_bad_name(write_scenario):
    path = write_scenario("name: t\nsteps:\n  - {say: hi}\n")
    assert check_scenario(path, overrides={"a b": "x"}) == [
        "variable override: bad name 'a b', use letters, digits and underscores"
    ]


def test_missing_file(tmp_path):
    path = tmp_path / "nope.yaml"
    assert check_scenario(path) == [f"no such scenario file: {path}"]


def test_not_valid_yaml(write_scenario):
    path = write_scenario("name: [unclosed\n")
    (problem,) = check_scenario(path)
    assert problem.startswith("not valid YAML: ")


def test_not_utf8(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"name: \xff\xfe\n")
    (problem,) = check_scenario(path)
    assert problem.startswith("cannot read it: ")
