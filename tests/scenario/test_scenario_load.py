import hashlib
import re

import pytest

from voiceqa.scenario import (
    Expect,
    Fuzzy,
    ScenarioError,
    check_scenario,
    load_scenario,
)

SKETCH = """\
name: restaurant booking
variables: {name: Alex, party: "4"}
defaults: {timeout: 10s, end_silence_ms: 700, retries: 0}
steps:
  - label: greeting
    expect: {any_of: ["how can I help", "how may I help"], max_latency_ms: 3000}
  - label: ask
    say: "Hi, I'd like a table for {party} tonight at seven."
    expect: {contains: "name", timeout: 8s}
    on_fail: goto human
  - label: give-name
    say: {wav: prompts/name.wav}
    expect: {regex: "(?i)thank(s| you),? {name}", fuzzy: {text: "see you tonight", min: 0.8}}
    retries: 1
    complete: true
    on_match: goto end
  - label: human
    say: "Can I talk to a person?"
assert: [call_connected, agent_responded, task_completed]
"""


@pytest.fixture
def sketch(write_scenario, silent_wav, tmp_path):
    silent_wav(tmp_path / "prompts" / "name.wav")
    return write_scenario(SKETCH)


def test_the_plan_sketch_loads_with_every_field(sketch, tmp_path):
    s = load_scenario(sketch)

    assert s.name == "restaurant booking"
    assert s.path == sketch.resolve()
    assert s.sha256 == hashlib.sha256(sketch.read_bytes()).hexdigest()
    assert s.variables == {"name": "Alex", "party": "4"}
    assert s.end_silence_ms == 700
    assert s.max_steps == 50
    assert s.assertions == ("call_connected", "agent_responded", "task_completed")
    assert [st.label for st in s.steps] == ["greeting", "ask", "give-name", "human"]
    assert [st.index for st in s.steps] == [1, 2, 3, 4]

    greeting, ask, give_name, human = s.steps
    assert greeting.say_text is None and greeting.say_wav is None
    assert greeting.expect == Expect(
        any_of=("how can I help", "how may I help"), max_latency_ms=3000
    )
    assert greeting.timeout == 10.0
    assert greeting.retries == 0
    assert greeting.complete is False
    assert greeting.on_match is None and greeting.on_fail is None

    assert ask.say_text == "Hi, I'd like a table for 4 tonight at seven."
    assert ask.say_wav is None
    assert ask.expect == Expect(contains=("name",))
    assert ask.timeout == 8.0
    assert ask.on_fail == "human"
    assert ask.on_match is None

    assert give_name.say_text is None
    assert give_name.say_wav == (tmp_path / "prompts" / "name.wav").resolve()
    assert give_name.say_wav.is_absolute()
    assert give_name.expect == Expect(
        regex="(?i)thank(s| you),? Alex", fuzzy=Fuzzy(text="see you tonight", min=0.8)
    )
    assert give_name.retries == 1
    assert give_name.complete is True
    assert give_name.on_match == "end"
    assert give_name.timeout == 10.0

    assert human.say_text == "Can I talk to a person?"
    assert human.expect is None
    assert s.step("human") is human


def test_check_scenario_is_empty_for_a_valid_file(sketch):
    assert check_scenario(sketch) == []


def test_defaults_apply_when_left_out(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: minimal
            steps:
              - say: hello
                expect: {fuzzy: {text: hi there}}
            """
        )
    )
    (step,) = s.steps
    assert step.label == "step-1"
    assert step.timeout == 10.0
    assert step.retries == 0
    assert step.expect.fuzzy.min == 0.8
    assert s.end_silence_ms == 700
    assert s.max_steps == 50
    assert s.assertions == ()
    assert s.variables == {}


def test_step_values_override_the_defaults(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: overrides
            defaults: {timeout: 500ms, retries: 2, end_silence_ms: 900}
            max_steps: 7
            steps:
              - {label: a, say: hi, expect: {contains: ok}}
              - {label: b, say: hi, expect: {contains: ok, timeout: 2}, retries: 0}
            """
        )
    )
    a, b = s.steps
    assert (a.timeout, a.retries) == (0.5, 2)
    assert (b.timeout, b.retries) == (2.0, 0)
    assert s.end_silence_ms == 900
    assert s.max_steps == 7


def test_step_end_silence_ms_falls_back_to_the_default(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: pauses
            defaults: {end_silence_ms: 900}
            steps:
              - {label: a, say: hi, end_silence_ms: 1200}
              - {label: b, say: hi}
            """
        )
    )
    a, b = s.steps
    assert (a.end_silence_ms, b.end_silence_ms) == (1200, 900)


def test_wav_say_takes_optional_text_with_variables(write_scenario, silent_wav, tmp_path):
    silent_wav(tmp_path / "prompts" / "name.wav")
    s = load_scenario(
        write_scenario(
            """
            name: wav text
            variables: {name: Alex}
            steps:
              - say: {wav: prompts/name.wav, text: "My name is {name}."}
              - say: {wav: prompts/name.wav}
            """
        )
    )
    with_text, without = s.steps
    assert with_text.say_text == "My name is Alex."
    assert with_text.caller_text == "My name is Alex."
    assert without.say_text is None
    assert without.caller_text == "[name.wav]"


def test_caller_text_of_a_text_say_and_of_an_expect_only_step(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: texts
            steps:
              - expect: {contains: hello}
              - say: hi there
            """
        )
    )
    assert [step.caller_text for step in s.steps] == [None, "hi there"]


def test_contains_and_not_contains_take_a_list(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: lists
            steps:
              - expect: {contains: [table, seven], not_contains: [sorry, closed]}
            """
        )
    )
    assert s.steps[0].expect == Expect(
        contains=("table", "seven"), not_contains=("sorry", "closed")
    )


def test_a_goto_loop_is_allowed(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: loop
            steps:
              - {label: a, say: hi, expect: {contains: x}, on_fail: goto b}
              - {label: b, say: again, expect: {contains: x}, on_fail: goto a}
            """
        )
    )
    assert [st.on_fail for st in s.steps] == ["b", "a"]


def test_variables_fill_say_and_every_expect_string(write_scenario):
    s = load_scenario(
        write_scenario(
            """
            name: vars
            variables: {who: Alex, n: 4, price: 9.5}
            steps:
              - say: "{who} for {n}, {{literal}}"
                expect:
                  contains: "{who}"
                  any_of: ["for {n}", "x"]
                  not_contains: "not {who}"
                  regex: "{who}"
                  fuzzy: {text: "costs {price}"}
            """
        )
    )
    step = s.steps[0]
    assert step.say_text == "Alex for 4, {literal}"
    assert step.expect == Expect(
        contains=("Alex",),
        any_of=("for 4", "x"),
        not_contains=("not Alex",),
        regex="Alex",
        fuzzy=Fuzzy(text="costs 9.5", min=0.8),
    )
    assert s.variables == {"who": "Alex", "n": "4", "price": "9.5"}


def test_variable_values_are_literal_inside_a_regex(write_scenario):
    path = write_scenario(
        """
        name: regex vars
        variables: {name: J.R.}
        steps:
          - expect: {regex: "thanks {name}$"}
        """
    )
    regex = load_scenario(path).steps[0].expect.regex
    assert regex == r"thanks J\.R\.$"
    assert re.search(regex, "thanks J.R.")
    assert not re.search(regex, "thanks JXRX")


def test_overrides_win_and_can_add_variables(write_scenario):
    path = write_scenario(
        """
        name: overrides
        variables: {name: Alex}
        steps:
          - say: "{name} {extra}"
        """
    )
    s = load_scenario(path, overrides={"name": "Sam", "extra": "plus"})
    assert s.steps[0].say_text == "Sam plus"
    assert s.variables == {"name": "Sam", "extra": "plus"}
    assert check_scenario(path, overrides={"extra": "plus"}) == []


def test_absolute_wav_path_is_kept(write_scenario, silent_wav, tmp_path):
    wav = silent_wav(tmp_path / "elsewhere" / "x.wav").resolve()
    path = write_scenario(
        f"""
        name: abs
        steps:
          - say: {{wav: "{wav.as_posix()}"}}
        """
    )
    assert load_scenario(path).steps[0].say_wav == wav


def test_relative_wav_resolves_against_the_scenario_folder_not_the_cwd(
    write_scenario, silent_wav, tmp_path, monkeypatch
):
    silent_wav(tmp_path / "sub" / "prompts" / "hi.wav")
    (tmp_path / "sub").mkdir(exist_ok=True)
    path = write_scenario(
        """
        name: rel
        steps:
          - say: {wav: prompts/hi.wav}
        """,
        name="sub/scenario.yaml",
    )
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert (
        load_scenario(path).steps[0].say_wav == (tmp_path / "sub" / "prompts" / "hi.wav").resolve()
    )


def test_load_error_lists_every_problem_with_the_path(write_scenario):
    path = write_scenario(
        """
        name: two problems
        steps:
          - {label: a, on_fail: goto nowhere, expect: {contains: x}}
          - {label: b}
        """
    )
    with pytest.raises(ScenarioError) as caught:
        load_scenario(path)
    assert caught.value.problems == [
        "step 1 (a): on_fail: no step labelled 'nowhere'",
        "step 2 (b): needs say or expect",
    ]
    assert str(caught.value) == (
        f"{path}: invalid scenario:\n"
        "  step 1 (a): on_fail: no step labelled 'nowhere'\n"
        "  step 2 (b): needs say or expect"
    )
    assert check_scenario(path) == caught.value.problems
