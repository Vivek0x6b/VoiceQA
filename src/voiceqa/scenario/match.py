"""Checking an agent's turn against a step's expect.

`contains`, `any_of`, `not_contains` and `fuzzy` compare words after the
WER helper's normalisation (case folded, punctuation dropped, hyphens split
words), so a phrase must appear as whole words in order: "name" is found in
"your name?" but not in "username". `regex` is searched in the raw
transcript. `fuzzy` scores 1 - WER of the expected text against every
window of the turn with the same number of words (the whole turn when it is
shorter) and keeps the best, from 0 (no word in common) to 1.
"""

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from voiceqa.scenario.schema import Expect
from voiceqa.speech.wer import edit_distance, normalize_words


class MatcherResult(BaseModel):
    """One matcher's verdict. `expected` is what the step asked for,
    `reason` says why it passed or failed. `score` is the fuzzy similarity,
    or the latency in ms for max_latency_ms."""

    model_config = ConfigDict(frozen=True)

    matcher: str
    passed: bool
    expected: str
    reason: str
    score: float | None = None


class MatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    heard: str
    results: tuple[MatcherResult, ...]

    @property
    def failures(self) -> tuple[MatcherResult, ...]:
        return tuple(r for r in self.results if not r.passed)

    def summary(self) -> str:
        """The failed matchers and why, on one line. Empty on a pass."""
        return "; ".join(f"{r.matcher}: {r.reason}" for r in self.failures)


def _has_phrase(words: Sequence[str], phrase: Sequence[str]) -> bool:
    n = len(phrase)
    return any(list(words[i : i + n]) == list(phrase) for i in range(len(words) - n + 1))


def fuzzy_similarity(expected: str, heard: str) -> tuple[float, str]:
    """Best 1 - WER of `expected` against a window of `heard`, and that
    window's normalised words joined by spaces."""
    want = normalize_words(expected)
    got = normalize_words(heard)
    n = len(want)
    if not n:
        raise ValueError("fuzzy text needs at least one word")
    starts = range(len(got) - n + 1) if len(got) > n else range(1)
    best, window = -1.0, ""
    for start in starts:
        candidate = got[start : start + n]
        # A window has at most n words, so at most n edits: score is 0 to 1.
        score = 1 - edit_distance(want, candidate) / n
        if score > best:
            best, window = score, " ".join(candidate)
    return best, window


def match_expect(expect: Expect, heard: str, latency_ms: float | None = None) -> MatchResult:
    """Check `heard` (the agent turn's transcript) against `expect`.
    `latency_ms` is the measured response latency, used by max_latency_ms."""
    words = normalize_words(heard)
    results: list[MatcherResult] = []

    for phrase in expect.contains:
        norm = normalize_words(phrase)
        found = _has_phrase(words, norm)
        shown = " ".join(norm)
        reason = f'heard "{shown}"' if found else f'did not hear "{shown}"'
        results.append(
            MatcherResult(matcher="contains", passed=found, expected=phrase, reason=reason)
        )

    if expect.any_of:
        hit = next(
            (p for p in expect.any_of if _has_phrase(words, normalize_words(p))),
            None,
        )
        reason = f'heard "{" ".join(normalize_words(hit))}"' if hit else "heard none of them"
        results.append(
            MatcherResult(
                matcher="any_of",
                passed=hit is not None,
                expected=" | ".join(f'"{p}"' for p in expect.any_of),
                reason=reason,
            )
        )

    for phrase in expect.not_contains:
        norm = normalize_words(phrase)
        found = _has_phrase(words, norm)
        shown = " ".join(norm)
        reason = f'heard "{shown}"' if found else f'did not hear "{shown}"'
        results.append(
            MatcherResult(matcher="not_contains", passed=not found, expected=phrase, reason=reason)
        )

    if expect.regex is not None:
        match = re.search(expect.regex, heard)
        reason = f'matched "{match.group(0)}"' if match else "no match in what was heard"
        results.append(
            MatcherResult(
                matcher="regex", passed=match is not None, expected=expect.regex, reason=reason
            )
        )

    if expect.fuzzy is not None:
        score, window = fuzzy_similarity(expect.fuzzy.text, heard)
        results.append(
            MatcherResult(
                matcher="fuzzy",
                passed=score >= expect.fuzzy.min,
                expected=expect.fuzzy.text,
                reason=f'best window "{window}" scores {score:.2f}, needs {expect.fuzzy.min:.2f}',
                score=score,
            )
        )

    if expect.max_latency_ms is not None:
        limit = expect.max_latency_ms
        if latency_ms is None:
            passed, reason = False, "no latency measured"
        elif latency_ms <= limit:
            passed, reason = True, f"latency {latency_ms:.0f} ms"
        else:
            passed, reason = False, f"latency {latency_ms:.0f} ms is over {limit:.0f} ms"
        results.append(
            MatcherResult(
                matcher="max_latency_ms",
                passed=passed,
                expected=f"at most {limit:.0f} ms",
                reason=reason,
                score=latency_ms,
            )
        )

    return MatchResult(passed=all(r.passed for r in results), heard=heard, results=tuple(results))
