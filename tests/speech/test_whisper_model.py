"""Real tiny.en runs against the LibriVox fixture.

Skipped, with the reason shown, when the [stt] extra is missing or tiny.en
is not in the models dir. CI pulls it before the tests run.
"""

import os
import re

import pytest

from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.model_cache import find_model, resolve_models_dir, stt_extra_installed

MODEL = "tiny.en"

# Measured on Windows (faster-whisper 1.2.1): whole clip 0.128, utterances
# 0.429, 0.0, 0.067, 0.071 (mean 0.142). The first utterance is the hard
# one: tiny.en hears "Stella, the in slender" for "Della, being slender".
# The bounds leave room for differences between CPUs and library builds.
MAX_CLIP_WER = 0.25
MAX_UTTERANCE_WER = 0.6
MAX_MEAN_UTTERANCE_WER = 0.25

pytestmark = pytest.mark.model


def _skip_reason() -> str | None:
    if not stt_extra_installed():
        return "needs the [stt] extra: pip install voiceqa[stt]"
    if find_model(MODEL, resolve_models_dir()) is None:
        return f"{MODEL} is not in {resolve_models_dir()}; run: voiceqa models pull stt {MODEL}"
    return None


@pytest.fixture(scope="module")
def stt():
    reason = _skip_reason()
    if reason:
        # CI sets this so a missing model fails loudly instead of skipping.
        if os.environ.get("VOICEQA_REQUIRE_MODEL_TESTS"):
            pytest.fail(reason)
        pytest.skip(reason)
    from voiceqa.speech.whisper import FasterWhisperStt

    # CTranslate2 keeps its intra-op thread pool alive for the life of the
    # process; there is no API that releases it once this module's tests are
    # done. cpu_threads=1 keeps that permanent footprint small instead of
    # defaulting to one thread per core, since these tests only check
    # accuracy, not speed.
    return FasterWhisperStt(model=MODEL, cpu_threads=1)


@pytest.fixture(scope="module")
def clip():
    return load_bench_clip(default_clip_dir())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _wer(reference: str, hypothesis: str) -> float:
    ref, hyp = _words(reference), _words(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1] / len(ref)


def test_private_wer_helper_counts_edits_per_reference_word():
    assert _wer("the cat sat", "the cat sat") == 0.0
    assert _wer("the cat sat", "the bat sat") == pytest.approx(1 / 3)
    assert _wer("the cat sat", "the cat") == pytest.approx(1 / 3)
    assert _wer("the cat", "the cat sat down") == pytest.approx(1.0)
    assert _wer("Della, being slender.", "della being slender") == 0.0


def test_whole_clip_wer_is_within_bound(stt, clip):
    transcript = stt.transcribe(clip.samples, clip.rate)
    reference = " ".join(u.text for u in clip.utterances)
    hypothesis = " ".join(s.text for s in transcript.segments)
    wer = _wer(reference, hypothesis)
    assert wer <= MAX_CLIP_WER, f"WER {wer:.3f}: {hypothesis!r}"


def test_each_utterance_wer_is_within_bound(stt, clip):
    wers = []
    for u in clip.utterances:
        part = clip.samples[int(u.start * clip.rate) : int(u.end * clip.rate)]
        transcript = stt.transcribe(part, clip.rate)
        hypothesis = " ".join(s.text for s in transcript.segments)
        wer = _wer(u.text, hypothesis)
        assert wer <= MAX_UTTERANCE_WER, f"WER {wer:.3f} for {u.text!r}: {hypothesis!r}"
        wers.append(wer)
    assert sum(wers) / len(wers) <= MAX_MEAN_UTTERANCE_WER, wers


def test_transcript_metadata_and_word_times(stt, clip):
    transcript = stt.transcribe(clip.samples, clip.rate)
    duration = len(clip.samples) / clip.rate

    assert transcript.engine == "faster-whisper"
    assert transcript.model == MODEL
    assert transcript.language == "en"
    words = [w for s in transcript.segments for w in s.words]
    assert words, "word timestamps are on by default"
    for w in words:
        assert 0.0 <= w.start <= w.end <= duration + 0.5
        assert 0.0 <= w.probability <= 1.0


def test_words_fall_in_their_utterance(stt, clip):
    """Words timed inside a reference utterance's span (0.5 s slack) are
    mostly that utterance's own words, so the word timestamps line up with
    the fixture's boundaries."""
    transcript = stt.transcribe(clip.samples, clip.rate)
    words = [w for s in transcript.segments for w in s.words]
    for u in clip.utterances:
        vocabulary = set(_words(u.text))
        inside = [t for w in words if u.start - 0.5 <= w.start <= u.end for t in _words(w.text)]
        assert inside, u
        own = sum(t in vocabulary for t in inside)
        assert own / len(inside) >= 0.5, (u.text, inside)
