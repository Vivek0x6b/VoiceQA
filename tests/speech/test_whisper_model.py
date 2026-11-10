"""Real tiny.en runs against the LibriVox fixture.

Skipped, with the reason shown, when the [stt] extra is missing or tiny.en
is not in the models dir. CI pulls it before the tests run.
"""

import pytest

from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.wer import normalize_words, wer

# Measured on Windows (faster-whisper 1.2.1): whole clip 0.128, utterances
# 0.429, 0.0, 0.067, 0.071 (mean 0.142). The first utterance is the hard
# one: tiny.en hears "Stella, the in slender" for "Della, being slender".
# The bounds leave room for differences between CPUs and library builds.
MAX_CLIP_WER = 0.25
MAX_UTTERANCE_WER = 0.6
MAX_MEAN_UTTERANCE_WER = 0.25

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def stt(test_model):
    from voiceqa.speech.whisper import FasterWhisperStt

    # CTranslate2 keeps its intra-op thread pool alive for the life of the
    # process; there is no API that releases it once this module's tests are
    # done. cpu_threads=1 keeps that permanent footprint small instead of
    # defaulting to one thread per core, since these tests only check
    # accuracy, not speed.
    return FasterWhisperStt(model=test_model, cpu_threads=1)


@pytest.fixture(scope="module")
def clip():
    return load_bench_clip(default_clip_dir())


def test_whole_clip_wer_is_within_bound(stt, clip):
    transcript = stt.transcribe(clip.samples, clip.rate)
    reference = " ".join(u.text for u in clip.utterances)
    hypothesis = " ".join(s.text for s in transcript.segments)
    error_rate = wer(reference, hypothesis)
    assert error_rate <= MAX_CLIP_WER, f"WER {error_rate:.3f}: {hypothesis!r}"


def test_each_utterance_wer_is_within_bound(stt, clip):
    wers = []
    for u in clip.utterances:
        part = clip.samples[int(u.start * clip.rate) : int(u.end * clip.rate)]
        transcript = stt.transcribe(part, clip.rate)
        hypothesis = " ".join(s.text for s in transcript.segments)
        error_rate = wer(u.text, hypothesis)
        assert error_rate <= MAX_UTTERANCE_WER, (
            f"WER {error_rate:.3f} for {u.text!r}: {hypothesis!r}"
        )
        wers.append(error_rate)
    assert sum(wers) / len(wers) <= MAX_MEAN_UTTERANCE_WER, wers


def test_transcript_metadata_and_word_times(stt, clip):
    transcript = stt.transcribe(clip.samples, clip.rate)
    duration = len(clip.samples) / clip.rate

    assert transcript.engine == "faster-whisper"
    assert transcript.model == "tiny.en"
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
        vocabulary = set(normalize_words(u.text))
        inside = [
            t for w in words if u.start - 0.5 <= w.start <= u.end for t in normalize_words(w.text)
        ]
        assert inside, u
        own = sum(t in vocabulary for t in inside)
        assert own / len(inside) >= 0.5, (u.text, inside)
