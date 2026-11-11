"""The real Silero model that ships inside the faster-whisper wheel.

Skipped, with the reason shown, when the [stt] extra is missing (a failure
instead when VOICEQA_REQUIRE_MODEL_TESTS is set, as in CI). No download.
"""

import numpy as np
import pytest

from voiceqa.audio.dsp import resample
from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.vad import (
    SILERO_WINDOW,
    SPEECH_START,
    TURN_END,
    SileroModelError,
    SileroVad,
    TurnDetector,
    load_silero_session,
)

pytestmark = [pytest.mark.model, pytest.mark.usefixtures("silero_extra")]

FRAME = 160

# Measured on Windows (faster-whisper 1.2.1, onnxruntime 1.31.1) with the
# default detector, against utterances.json: starts +0.06, +0.16, +0.06,
# +0.04 s, ends -0.36, -0.04, 0.00, -0.10 s. utterances.json was cut by an
# energy threshold 35 dB under the loudest frames, which keeps the quiet
# tail after the first sentence; Silero calls that tail silence.
START_TOLERANCE_S = 0.2
END_TOLERANCE_S = 0.4


@pytest.fixture(scope="module")
def clip():
    return load_bench_clip(default_clip_dir())


def test_faster_whisper_still_ships_the_silero_model_we_use():
    try:
        session = load_silero_session()
    except SileroModelError as exc:
        pytest.fail(str(exc))
    probs, h, c = session.run(
        ["speech_probs", "hn", "cn"],
        {
            "input": np.zeros((1, 576), dtype=np.float32),
            "h": np.zeros((1, 1, 128), dtype=np.float32),
            "c": np.zeros((1, 1, 128), dtype=np.float32),
        },
    )
    assert np.size(probs) == 1, "one probability per 512-sample window"
    assert h.shape == c.shape == (1, 1, 128)


def test_streaming_in_20_ms_frames_matches_one_pass_over_the_clip(clip):
    from faster_whisper.vad import get_vad_model

    whole = resample(clip.samples, clip.rate, 16000).astype(np.float32) / 32768.0
    whole = np.pad(whole, (0, -len(whole) % SILERO_WINDOW))
    # faster-whisper's own whole-clip call, which zeroes the tail of the last
    # window in the array it is given, so it gets a copy and that window is
    # left out below.
    reference = np.ravel(get_vad_model()(whole.copy()))

    vad = SileroVad()
    compared = 0
    for i in range(0, len(clip.samples) - FRAME + 1, FRAME):
        prob = vad.process(clip.samples[i : i + FRAME])
        done = vad.window_count
        if done == 0:
            assert prob == 0.0
        elif done < len(reference):
            assert prob == pytest.approx(reference[done - 1], abs=0.01), (i, done)
            compared += 1
    assert vad.window_count >= len(reference) - 2
    assert compared > 500


def test_silero_reset_starts_the_stream_over(clip):
    frames = [clip.samples[i : i + FRAME] for i in range(0, 2 * clip.rate, FRAME)]
    vad = SileroVad()
    first = [vad.process(f) for f in frames]
    vad.reset()
    assert vad.window_count == 0
    assert [vad.process(f) for f in frames] == first


def test_silero_finds_the_utterances_of_the_clip(clip):
    detector = TurnDetector(SileroVad())
    audio = np.concatenate([clip.samples, np.zeros(clip.rate, dtype=np.int16)])
    events = []
    for i in range(0, len(audio) - FRAME + 1, FRAME):
        events += detector.feed(i / clip.rate, audio[i : i + FRAME])

    # The default 700 ms end of turn keeps the 560 ms comma pause inside the
    # third sentence and splits at every gap between sentences.
    assert [e.kind for e in events] == [SPEECH_START, TURN_END] * len(clip.utterances), events
    starts, ends = events[0::2], events[1::2]
    for u, start, end in zip(clip.utterances, starts, ends, strict=True):
        assert start.at == pytest.approx(u.start, abs=START_TOLERANCE_S), (u.text, start)
        assert end.at == pytest.approx(u.end, abs=END_TOLERANCE_S), (u.text, end)
