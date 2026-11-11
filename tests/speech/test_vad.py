import logging
import sys
import threading
import types

import numpy as np
import pytest

from voiceqa.audio.dsp import generate_tone
from voiceqa.speech import model_cache, vad
from voiceqa.speech.model_cache import STT_EXTRA_HINT, SttExtraMissingError
from voiceqa.speech.vad import (
    SPEECH_START,
    TURN_END,
    EnergyVad,
    SileroModelError,
    ThreadedTurnDetector,
    TurnDetector,
    TurnEvent,
    check_silero_session,
    create_vad,
    load_silero_session,
)

RATE = 8000
FRAME = 160
FRAME_S = FRAME / RATE
# A media_clock-like start time, so a bug that reports times from zero shows.
T0 = 5000.0


def tone_frames(segments: list[tuple[str, int]], seed: int = 0) -> list[np.ndarray]:
    """20 ms frames of a 1 kHz tone ("tone") or low noise ("gap"), by duration in ms."""
    rng = np.random.default_rng(seed)
    frames = []
    for kind, ms in segments:
        n = ms * RATE // 1000
        if kind == "tone":
            audio = generate_tone(1000, n / RATE, RATE, 0.3)
        else:
            audio = rng.integers(-30, 31, n).astype(np.int16)
        frames += [audio[i : i + FRAME] for i in range(0, n, FRAME)]
    return frames


def run_detector(detector: TurnDetector, frames: list[np.ndarray]) -> list[TurnEvent]:
    events = []
    for i, frame in enumerate(frames):
        events += detector.feed(T0 + i * FRAME_S, frame)
    return events


class ScriptedVad:
    """Returns the given probabilities in order, ignoring the audio."""

    name = "scripted"

    def __init__(self, probs):
        self._probs = iter(probs)

    def process(self, frame):
        return next(self._probs)

    def reset(self):
        pass


def frames_of(probs):
    return [np.zeros(FRAME, dtype=np.int16) for _ in probs], ScriptedVad(probs)


def assert_events(events, expected):
    assert [e.kind for e in events] == [k for k, _ in expected], events
    for event, (_, at) in zip(events, expected, strict=True):
        assert event.at == pytest.approx(at, abs=1e-9), (event, at)


# Energy VAD


def test_energy_vad_tone_is_speech_and_low_noise_is_not():
    v = EnergyVad()
    probs = [v.process(f) for f in tone_frames([("gap", 200), ("tone", 200), ("gap", 200)])]
    assert probs == [0.0] * 10 + [1.0] * 10 + [0.0] * 10


def test_energy_vad_ignores_sound_below_the_absolute_threshold():
    v = EnergyVad(threshold_dbfs=-20.0)
    frames = tone_frames([("gap", 100)]) + [generate_tone(1000, 0.02, RATE, 0.05)] * 5
    assert [v.process(f) for f in frames] == [0.0] * 10


def test_energy_vad_needs_the_margin_above_a_steady_noise_floor():
    rng = np.random.default_rng(1)
    v = EnergyVad(threshold_dbfs=-60.0, margin_db=10.0)
    # About -35 dBFS: above the absolute threshold, but it is the floor.
    noise = [rng.normal(0, 580, FRAME).astype(np.int16) for _ in range(50)]
    assert [v.process(f) for f in noise] == [0.0] * 50
    # 5 dB above the floor is still noise, 20 dB above is speech.
    assert v.process((noise[0] * 1.78).astype(np.int16)) == 0.0
    assert v.process(generate_tone(1000, 0.02, RATE, 0.3)) == 1.0


def test_energy_vad_digital_silence_does_not_pull_the_floor_down():
    rng = np.random.default_rng(2)
    v = EnergyVad(threshold_dbfs=-60.0, margin_db=10.0)
    noise = [rng.normal(0, 580, FRAME).astype(np.int16) for _ in range(20)]
    for f in noise:
        v.process(f)
    v.process(np.zeros(FRAME, dtype=np.int16))
    assert v.process(noise[0]) == 0.0


def test_energy_vad_first_sound_sets_the_floor_and_reset_forgets_it():
    v = EnergyVad()
    assert v.noise_floor_db is None
    v.process(np.zeros(FRAME, dtype=np.int16))
    assert v.noise_floor_db is None
    v.process(generate_tone(1000, 0.02, RATE, 0.3))
    assert v.noise_floor_db == pytest.approx(-13.5, abs=0.1)
    v.reset()
    assert v.noise_floor_db is None


def test_energy_vad_rejects_non_int16():
    with pytest.raises(TypeError, match="int16"):
        EnergyVad().process(np.zeros(FRAME, dtype=np.float32))


# Turn detection


def test_turn_detector_reports_first_and_last_speech_frame_times():
    probs = [0.0] * 20 + [0.9] * 30 + [0.1] * 40
    frames, scripted = frames_of(probs)
    events = run_detector(TurnDetector(scripted), frames)
    # speech_start is decided 100 ms after the first speech frame and
    # turn_end 700 ms after the last one; neither delay may show in `at`.
    assert_events(events, [(SPEECH_START, T0 + 20 * FRAME_S), (TURN_END, T0 + 50 * FRAME_S)])


def test_turn_detector_emits_when_the_decision_is_made():
    probs = [0.0] * 20 + [0.9] * 30 + [0.1] * 40
    frames, scripted = frames_of(probs)
    detector = TurnDetector(scripted)
    fed_at = {}
    for i, frame in enumerate(frames):
        for e in detector.feed(T0 + i * FRAME_S, frame):
            fed_at[e.kind] = i
    assert fed_at[SPEECH_START] == 24  # 5th speech frame completes 100 ms
    assert fed_at[TURN_END] == 84  # 35th silent frame completes 700 ms


def test_turn_detector_ignores_speech_shorter_than_min_speech():
    probs = [0.0] * 5 + [0.9] * 4 + [0.0] * 50
    frames, scripted = frames_of(probs)
    assert run_detector(TurnDetector(scripted), frames) == []


def test_turn_detector_min_speech_and_threshold_are_configurable():
    probs = [0.0] * 5 + [0.6] * 4 + [0.0] * 50
    frames, scripted = frames_of(probs)
    events = run_detector(TurnDetector(scripted, min_speech_ms=80), frames)
    assert_events(events, [(SPEECH_START, T0 + 5 * FRAME_S), (TURN_END, T0 + 9 * FRAME_S)])
    frames, scripted = frames_of(probs)
    assert run_detector(TurnDetector(scripted, threshold=0.7, min_speech_ms=80), frames) == []


def test_turn_detector_measures_silence_by_frame_times_not_frame_count():
    # Frames missing from the stream (dropped) still count as time passing.
    frames, scripted = frames_of([0.9] * 10 + [0.0] * 2)
    detector = TurnDetector(scripted)
    events = []
    for i in range(10):
        events += detector.feed(T0 + i * FRAME_S, frames[i])
    events += detector.feed(T0 + 30 * FRAME_S, frames[10])
    assert [e.kind for e in events] == [SPEECH_START]
    events += detector.feed(T0 + 44 * FRAME_S, frames[11])
    assert_events(events, [(SPEECH_START, T0), (TURN_END, T0 + 10 * FRAME_S)])


def test_turn_detector_reset_drops_a_turn_in_progress():
    frames, scripted = frames_of([0.9] * 10 + [0.0] * 40)
    detector = TurnDetector(scripted)
    run_detector(detector, frames[:10])
    assert detector.in_turn
    detector.reset()
    assert not detector.in_turn
    assert run_detector(detector, frames[10:]) == []


def test_turn_detector_rejects_bad_settings():
    with pytest.raises(ValueError, match="end_silence_ms"):
        TurnDetector(EnergyVad(), end_silence_ms=0)
    with pytest.raises(ValueError, match="min_speech_ms"):
        TurnDetector(EnergyVad(), min_speech_ms=-1)


BURSTS = [
    ("gap", 400),
    ("tone", 600),
    ("gap", 500),
    ("tone", 400),
    ("gap", 900),
    ("tone", 300),
    ("gap", 1000),
]


def test_energy_turns_on_tone_bursts():
    events = run_detector(TurnDetector(EnergyVad()), tone_frames(BURSTS))
    # The 500 ms gap stays inside the first turn, the 900 ms gap ends it.
    assert_events(
        events,
        [
            (SPEECH_START, T0 + 0.4),
            (TURN_END, T0 + 1.9),
            (SPEECH_START, T0 + 2.8),
            (TURN_END, T0 + 3.1),
        ],
    )


def test_energy_turns_with_a_shorter_end_silence_split_at_the_500_ms_gap():
    events = run_detector(TurnDetector(EnergyVad(), end_silence_ms=400), tone_frames(BURSTS))
    assert [round(e.at - T0, 6) for e in events] == [0.4, 1.0, 1.5, 1.9, 2.8, 3.1]


# Threaded detector


def test_threaded_detector_gives_the_same_events_from_its_own_thread():
    frames = tone_frames(BURSTS)
    expected = run_detector(TurnDetector(EnergyVad()), frames)
    got, threads = [], set()

    def on_event(event):
        got.append(event)
        threads.add(threading.get_ident())

    with ThreadedTurnDetector(TurnDetector(EnergyVad()), on_event, max_frames=1000) as td:
        for i, frame in enumerate(frames):
            assert td.feed(T0 + i * FRAME_S, frame)
    assert got == expected
    assert threads and threading.get_ident() not in threads
    assert td.dropped_frames == 0


class BlockingVad:
    name = "blocking"

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def process(self, frame):
        self.entered.set()
        assert self.release.wait(10)
        return 0.0

    def reset(self):
        pass


def test_threaded_detector_counts_frames_dropped_on_a_full_queue():
    blocking = BlockingVad()
    td = ThreadedTurnDetector(TurnDetector(blocking), lambda e: None, max_frames=4)
    td.start()
    frame = np.zeros(FRAME, dtype=np.int16)
    try:
        assert td.feed(T0, frame)
        assert blocking.entered.wait(10)
        assert all(td.feed(T0, frame) for _ in range(4))
        assert [td.feed(T0, frame) for _ in range(3)] == [False] * 3
        assert td.dropped_frames == 3
    finally:
        blocking.release.set()
        td.close()


def test_threaded_detector_copies_frames():
    seen = []

    class Recording(ScriptedVad):
        def process(self, frame):
            seen.append(frame.copy())
            return 0.0

    frame = np.full(FRAME, 7, dtype=np.int16)
    with ThreadedTurnDetector(TurnDetector(Recording([])), lambda e: None) as td:
        td.feed(T0, frame)
        frame[:] = 0
    assert (seen[0] == 7).all()


def test_threaded_detector_close_raises_the_worker_error():
    class Broken(ScriptedVad):
        def process(self, frame):
            raise RuntimeError("model broke")

    td = ThreadedTurnDetector(TurnDetector(Broken([])), lambda e: None)
    td.start()
    td.feed(T0, np.zeros(FRAME, dtype=np.int16))
    with pytest.raises(RuntimeError, match="model broke"):
        td.close()


def test_threaded_detector_ignores_frames_after_close():
    td = ThreadedTurnDetector(TurnDetector(EnergyVad()), lambda e: None)
    td.start()
    td.close()
    assert td.feed(T0, np.zeros(FRAME, dtype=np.int16)) is False
    assert td.dropped_frames == 0


# Registry


def test_create_vad_energy_takes_its_options():
    v = create_vad("energy", threshold_dbfs=-30.0)
    assert isinstance(v, EnergyVad)
    assert v.name == "energy"
    assert v.threshold_dbfs == -30.0


def test_create_vad_unknown_name():
    with pytest.raises(ValueError, match="auto, energy, silero"):
        create_vad("webrtc")


def test_create_vad_silero_without_the_extra_names_the_install_hint(monkeypatch):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    with pytest.raises(SttExtraMissingError) as info:
        create_vad("silero")
    assert STT_EXTRA_HINT in str(info.value)


def test_create_vad_auto_without_the_extra_falls_back_to_energy(monkeypatch, caplog):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    with caplog.at_level(logging.INFO, logger="voiceqa.speech.vad"):
        v = create_vad("auto")
    assert isinstance(v, EnergyVad)
    assert caplog.messages == [vad.ENERGY_FALLBACK_NOTICE]
    assert STT_EXTRA_HINT in vad.ENERGY_FALLBACK_NOTICE
    assert "\n" not in vad.ENERGY_FALLBACK_NOTICE


def test_create_vad_auto_with_the_extra_is_silero(monkeypatch):
    class FakeSilero:
        name = "silero"

    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(vad, "SileroVad", FakeSilero)
    assert isinstance(create_vad("auto"), FakeSilero)
    assert vad.auto_vad_name() == "silero"
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    assert vad.auto_vad_name() == "energy"


def test_create_vad_auto_takes_no_options():
    with pytest.raises(TypeError, match="auto"):
        create_vad("auto", threshold_dbfs=-30.0)


# Silero model check, with stand-in sessions


class FakeArg:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class FakeSession:
    def __init__(self, inputs=None, outputs=None):
        self._inputs = inputs or [
            FakeArg("input", ["seq_len", 576]),
            FakeArg("h", [1, 1, 128]),
            FakeArg("c", [1, 1, 128]),
        ]
        self._outputs = outputs or [
            FakeArg("speech_probs", ["n"]),
            FakeArg("hn", [1, 1, 128]),
            FakeArg("cn", [1, 1, 128]),
        ]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def fake_faster_whisper(monkeypatch, **vad_attrs):
    package = types.ModuleType("faster_whisper")
    vad_module = types.ModuleType("faster_whisper.vad")
    for name, value in vad_attrs.items():
        setattr(vad_module, name, value)
    package.vad = vad_module
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad_module)
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)


def test_load_silero_session_when_get_vad_model_is_gone(monkeypatch):
    fake_faster_whisper(monkeypatch)
    with pytest.raises(SileroModelError, match="get_vad_model cannot be imported"):
        load_silero_session()


def test_load_silero_session_when_the_model_has_no_session(monkeypatch):
    fake_faster_whisper(monkeypatch, get_vad_model=lambda: object())
    with pytest.raises(SileroModelError, match="without .session"):
        load_silero_session()


def test_load_silero_session_when_the_model_fails_to_load(monkeypatch):
    def broken():
        raise FileNotFoundError("silero_vad_v6.onnx")

    fake_faster_whisper(monkeypatch, get_vad_model=broken)
    with pytest.raises(SileroModelError, match="silero_vad_v6.onnx"):
        load_silero_session()


def test_load_silero_session_checks_the_session(monkeypatch):
    model = types.SimpleNamespace(session=FakeSession(outputs=[FakeArg("output", [1])]))
    fake_faster_whisper(monkeypatch, get_vad_model=lambda: model)
    with pytest.raises(SileroModelError, match="outputs"):
        load_silero_session()
    model.session = FakeSession()
    assert load_silero_session() is model.session


def test_check_silero_session_accepts_the_expected_model():
    check_silero_session(FakeSession())


@pytest.mark.parametrize(
    ("session", "detail"),
    [
        (
            FakeSession(inputs=[FakeArg("input", ["seq_len", 576]), FakeArg("state", [2, 1, 128])]),
            "inputs",
        ),
        (
            FakeSession(
                inputs=[
                    FakeArg("input", ["seq_len", 512]),
                    FakeArg("h", [1, 1, 128]),
                    FakeArg("c", [1, 1, 128]),
                ]
            ),
            "input",
        ),
        (FakeSession(outputs=[FakeArg("output", [1])]), "outputs"),
    ],
)
def test_check_silero_session_says_what_changed_and_what_to_check(session, detail):
    with pytest.raises(SileroModelError) as info:
        check_silero_session(session)
    message = str(info.value)
    assert "faster-whisper" in message
    assert "changed its Silero VAD model" in message
    assert detail in message
    assert "faster_whisper/vad.py" in message
    assert "voiceqa/speech/vad.py" in message
