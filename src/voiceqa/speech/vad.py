"""Voice activity detection and end-of-turn detection on 8 kHz call audio.

A `Vad` takes int16 mono frames at 8 kHz (the 160-sample, 20 ms frames of
the media interface) and returns a speech probability per frame. Engines:

- `EnergyVad`: RMS level against a threshold, NumPy only. Cheap enough to
  run inline in a media callback.
- `SileroVad`: the Silero model that faster-whisper ships (needs the [stt]
  extra). Run it through `ThreadedTurnDetector`, never on the event loop.

`TurnDetector` turns probabilities into speech_start and turn_end events.
Event times are frame times on the caller's clock (in a call, media_clock):
the start of the first speech frame and the end of the last one, never the
moment the decision was made.
"""

import importlib.metadata
import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from voiceqa.audio.dsp import StreamResampler

from . import model_cache
from .model_cache import STT_EXTRA_HINT, SttExtraMissingError

logger = logging.getLogger(__name__)

RATE = 8000
SILERO_RATE = 16000
SILERO_WINDOW = 512
SILERO_CONTEXT = 64
_SILERO_STATE_SHAPE = [1, 1, 128]
_SILERO_INPUTS = ["input", "h", "c"]
_SILERO_OUTPUTS = ["speech_probs", "hn", "cn"]

SPEECH_START = "speech_start"
TURN_END = "turn_end"

ENERGY_FALLBACK_NOTICE = (
    f"using the energy VAD, the silero VAD needs the [stt] extra: {STT_EXTRA_HINT}"
)

# Frames quieter than this are digital silence (zeros, or nearly), not noise.
_SILENT_DBFS = -90.0
# Frame times are sums of floats; this keeps 700 ms of silence from reading
# as 699.9999999 ms.
_TIME_EPS = 1e-6


class SileroModelError(RuntimeError):
    pass


class Vad(Protocol):
    name: str

    def process(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


def _check_frame(frame: np.ndarray) -> None:
    if frame.dtype != np.int16 or frame.ndim != 1:
        raise TypeError(f"VAD frames must be 1-D int16, got {frame.dtype} {frame.shape}")


class EnergyVad:
    """Speech is a frame whose RMS level is at least `threshold_dbfs` and at
    least `margin_db` above the noise floor, giving 1.0 or 0.0.

    The noise floor starts at the first frame that is not digital silence,
    follows quieter frames down at once and rises at most
    `floor_rise_db_per_s`, so steady noise becomes the floor within a few
    seconds while speech, which keeps dipping between words, does not.
    """

    name = "energy"

    def __init__(
        self,
        threshold_dbfs: float = -45.0,
        margin_db: float = 10.0,
        floor_rise_db_per_s: float = 2.0,
    ) -> None:
        self.threshold_dbfs = threshold_dbfs
        self.margin_db = margin_db
        self.floor_rise_db_per_s = floor_rise_db_per_s
        self._floor: float | None = None

    @property
    def noise_floor_db(self) -> float | None:
        return self._floor

    def reset(self) -> None:
        self._floor = None

    def process(self, frame: np.ndarray) -> float:
        _check_frame(frame)
        if len(frame) == 0:
            return 0.0
        rms = math.sqrt(float(np.mean(frame.astype(np.float64) ** 2))) / 32768.0
        level = 20.0 * math.log10(max(rms, 1e-5))
        if level > _SILENT_DBFS:
            if self._floor is None or level < self._floor:
                self._floor = level
            else:
                rise = self.floor_rise_db_per_s * len(frame) / RATE
                self._floor = min(level, self._floor + rise)
        if self._floor is None:
            return 0.0
        speech = level >= max(self.threshold_dbfs, self._floor + self.margin_db)
        return 1.0 if speech else 0.0


def _faster_whisper_version() -> str:
    try:
        return importlib.metadata.version("faster-whisper")
    except importlib.metadata.PackageNotFoundError:
        return "(version unknown)"


def _model_changed(detail: str) -> SileroModelError:
    return SileroModelError(
        f"faster-whisper {_faster_whisper_version()} moved or changed its Silero VAD model: "
        f"{detail}. VoiceQA expects faster_whisper.vad.get_vad_model() to return an object "
        "whose .session is an onnxruntime session with inputs input [seq_len, 576], "
        "h [1, 1, 128] and c [1, 1, 128], and outputs speech_probs, hn and cn. "
        "Check faster_whisper/vad.py in this version and update voiceqa/speech/vad.py to match."
    )


def check_silero_session(session) -> None:
    inputs = {arg.name: list(arg.shape) for arg in session.get_inputs()}
    outputs = [arg.name for arg in session.get_outputs()]
    if sorted(inputs) != sorted(_SILERO_INPUTS):
        raise _model_changed(f"model inputs are {sorted(inputs)}")
    if inputs["input"][-1:] != [SILERO_WINDOW + SILERO_CONTEXT] or len(inputs["input"]) != 2:
        raise _model_changed(f"input shape is {inputs['input']}")
    for name in ("h", "c"):
        if inputs[name] != _SILERO_STATE_SHAPE:
            raise _model_changed(f"input {name} shape is {inputs[name]}")
    if sorted(outputs) != sorted(_SILERO_OUTPUTS):
        raise _model_changed(f"model outputs are {sorted(outputs)}")


def load_silero_session():
    """The onnxruntime session of faster-whisper's Silero model, checked.

    The model file ships inside the faster-whisper wheel, so this never
    downloads anything.
    """
    if not model_cache.stt_extra_installed():
        raise SttExtraMissingError(f"the silero VAD needs the [stt] extra: {STT_EXTRA_HINT}")
    try:
        from faster_whisper.vad import get_vad_model
    except ImportError as exc:
        raise _model_changed(
            f"faster_whisper.vad.get_vad_model cannot be imported ({exc})"
        ) from exc
    try:
        model = get_vad_model()
    except Exception as exc:
        raise _model_changed(f"get_vad_model() failed ({exc})") from exc
    session = getattr(model, "session", None)
    if session is None:
        raise _model_changed("get_vad_model() returned an object without .session")
    check_silero_session(session)
    return session


class SileroVad:
    """Silero VAD, streamed window by window.

    faster-whisper's own model call resets the LSTM state on every call, so
    this drives its onnxruntime session directly and carries h, c and the
    64-sample context between windows, the way faster-whisper builds them
    for a whole clip. Audio is upsampled to 16 kHz with StreamResampler.

    A window is 512 samples at 16 kHz (32 ms), which does not line up with
    20 ms frames: each frame gets the probability of the latest finished
    window, 0.0 before the first. That probability describes audio ending
    up to one window plus the resampler's 2 ms before the frame's end.
    """

    name = "silero"

    def __init__(self) -> None:
        self._session = load_silero_session()
        # The first run of a new session does one-off set-up. Done here,
        # before a call is dialed, instead of on the call's first window.
        zeros = np.zeros(_SILERO_STATE_SHAPE, dtype=np.float32)
        self._session.run(
            _SILERO_OUTPUTS,
            {
                "input": np.zeros((1, SILERO_WINDOW + SILERO_CONTEXT), dtype=np.float32),
                "h": zeros,
                "c": zeros,
            },
        )
        self.reset()

    def reset(self) -> None:
        self._resampler = StreamResampler(RATE, SILERO_RATE)
        self._pending = np.zeros(0, dtype=np.float32)
        self._context = np.zeros(SILERO_CONTEXT, dtype=np.float32)
        self._h = np.zeros(_SILERO_STATE_SHAPE, dtype=np.float32)
        self._c = np.zeros(_SILERO_STATE_SHAPE, dtype=np.float32)
        self._prob = 0.0
        self.window_count = 0

    def process(self, frame: np.ndarray) -> float:
        _check_frame(frame)
        audio = self._resampler.push(frame).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, audio])
        while len(self._pending) >= SILERO_WINDOW:
            window = self._pending[:SILERO_WINDOW]
            self._pending = self._pending[SILERO_WINDOW:]
            probs, self._h, self._c = self._session.run(
                _SILERO_OUTPUTS,
                {
                    "input": np.concatenate([self._context, window])[None, :],
                    "h": self._h,
                    "c": self._c,
                },
            )
            self._context = window[-SILERO_CONTEXT:]
            self._prob = float(np.ravel(probs)[0])
            self.window_count += 1
        return self._prob


@dataclass(frozen=True)
class TurnEvent:
    kind: str
    at: float


class TurnDetector:
    """Finds where turns start and end in a stream of timed frames.

    `feed(at, frame)` takes a frame and the time its first sample was
    played or received. A frame is speech when the VAD gives it at least
    `threshold`. speech_start comes once speech has run for `min_speech_ms`
    without a break, with `at` = the first of those frames' time. turn_end
    comes once `end_silence_ms` has passed since the end of the last speech
    frame, measured by frame times (so missing frames still count as time),
    with `at` = that end.
    """

    def __init__(
        self,
        vad: Vad,
        threshold: float = 0.5,
        min_speech_ms: float = 100,
        end_silence_ms: float = 700,
    ) -> None:
        if min_speech_ms < 0:
            raise ValueError(f"min_speech_ms must be 0 or more, got {min_speech_ms}")
        if end_silence_ms <= 0:
            raise ValueError(f"end_silence_ms must be positive, got {end_silence_ms}")
        self.vad = vad
        self.threshold = threshold
        self.min_speech_ms = min_speech_ms
        self.end_silence_ms = end_silence_ms
        self._in_turn = False
        self._run_start: float | None = None
        self._run_samples = 0
        self._last_speech_end = 0.0

    @property
    def in_turn(self) -> bool:
        return self._in_turn

    def reset(self) -> None:
        self.vad.reset()
        self._in_turn = False
        self._run_start = None
        self._run_samples = 0

    def feed(self, at: float, frame: np.ndarray) -> list[TurnEvent]:
        prob = self.vad.process(frame)
        end = at + len(frame) / RATE
        if prob >= self.threshold:
            self._last_speech_end = end
            if self._in_turn:
                return []
            if self._run_start is None:
                self._run_start = at
                self._run_samples = 0
            self._run_samples += len(frame)
            if self._run_samples * 1000 >= self.min_speech_ms * RATE:
                self._in_turn = True
                return [TurnEvent(SPEECH_START, self._run_start)]
            return []
        self._run_start = None
        if self._in_turn and end - self._last_speech_end >= self.end_silence_ms / 1000 - _TIME_EPS:
            self._in_turn = False
            return [TurnEvent(TURN_END, self._last_speech_end)]
        return []

    def finish(self) -> list[TurnEvent]:
        """End of stream: a turn still open (speaking, or inside the
        end-silence wait) ends at the end of its last speech frame."""
        self._run_start = None
        if not self._in_turn:
            return []
        self._in_turn = False
        return [TurnEvent(TURN_END, self._last_speech_end)]


_STOP = object()
_STOP_FINISH = object()


class ThreadedTurnDetector:
    """Runs a TurnDetector in its own thread, fed through a bounded queue.

    `feed()` never blocks: it copies the frame into the queue, or counts it
    in `dropped_frames` and returns False when the queue is full. Every
    frame `feed()` accepted is either processed or counted as dropped.
    `on_event` runs in the detector's thread; to hand events to asyncio,
    pass something like
    `lambda e: loop.call_soon_threadsafe(events.put_nowait, e)`.
    If the detector or `on_event` raises, `on_error` (if given) gets the
    error at once in the detector's thread, `error` holds it, later frames
    are counted as dropped and `close()` raises it.
    """

    def __init__(
        self,
        detector: TurnDetector,
        on_event: Callable[[TurnEvent], object],
        max_frames: int = 50,
        on_error: Callable[[BaseException], object] | None = None,
    ) -> None:
        self.detector = detector
        self._on_event = on_event
        self._on_error = on_error
        self._queue: queue.Queue = queue.Queue(maxsize=max_frames)
        self._thread = threading.Thread(target=self._run, name="voiceqa-vad", daemon=True)
        # Held only around the closed check and a put_nowait, never for a
        # blocking call, so feed() still never waits for long.
        self._lock = threading.Lock()
        self._closed = False
        self._stop_sent = False
        self._dropped = 0
        self._error: BaseException | None = None

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        self._thread.start()

    def feed(self, at: float, frame: np.ndarray) -> bool:
        item = (at, np.array(frame, dtype=np.int16, copy=True))
        with self._lock:
            if self._closed:
                return False
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self._dropped += 1
                return False
        return True

    def close(self, timeout: float = 5.0, finish: bool = False) -> None:
        """Processes the frames already queued, then stops the thread.

        With `finish`, a turn still open after the last frame gets its
        turn_end (see TurnDetector.finish) before the thread stops. Raises
        TimeoutError if the thread has not stopped within `timeout` seconds.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            self._closed = True
        if not self._stop_sent and self._thread.is_alive():
            try:
                self._queue.put(_STOP_FINISH if finish else _STOP, timeout=timeout)
            except queue.Full:
                raise TimeoutError(f"the VAD thread did not stop within {timeout} s") from None
            self._stop_sent = True
        if self._thread.ident is not None:
            self._thread.join(max(0.0, deadline - time.monotonic()))
            if self._thread.is_alive():
                raise TimeoutError(f"the VAD thread did not stop within {timeout} s")
        self._drop_leftovers()
        if self._error is not None:
            raise self._error

    def __enter__(self) -> "ThreadedTurnDetector":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _drop_leftovers(self) -> None:
        # Frames fed before a start() that never came.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(item, tuple):
                with self._lock:
                    self._dropped += 1

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP or item is _STOP_FINISH:
                if item is _STOP_FINISH and self._error is None:
                    self._handle(self.detector.finish)
                return
            if self._error is not None:
                with self._lock:
                    self._dropped += 1
                continue
            self._handle(self.detector.feed, *item)

    def _handle(self, step: Callable[..., list[TurnEvent]], *args: object) -> None:
        try:
            for event in step(*args):
                self._on_event(event)
        except Exception as exc:
            self._error = exc
            if self._on_error is not None:
                try:
                    self._on_error(exc)
                except Exception:
                    logger.exception("VAD on_error callback failed")


_VAD_NAMES = ("auto", "energy", "silero")


def auto_vad_name() -> str:
    """The VAD `create_vad("auto")` builds: silero with the [stt] extra, else energy."""
    return "silero" if model_cache.stt_extra_installed() else "energy"


def vad_notice(name: str) -> str | None:
    """The one-line notice a CLI should show for `name`: ENERGY_FALLBACK_NOTICE
    when "auto" falls back to energy, else None."""
    if name == "auto" and auto_vad_name() == "energy":
        return ENERGY_FALLBACK_NOTICE
    return None


def create_vad(name: str, **kwargs: object) -> Vad:
    """A VAD by name, from plain keyword arguments.

    "auto" takes no options. When it falls back to energy it logs
    ENERGY_FALLBACK_NOTICE at INFO, for a CLI to show.
    """
    if name == "energy":
        return EnergyVad(**kwargs)
    if name == "silero":
        return SileroVad(**kwargs)
    if name == "auto":
        if kwargs:
            raise TypeError("the auto VAD takes no options; name energy or silero to set them")
        if auto_vad_name() == "silero":
            return SileroVad()
        logger.info(ENERGY_FALLBACK_NOTICE)
        return EnergyVad()
    raise ValueError(f"unknown VAD {name!r}, known VADs: {', '.join(_VAD_NAMES)}")
