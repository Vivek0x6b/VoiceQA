"""Speech to text in a separate worker process.

SttWorker starts a child process with the "spawn" start method on every OS,
builds the engine there from its registry name and plain keyword arguments
(see voiceqa.speech.registry), loads it once, and then transcribes jobs one
at a time. Speech work never runs in the process or event loop that handles
media, and whatever the engine leaves behind (CTranslate2 keeps its thread
pool for the life of the process) stays in the child.

Jobs go to the child over a multiprocessing queue as (job id, samples,
rate). Results come back over a pipe as plain data and are matched to the
job's future by a reader thread in this process, so nothing here blocks an
event loop. If the child dies, every pending job fails with SttWorkerError
instead of waiting forever.

    async with SttWorker("faster-whisper", model="base.en") as worker:
        transcript = await worker.transcribe(samples, 8000)

M3 reuses this for live per-turn transcription during calls.
"""

import asyncio
import itertools
import multiprocessing
import queue
import signal
import threading
from concurrent.futures import Future, InvalidStateError
from multiprocessing.connection import Connection, wait

import numpy as np

from .models import Transcript
from .registry import create_stt

DEFAULT_CLOSE_TIMEOUT = 10.0
# How long to wait after terminate() (and again after kill()) for the child
# to go.
_STOP_WAIT = 5.0
# How often an idle child checks that its parent is still there.
_PARENT_CHECK = 1.0


class SttWorkerError(RuntimeError):
    """A job could not be transcribed: the engine failed to load, the engine
    raised, or the worker process ended or was closed first."""


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


def _serve(engine: str, engine_kwargs: dict, jobs, results: Connection) -> None:
    # Ctrl+C reaches every process in the console group. The parent decides
    # when the worker stops, through close().
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        stt = create_stt(engine, **engine_kwargs)
    except Exception as exc:
        results.send(("failed", None, _describe(exc)))
        return
    results.send(("ready", None, None))
    parent = multiprocessing.parent_process()
    while True:
        try:
            job = jobs.get(timeout=_PARENT_CHECK)
        except queue.Empty:
            # A parent that died without close() never sends the sentinel.
            if parent is not None and not parent.is_alive():
                return
            continue
        if job is None:
            return
        job_id, samples, rate = job
        try:
            transcript = stt.transcribe(samples, rate)
        except Exception as exc:
            results.send(("error", job_id, _describe(exc)))
        else:
            results.send(("done", job_id, transcript.model_dump()))


class SttWorker:
    def __init__(self, engine: str, **engine_kwargs: object) -> None:
        self._engine = engine
        self._engine_kwargs = engine_kwargs
        self._process = None
        self._jobs = None
        self._results: Connection | None = None
        self._reader: threading.Thread | None = None
        self._ready: Future[None] = Future()
        self._pending: dict[int, Future[Transcript]] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count()
        self._load_error: SttWorkerError | None = None
        self._ended = False
        self._closed = False

    def start(self) -> "SttWorker":
        if self._process is not None:
            raise SttWorkerError("STT worker already started")
        ctx = multiprocessing.get_context("spawn")
        self._jobs = ctx.Queue()
        self._results, child_end = ctx.Pipe(duplex=False)
        self._process = ctx.Process(
            target=_serve,
            args=(self._engine, self._engine_kwargs, self._jobs, child_end),
            name=f"voiceqa-stt-{self._engine}",
            daemon=True,
        )
        self._process.start()
        # Without our copy of the child's end, the pipe reads as closed as
        # soon as the child exits.
        child_end.close()
        self._reader = threading.Thread(
            target=self._read_results, name="voiceqa-stt-results", daemon=True
        )
        self._reader.start()
        return self

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode if self._process else None

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def submit(self, samples: np.ndarray, rate: int) -> "Future[Transcript]":
        """Queue one job and return its future without waiting.

        int16 mono samples at `rate` Hz. The future fails with SttWorkerError
        if the engine raises or the worker ends before the job is done.
        """
        if samples.dtype != np.int16:
            raise TypeError(f"samples must be int16, got {samples.dtype}")
        if samples.ndim != 1:
            raise ValueError(f"samples must be mono (1-D), got shape {samples.shape}")
        if self._process is None:
            raise SttWorkerError("STT worker not started")
        future: Future[Transcript] = Future()
        with self._lock:
            if self._closed:
                raise SttWorkerError("STT worker is closed")
            if self._load_error is not None or self._ended:
                future.set_exception(self._load_error or self._ended_error())
                return future
            job_id = next(self._ids)
            self._pending[job_id] = future
        self._jobs.put((job_id, samples, rate))
        return future

    async def transcribe(self, samples: np.ndarray, rate: int) -> Transcript:
        return await asyncio.wrap_future(self.submit(samples, rate))

    async def wait_ready(self) -> None:
        """Wait until the engine is loaded in the child. Raises SttWorkerError
        if it could not be built (missing extra or model, unknown name)."""
        if self._process is None:
            raise SttWorkerError("STT worker not started")
        await asyncio.wrap_future(self._ready)

    def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        """Stop the child. Jobs already queued may finish within `timeout`
        seconds; after that the child is terminated and the jobs still
        pending fail with SttWorkerError. Safe to call more than once."""
        with self._lock:
            if self._process is None or self._closed:
                return
            self._closed = True
        process = self._process
        if process.is_alive():
            self._jobs.put(None)
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(_STOP_WAIT)
        if process.is_alive():
            process.kill()
            process.join(_STOP_WAIT)
        # Data still in the queue's pipe must not keep this process from
        # exiting later.
        self._jobs.cancel_join_thread()
        self._jobs.close()
        self._reader.join(_STOP_WAIT)
        self._results.close()

    async def aclose(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self.close, timeout)

    def __enter__(self) -> "SttWorker":
        return self.start()

    # Leaving the block on an error (Ctrl+C included) stops the child at
    # once instead of waiting for queued jobs nobody will read.
    def __exit__(self, exc_type: object, *exc: object) -> None:
        self.close(DEFAULT_CLOSE_TIMEOUT if exc_type is None else 0)

    async def __aenter__(self) -> "SttWorker":
        return self.start()

    async def __aexit__(self, exc_type: object, *exc: object) -> None:
        await self.aclose(DEFAULT_CLOSE_TIMEOUT if exc_type is None else 0)

    def _ended_error(self) -> SttWorkerError:
        if self._closed:
            return SttWorkerError("STT worker was closed before the job finished")
        return SttWorkerError("STT worker process ended unexpectedly before the job finished")

    def _read_results(self) -> None:
        conn = self._results
        try:
            while conn in wait([conn, self._process.sentinel]):
                try:
                    kind, job_id, payload = conn.recv()
                except (EOFError, OSError):
                    break
                self._handle(kind, job_id, payload)
        finally:
            with self._lock:
                self._ended = True
                error = self._load_error or self._ended_error()
                pending, self._pending = self._pending, {}
            _resolve(self._ready, exception=error)
            for future in pending.values():
                _resolve(future, exception=error)

    def _handle(self, kind: str, job_id: int | None, payload: object) -> None:
        if kind == "ready":
            _resolve(self._ready, result=None)
            return
        if kind == "failed":
            error = SttWorkerError(payload)
            with self._lock:
                self._load_error = error
            _resolve(self._ready, exception=error)
            return
        with self._lock:
            future = self._pending.pop(job_id, None)
        if future is None:
            return
        if kind == "done":
            _resolve(future, result=Transcript.model_validate(payload))
        else:
            _resolve(future, exception=SttWorkerError(payload))


def _resolve(future: Future, result: object = None, exception: BaseException | None = None):
    # A future whose caller gave up (an asyncio task cancelled while it
    # waited) is already cancelled; its result has nowhere to go.
    try:
        if exception is not None:
            future.set_exception(exception)
        else:
            future.set_result(result)
    except InvalidStateError:
        pass
