"""Speech to text in a separate worker process.

SttWorker starts a child process with the "spawn" start method on every OS,
builds the engine there from its registry name and plain keyword arguments
(see voiceqa.speech.registry), loads it once, and then transcribes jobs one
at a time. Speech work never runs in the process or event loop that handles
media, and whatever the engine leaves behind (CTranslate2 keeps its thread
pool for the life of the process) stays in the child. The child runs with
HF_HUB_OFFLINE=1, so nothing in it can fetch from the Hugging Face hub.

Jobs go to the child over a multiprocessing queue as (job id, samples,
rate). Results come back over a pipe as plain data and are matched to the
job's future by a reader thread in this process, so nothing here blocks an
event loop. If the child dies, every pending job fails with SttWorkerError
instead of waiting forever. A result that cannot be read fails only its
own job.

    async with SttWorker("faster-whisper", model="base.en") as worker:
        transcript = await worker.transcribe(samples, 8000)

`async with` (or start_async()) spawns the child from a thread, so the
event loop keeps running meanwhile. M3 reuses this for live per-turn
transcription during calls.
"""

import asyncio
import contextlib
import itertools
import multiprocessing
import os
import pickle
import queue
import signal
import sys
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

_console_lock = threading.Lock()


class SttWorkerError(RuntimeError):
    """A job could not be transcribed: the engine failed to load, the engine
    raised, or the worker process ended or was closed first."""


def _describe(exc: BaseException) -> str:
    return str(exc) or type(exc).__name__


@contextlib.contextmanager
def _children_ignore_ctrl_c():
    """Processes started inside this block ignore Ctrl+C from their first
    instruction. _serve ignores it for good once it runs; before that, a
    Ctrl+C would stop the child in its start-up with a traceback."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        # The "ignore Ctrl+C" flag is per process and new processes copy it.
        # Windows has no call to read it, so afterwards it is always turned
        # back off, even in a process that had inherited it.
        with _console_lock:
            ignoring = kernel32.SetConsoleCtrlHandler(None, True)
            try:
                yield
            finally:
                if ignoring:
                    kernel32.SetConsoleCtrlHandler(None, False)
    else:
        # A blocked signal stays blocked across exec. Only this thread's mask
        # changes, so a Ctrl+C meanwhile still reaches this process.
        old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _serve(engine: str, engine_kwargs: dict, jobs, results: Connection) -> None:
    # Ctrl+C reaches every process in the console group. The parent decides
    # when the worker stops, through close().
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if sys.platform != "win32":
        # Blocked since spawn (see _children_ignore_ctrl_c). A Ctrl+C that
        # came during start-up is pending and is dropped here, now that
        # SIGINT is ignored.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})
    # huggingface_hub reads this once, on import, which happens below when
    # the engine is built.
    os.environ["HF_HUB_OFFLINE"] = "1"
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
        self._load_answered = False
        self._pending: dict[int, Future[Transcript]] = {}
        self._lock = threading.Lock()
        self._start_lock = threading.Lock()
        # Starts in flight, from start() and start_async(); guarded by _lock.
        self._starting = 0
        self._ids = itertools.count()
        self._load_error: SttWorkerError | None = None
        self._ended = False
        self._closed = False

    def start(self) -> "SttWorker":
        """Spawn the child. Raises SttWorkerError, leaving nothing running,
        if it cannot be started (for example an engine argument that cannot
        be pickled) or if close() came first."""
        with self._lock:
            if self._closed:
                raise SttWorkerError("STT worker is closed")
            self._starting += 1
        try:
            # close() takes this lock too, so it waits for a spawn in progress
            # and then stops what it brought up.
            with self._start_lock:
                with self._lock:
                    if self._closed:
                        raise SttWorkerError("STT worker is closed")
                return self._spawn()
        finally:
            with self._lock:
                self._starting -= 1

    def _spawn(self) -> "SttWorker":
        if self._process is not None:
            raise SttWorkerError("STT worker already started")
        # Checked before any child exists. On Windows, Process.start() creates
        # the child first and pickles after; when that fails, the stray child
        # can later close a handle in this process that has been reused.
        try:
            pickle.dumps((self._engine, self._engine_kwargs))
        except Exception as exc:
            raise SttWorkerError(
                "could not start the STT worker: engine arguments must be plain, "
                f"picklable data: {_describe(exc)}"
            ) from exc
        ctx = multiprocessing.get_context("spawn")
        jobs = ctx.Queue()
        results, child_end = ctx.Pipe(duplex=False)
        process = ctx.Process(
            target=_serve,
            args=(self._engine, self._engine_kwargs, jobs, child_end),
            name=f"voiceqa-stt-{self._engine}",
            daemon=True,
        )
        try:
            with _children_ignore_ctrl_c():
                process.start()
        except BaseException as exc:
            jobs.cancel_join_thread()
            jobs.close()
            results.close()
            if isinstance(exc, Exception):
                raise SttWorkerError(f"could not start the STT worker: {_describe(exc)}") from exc
            raise
        finally:
            # Without our copy of the child's end, the pipe reads as closed as
            # soon as the child exits.
            child_end.close()
        self._jobs, self._results, self._process = jobs, results, process
        self._reader = threading.Thread(
            target=self._read_results, name="voiceqa-stt-results", daemon=True
        )
        self._reader.start()
        return self

    async def start_async(self) -> "SttWorker":
        """start() without blocking the event loop: the spawn runs in a
        thread. If this is cancelled, the child is closed once it is up."""
        started: Future[SttWorker] = Future()

        def run() -> None:
            try:
                # Cancelled before this thread got going: nothing to start.
                if not started.set_running_or_notify_cancel():
                    return
                try:
                    started.set_result(self.start())
                except BaseException as exc:
                    started.set_exception(exc)
            finally:
                with self._lock:
                    self._starting -= 1

        # Counted from now, not from when the thread runs, so a close() in
        # between knows a start is on its way.
        with self._lock:
            self._starting += 1
        threading.Thread(target=run, name="voiceqa-stt-start", daemon=True).start()
        try:
            return await asyncio.wrap_future(started)
        except asyncio.CancelledError:
            # The spawn cannot be stopped half way. This runs in the start
            # thread once it is done, or here at once if it already is.
            started.add_done_callback(self._close_once_started)
            raise

    def _close_once_started(self, started: "Future[SttWorker]") -> None:
        # This can run on the event loop thread, and close() blocks for as
        # long as the child takes to stop.
        if not started.cancelled() and started.exception() is None:
            threading.Thread(
                target=self.close, args=(0,), name="voiceqa-stt-close", daemon=True
            ).start()

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
            # Under the lock, so close() cannot close the queue between the
            # check above and this put. The queue is unbounded: put never
            # waits.
            self._jobs.put((job_id, samples, rate))
        return future

    async def transcribe(self, samples: np.ndarray, rate: int) -> Transcript:
        return await asyncio.wrap_future(self.submit(samples, rate))

    async def wait_ready(self) -> None:
        """Wait until the engine is loaded in the child. Raises SttWorkerError
        if it could not be built (missing extra or model, unknown name)."""
        if self._process is None:
            raise SttWorkerError("STT worker not started")
        # _ready is shared by every caller; one caller giving up (a timeout)
        # must not cancel it for the rest.
        await asyncio.shield(asyncio.wrap_future(self._ready))

    def close(self, timeout: float = DEFAULT_CLOSE_TIMEOUT) -> None:
        """Stop the child. Jobs already queued may finish within `timeout`
        seconds; after that the child is terminated and the jobs still
        pending fail with SttWorkerError. Safe to call more than once, and
        on a worker that never started. A start in progress is waited for
        and then stopped; one not yet begun fails with SttWorkerError."""
        with self._lock:
            if self._closed or (self._process is None and not self._starting):
                return
            self._closed = True
        with self._start_lock:
            pass
        process = self._process
        if process is None:
            return
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
        return await self.start_async()

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
                    data = conn.recv_bytes()
                except (EOFError, OSError):
                    break
                try:
                    kind, job_id, payload = pickle.loads(data)
                except Exception as exc:
                    self._unreadable(exc)
                else:
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
            self._load_answered = True
            _resolve(self._ready, result=None)
            return
        if kind == "failed":
            self._load_answered = True
            self._fail_load(SttWorkerError(payload))
            return
        with self._lock:
            future = self._pending.pop(job_id, None)
        if future is None:
            return
        if kind != "done":
            _resolve(future, exception=SttWorkerError(payload))
            return
        try:
            transcript = Transcript.model_validate(payload)
        except Exception as exc:
            error = SttWorkerError(f"STT worker sent a bad transcript: {_describe(exc)}")
            _resolve(future, exception=error)
        else:
            _resolve(future, result=transcript)

    def _unreadable(self, exc: Exception) -> None:
        error = SttWorkerError(f"unreadable message from the STT worker: {_describe(exc)}")
        if not self._load_answered:
            # The child's first message always says whether the engine loaded.
            self._load_answered = True
            self._fail_load(error)
            return
        # The child answers jobs one at a time in the order they were sent,
        # so this was the result for the oldest job still pending.
        with self._lock:
            if not self._pending:
                return
            future = self._pending.pop(min(self._pending))
        _resolve(future, exception=error)

    def _fail_load(self, error: SttWorkerError) -> None:
        with self._lock:
            self._load_error = error
        _resolve(self._ready, exception=error)


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
