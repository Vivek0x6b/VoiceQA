"""SttWorker runs an engine in a spawned child process.

FakeStt runs in the child for everything but the one model test, so no
model is needed. Spawn start-up is slow on Windows CI (a few seconds), so
waits have generous bounds, and every test checks that no child process is
left behind.
"""

import asyncio
import ctypes
import multiprocessing
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from voiceqa.audio.wav import read_wav
from voiceqa.speech import FakeStt, Segment, Transcript
from voiceqa.speech import worker as worker_module
from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.wer import wer
from voiceqa.speech.worker import DEFAULT_CLOSE_TIMEOUT, SttWorker, SttWorkerError

# Bound for anything that waits on the child, so a broken worker fails the
# test instead of hanging the run.
WAIT = 20.0

SCRIPTED = Transcript(
    segments=[Segment(start=0.5, end=1.5, text="hello there")],
    engine="fake",
    model="scripted",
    language="en",
)


@pytest.fixture(autouse=True)
def no_child_left_behind():
    yield
    left = multiprocessing.active_children()
    for child in left:
        child.kill()
        child.join(5)
    assert left == [], f"child processes left running: {left}"


@pytest.fixture(scope="module")
def clip():
    samples, rate = read_wav(default_clip_dir() / "magi_8k.wav")
    return samples, rate


def fake_worker(**kwargs) -> SttWorker:
    return SttWorker("fake", transcript=SCRIPTED.model_dump(), **kwargs)


def run(coro, bound=WAIT):
    async def bounded():
        return await asyncio.wait_for(coro, bound)

    return asyncio.run(bounded())


@pytest.mark.realtime
def test_transcribes_a_clip_and_closes_cleanly(clip):
    worker = fake_worker().start()
    try:
        result = run(worker.transcribe(*clip))
    finally:
        worker.close()

    assert result == SCRIPTED
    assert worker.exitcode == 0
    assert not worker.is_alive()


@pytest.mark.realtime
def test_context_managers_start_and_close(clip):
    with fake_worker() as worker:
        assert worker.is_alive()
        assert run(worker.transcribe(*clip)) == SCRIPTED
    assert worker.exitcode == 0

    async def main():
        async with fake_worker() as worker:
            await worker.wait_ready()
            return worker, await worker.transcribe(*clip)

    worker, result = run(main())
    assert result == SCRIPTED
    assert worker.exitcode == 0


def echoed(transcript: Transcript) -> str:
    return " ".join(s.text for s in transcript.segments)


@pytest.mark.realtime
def test_jobs_come_back_to_the_job_that_asked():
    lengths = [800, 2400, 160, 4000, 1200]

    async def main():
        with fake_worker(echo=True) as worker:
            jobs = (worker.transcribe(np.zeros(n, dtype=np.int16), 8000) for n in lengths)
            return await asyncio.gather(*jobs)

    results = run(main())

    assert [echoed(r) for r in results] == [f"{n} samples at 8000 Hz" for n in lengths]


@pytest.mark.realtime
def test_close_lets_queued_jobs_finish_within_the_timeout(clip):
    worker = fake_worker(delay=0.2).start()
    jobs = [worker.submit(*clip) for _ in range(3)]
    worker.close(timeout=WAIT)

    assert [job.result(timeout=0) for job in jobs] == [SCRIPTED] * 3
    assert worker.exitcode == 0


@pytest.mark.realtime
def test_close_fails_jobs_still_pending_when_the_timeout_runs_out(clip):
    worker = fake_worker(delay=60).start()
    jobs = [worker.submit(*clip) for _ in range(2)]
    worker.close(timeout=0.5)

    assert not worker.is_alive()
    for job in jobs:
        with pytest.raises(SttWorkerError, match="closed"):
            job.result(timeout=0)


@pytest.mark.realtime
def test_leaving_the_with_block_on_an_error_does_not_wait_for_jobs(clip):
    with pytest.raises(KeyboardInterrupt):
        with fake_worker(delay=60) as worker:
            run(worker.wait_ready())
            job = worker.submit(*clip)
            # Only the close path is timed, not spawn start-up.
            started = time.perf_counter()
            raise KeyboardInterrupt
    # close() would give the job DEFAULT_CLOSE_TIMEOUT (10 s) to finish on a
    # normal exit; on an error it stops the child straight away.
    assert time.perf_counter() - started < DEFAULT_CLOSE_TIMEOUT
    assert not worker.is_alive()
    with pytest.raises(SttWorkerError, match="closed"):
        job.result(timeout=0)


@pytest.mark.realtime
def test_child_killed_mid_job_is_an_error_not_a_hang(clip):
    with fake_worker(delay=60) as worker:
        run(worker.wait_ready())
        job = worker.submit(*clip)
        os.kill(worker.pid, signal.SIGTERM)
        with pytest.raises(SttWorkerError, match="ended"):
            job.result(timeout=WAIT)
        # Jobs sent after the child died fail straight away too.
        with pytest.raises(SttWorkerError):
            worker.submit(*clip).result(timeout=WAIT)
    assert worker.exitcode != 0


@pytest.mark.realtime
def test_event_loop_keeps_running_while_a_job_runs(clip):
    async def main():
        async with fake_worker(delay=1.0) as worker:
            await worker.wait_ready()
            ticks = 0

            async def ticker():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.02)
                    ticks += 1

            task = asyncio.create_task(ticker())
            try:
                result = await worker.transcribe(*clip)
            finally:
                task.cancel()
            return result, ticks

    result, ticks = run(main())

    assert result == SCRIPTED
    # The job takes 1 s, about 50 ticks of 20 ms. A transcribe() that
    # blocked the loop would let through none at all.
    assert ticks >= 10


@pytest.mark.realtime
def test_an_engine_error_comes_back_for_that_job(clip):
    with fake_worker(error="no speech engine today") as worker:
        with pytest.raises(SttWorkerError, match="no speech engine today"):
            run(worker.transcribe(*clip))
        assert worker.is_alive()
    assert worker.exitcode == 0


@pytest.mark.realtime
def test_an_engine_that_cannot_be_built_fails_ready_and_jobs(clip):
    with SttWorker("no-such-engine") as worker:
        with pytest.raises(SttWorkerError, match="unknown STT engine 'no-such-engine'"):
            run(worker.wait_ready())
        with pytest.raises(SttWorkerError, match="unknown STT engine"):
            run(worker.transcribe(*clip))


@pytest.mark.realtime
def test_samples_are_checked_before_they_are_sent():
    with fake_worker() as worker:
        with pytest.raises(TypeError):
            worker.submit(np.zeros(10, dtype=np.float32), 8000)
        with pytest.raises(ValueError):
            worker.submit(np.zeros((10, 2), dtype=np.int16), 8000)


@pytest.mark.realtime
def test_submit_needs_a_started_open_worker(clip):
    worker = fake_worker()
    with pytest.raises(SttWorkerError, match="not started"):
        worker.submit(*clip)
    worker.start()
    worker.close()
    with pytest.raises(SttWorkerError, match="closed"):
        worker.submit(*clip)


@pytest.mark.realtime
def test_submit_racing_close_fails_the_job_not_the_caller(clip):
    worker = fake_worker().start()
    run(worker.wait_ready())
    real_put = worker._jobs.put

    def put(item):
        # Let close() run in full between submit's closed check and the
        # put, whenever submit leaves that gap open.
        if item is not None and worker._lock.acquire(blocking=False):
            worker._lock.release()
            worker.close()
        real_put(item)

    worker._jobs.put = put
    try:
        job = worker.submit(*clip)
    finally:
        worker.close()

    try:
        assert job.result(timeout=WAIT) == SCRIPTED
    except SttWorkerError as exc:
        assert "closed" in str(exc)


def test_engine_arguments_that_cannot_be_pickled_fail_start_cleanly(clip):
    worker = SttWorker("fake", transcript=SCRIPTED.model_dump(), lock=threading.Lock())
    with pytest.raises(SttWorkerError, match="could not start.*pickl"):
        worker.start()

    worker.close()
    assert worker.pid is None
    assert not worker.is_alive()
    with pytest.raises(SttWorkerError, match="not started"):
        worker.submit(*clip)


def test_a_spawn_that_fails_leaves_nothing_behind(monkeypatch, clip):
    def no_spawn(self):
        raise OSError("no more processes")

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", no_spawn)
    worker = fake_worker()
    with pytest.raises(SttWorkerError, match="could not start.*no more processes"):
        worker.start()

    worker.close()
    assert worker.pid is None
    with pytest.raises(SttWorkerError, match="not started"):
        worker.submit(*clip)

    monkeypatch.undo()
    # Nothing half-built is left in the way of a second try.
    with worker:
        assert run(worker.transcribe(*clip)) == SCRIPTED


@pytest.mark.realtime
def test_an_unreadable_result_fails_only_its_job(monkeypatch):
    real_loads = pickle.loads
    garbled = []

    def loads(data):
        message = real_loads(data)
        if message[0] == "done" and not garbled:
            garbled.append(message)
            raise pickle.UnpicklingError("garbled message")
        return message

    monkeypatch.setattr(worker_module, "pickle", SimpleNamespace(loads=loads, dumps=pickle.dumps))
    with fake_worker(echo=True) as worker:
        first = worker.submit(np.zeros(800, dtype=np.int16), 8000)
        second = worker.submit(np.zeros(1600, dtype=np.int16), 8000)
        with pytest.raises(SttWorkerError, match="garbled message"):
            first.result(timeout=WAIT)
        assert echoed(second.result(timeout=WAIT)) == "1600 samples at 8000 Hz"
        assert worker.is_alive()
    assert worker.exitcode == 0


@pytest.mark.realtime
def test_an_unreadable_load_answer_fails_ready_and_jobs(monkeypatch, clip):
    real_loads = pickle.loads

    def loads(data):
        message = real_loads(data)
        if message[0] == "ready":
            raise pickle.UnpicklingError("garbled ready")
        return message

    monkeypatch.setattr(worker_module, "pickle", SimpleNamespace(loads=loads, dumps=pickle.dumps))
    with fake_worker() as worker:
        with pytest.raises(SttWorkerError, match="garbled ready"):
            run(worker.wait_ready())
        with pytest.raises(SttWorkerError, match="garbled ready"):
            worker.submit(*clip).result(timeout=0)


@pytest.mark.realtime
def test_a_result_that_does_not_validate_fails_only_its_job(monkeypatch):
    rejected = []

    class StrictTranscript:
        @staticmethod
        def model_validate(data):
            if not rejected:
                rejected.append(data)
                raise ValueError("not a transcript")
            return Transcript.model_validate(data)

    monkeypatch.setattr(worker_module, "Transcript", StrictTranscript)
    with fake_worker(echo=True) as worker:
        first = worker.submit(np.zeros(800, dtype=np.int16), 8000)
        second = worker.submit(np.zeros(1600, dtype=np.int16), 8000)
        with pytest.raises(SttWorkerError, match="not a transcript"):
            first.result(timeout=WAIT)
        assert echoed(second.result(timeout=WAIT)) == "1600 samples at 8000 Hz"
    assert worker.exitcode == 0


ORPHAN_PARENT = """
import asyncio, time
from voiceqa.speech.worker import SttWorker
worker = SttWorker("fake", transcript={transcript!r}).start()
asyncio.run(asyncio.wait_for(worker.wait_ready(), 60))
print(worker.pid, flush=True)
time.sleep(600)
"""

_SYNCHRONIZE = 0x00100000


def _open_process(pid: int):
    """A handle that keeps `pid` meaning this process (Windows only)."""
    if sys.platform != "win32":
        return None
    handle = ctypes.windll.kernel32.OpenProcess(_SYNCHRONIZE, False, pid)
    assert handle, f"cannot open process {pid}"
    return handle


def _process_gone(pid: int, handle, wait: float) -> bool:
    if handle is not None:
        return ctypes.windll.kernel32.WaitForSingleObject(handle, int(wait * 1000)) == 0
    deadline = time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        stat = Path(f"/proc/{pid}/stat")
        # An exited child waiting to be reaped by its new parent.
        if stat.exists() and stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return True
        time.sleep(0.1)
    return False


@pytest.mark.realtime
def test_the_child_exits_when_its_parent_is_killed():
    code = ORPHAN_PARENT.format(transcript=SCRIPTED.model_dump())
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    handle = None
    try:
        line = parent.stdout.readline()
        assert line.strip().isdigit(), f"parent did not report a worker pid: {line!r}"
        pid = int(line)
        handle = _open_process(pid)
        parent.kill()
        parent.wait(WAIT)
        # The child checks on its parent about once a second; the bound only
        # stops a broken guard from hanging the run.
        gone = _process_gone(pid, handle, 60)
        if not gone:
            os.kill(pid, signal.SIGTERM)
        assert gone, "the worker child outlived its parent"
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(WAIT)
        parent.stdout.close()
        if handle is not None:
            ctypes.windll.kernel32.CloseHandle(handle)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals; Windows test below")
@pytest.mark.realtime
def test_ctrl_c_while_the_child_starts_up_is_ignored(capfd):
    worker = fake_worker().start()

    # Ctrl+C over and over from spawn until the engine is loaded, so some
    # land while the child is still in Python's start-up.
    async def main():
        ready = asyncio.ensure_future(worker.wait_ready())
        while not ready.done():
            os.kill(worker.pid, signal.SIGINT)
            await asyncio.sleep(0.05)
        await ready

    try:
        run(main())
    finally:
        worker.close()

    assert worker.exitcode == 0
    assert "KeyboardInterrupt" not in capfd.readouterr().err


CTRL_C_PARENT = """
import asyncio, ctypes, signal
from voiceqa.speech.worker import SttWorker
kernel32 = ctypes.windll.kernel32
# This process may have inherited "ignore Ctrl+C" from whatever started the
# test run, which would hide the problem. Turn Ctrl+C back on, then ignore it
# here in the parent through Python only, which a child does not inherit.
kernel32.SetConsoleCtrlHandler(None, False)
signal.signal(signal.SIGINT, signal.SIG_IGN)

async def main():
    worker = SttWorker("fake", transcript={transcript!r}).start()
    ready = asyncio.ensure_future(worker.wait_ready())
    while not ready.done():
        kernel32.GenerateConsoleCtrlEvent(0, 0)
        await asyncio.sleep(0.05)
    await ready
    return worker

worker = asyncio.run(asyncio.wait_for(main(), 60))
worker.close()
print("exitcode", worker.exitcode, flush=True)
"""


@pytest.mark.skipif(sys.platform != "win32", reason="Windows console Ctrl+C")
@pytest.mark.realtime
def test_ctrl_c_while_the_child_starts_up_is_ignored_on_windows():
    # A console of its own, so the Ctrl+C reaches only that parent and its
    # worker, not pytest.
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    result = subprocess.run(
        [sys.executable, "-c", CTRL_C_PARENT.format(transcript=SCRIPTED.model_dump())],
        capture_output=True,
        text=True,
        timeout=120,
        creationflags=subprocess.CREATE_NEW_CONSOLE,
        startupinfo=startup,
    )

    assert result.returncode == 0, result.stderr
    assert "exitcode 0" in result.stdout
    assert "KeyboardInterrupt" not in result.stderr


@pytest.mark.realtime
def test_async_with_starts_the_child_off_the_event_loop(clip):
    started_on = []

    class Recording(SttWorker):
        def start(self):
            started_on.append(threading.current_thread())
            return super().start()

    async def main():
        async with Recording("fake", transcript=SCRIPTED.model_dump()) as worker:
            return threading.current_thread(), await worker.transcribe(*clip)

    loop_thread, result = run(main())

    assert result == SCRIPTED
    assert len(started_on) == 1
    assert started_on[0] is not loop_thread


@pytest.mark.realtime
def test_a_cancelled_start_async_stops_the_child_once_it_is_up(monkeypatch):
    # Held inside the spawn, past start()'s closed check, so the child does
    # come up after the cancel.
    entered = threading.Event()
    go = threading.Event()
    real_start = multiprocessing.context.SpawnProcess.start

    def slow_start(self):
        entered.set()
        go.wait(WAIT)
        real_start(self)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", slow_start)
    worker = fake_worker()

    async def main():
        task = asyncio.create_task(worker.start_async())
        while not entered.is_set():
            await asyncio.sleep(0.01)
        task.cancel()
        go.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(main())

    deadline = time.monotonic() + WAIT
    while worker.exitcode is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert worker.exitcode is not None
    assert not worker.is_alive()


@pytest.mark.realtime
def test_a_start_async_cancelled_before_its_thread_runs_starts_nothing(monkeypatch):
    held = []

    class HeldThread(threading.Thread):
        def start(self):
            held.append(self)

    started = []

    class Recording(SttWorker):
        def start(self):
            started.append(self)
            return super().start()

    monkeypatch.setattr(
        worker_module, "threading", SimpleNamespace(Thread=HeldThread, Lock=threading.Lock)
    )
    worker = Recording("fake", transcript=SCRIPTED.model_dump())

    async def main():
        task = asyncio.create_task(worker.start_async())
        while not held:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    run(main())
    threading.Thread.start(held[0])
    held[0].join(WAIT)

    assert started == []
    assert worker.pid is None


@pytest.mark.realtime
def test_a_cancelled_wait_ready_does_not_spoil_later_waits(monkeypatch):
    # The reader holds the child's "ready" until the first wait is cancelled.
    release = threading.Event()
    real_loads = pickle.loads

    def loads(data):
        message = real_loads(data)
        if message[0] == "ready":
            release.wait(WAIT)
        return message

    monkeypatch.setattr(worker_module, "pickle", SimpleNamespace(loads=loads, dumps=pickle.dumps))
    with fake_worker() as worker:

        async def main():
            waiting = asyncio.ensure_future(worker.wait_ready())
            await asyncio.sleep(0.05)
            assert not waiting.done()
            waiting.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiting
            release.set()
            await worker.wait_ready()

        run(main())


@pytest.mark.realtime
def test_a_retry_after_a_late_cancel_of_start_async_is_refused(monkeypatch):
    # Teardown threads are held back, so a retry runs before any of them.
    held = []

    class HeldCloseThread(threading.Thread):
        def start(self):
            if self.name == "voiceqa-stt-close":
                held.append(self)
            else:
                super().start()

    monkeypatch.setattr(
        worker_module, "threading", SimpleNamespace(Thread=HeldCloseThread, Lock=threading.Lock)
    )
    spawned = []
    real_start = multiprocessing.context.SpawnProcess.start

    def counting_start(self):
        spawned.append(self)
        real_start(self)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", counting_start)
    worker = fake_worker()

    async def main():
        task = asyncio.create_task(worker.start_async())
        await asyncio.sleep(0)
        # Block the loop until start() is done, so the cancel lands late.
        deadline = time.monotonic() + WAIT
        while worker._reader is None and time.monotonic() < deadline:
            time.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(SttWorkerError, match="closed"):
            worker.start()
        with pytest.raises(SttWorkerError, match="closed"):
            await worker.start_async()

    run(main())
    first_child = spawned[0]
    for thread in held:
        threading.Thread.start(thread)
        thread.join(WAIT)
    wait_for_exit(worker)

    assert spawned == [first_child]
    assert worker._process is first_child
    assert worker.exitcode is not None


def wait_for_exit(worker):
    deadline = time.monotonic() + WAIT
    while worker.exitcode is None and time.monotonic() < deadline:
        time.sleep(0.05)


@pytest.mark.realtime
def test_close_during_a_start_waits_for_it_and_stops_the_child(monkeypatch):
    in_spawn = threading.Event()
    go = threading.Event()
    real_start = multiprocessing.context.SpawnProcess.start

    def slow_start(self):
        in_spawn.set()
        go.wait(WAIT)
        real_start(self)

    monkeypatch.setattr(multiprocessing.context.SpawnProcess, "start", slow_start)
    worker = fake_worker()
    starter = threading.Thread(target=worker.start)
    starter.start()
    assert in_spawn.wait(WAIT)
    closer = threading.Thread(target=worker.close)
    closer.start()
    # close() must not return while the spawn it would miss is still going.
    closer.join(0.2)
    assert closer.is_alive()
    go.set()
    starter.join(WAIT)
    closer.join(WAIT)

    assert not closer.is_alive()
    assert worker.exitcode is not None
    assert not worker.is_alive()


@pytest.mark.realtime
def test_close_before_start_async_gets_going_starts_nothing(monkeypatch):
    held = []

    class HeldThread(threading.Thread):
        def start(self):
            held.append(self)

    monkeypatch.setattr(
        worker_module, "threading", SimpleNamespace(Thread=HeldThread, Lock=threading.Lock)
    )
    worker = fake_worker()

    async def main():
        task = asyncio.create_task(worker.start_async())
        while not held:
            await asyncio.sleep(0.01)
        worker.close()
        threading.Thread.start(held[0])
        with pytest.raises(SttWorkerError, match="closed"):
            await task

    run(main())

    assert worker.pid is None


@pytest.mark.realtime
def test_a_late_cancel_of_start_async_closes_off_the_event_loop():
    closed_on = []

    class Recording(SttWorker):
        def _stop(self, process, timeout):
            closed_on.append(threading.current_thread())
            super()._stop(process, timeout)

    worker = Recording("fake", transcript=SCRIPTED.model_dump())

    async def main():
        task = asyncio.create_task(worker.start_async())
        await asyncio.sleep(0)
        # Block the loop until start() is done in its thread, so the cancel
        # below lands after the start finished but before the task resumes.
        deadline = time.monotonic() + WAIT
        while worker._reader is None and time.monotonic() < deadline:
            time.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return threading.current_thread()

    loop_thread = run(main())
    wait_for_exit(worker)

    assert worker.exitcode is not None
    assert closed_on and loop_thread not in closed_on


def test_the_child_never_reaches_the_hugging_face_hub(monkeypatch):
    # _serve runs here in the test process so the engine can report the
    # environment it was built in.
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setattr(worker_module.signal, "signal", lambda *args: None)
    seen = []

    def create(name, **kwargs):
        seen.append(os.environ.get("HF_HUB_OFFLINE"))
        return FakeStt(**kwargs)

    monkeypatch.setattr(worker_module, "create_stt", create)
    jobs = queue.Queue()
    jobs.put(None)
    results, child_end = multiprocessing.Pipe(duplex=False)

    worker_module._serve("fake", {"transcript": SCRIPTED.model_dump()}, jobs, child_end)

    assert seen == ["1"]
    assert results.recv() == ("ready", None, None)


@pytest.mark.model
@pytest.mark.realtime
def test_tiny_en_in_the_worker(test_model):
    clip = load_bench_clip(default_clip_dir())

    async def main():
        async with SttWorker("faster-whisper", model=test_model, cpu_threads=1) as worker:
            return await worker.transcribe(clip.samples, clip.rate)

    # Spawn, model load and the whole clip: a hang guard, not a speed check.
    transcript = run(main(), bound=60)

    assert transcript.engine == "faster-whisper"
    assert transcript.model == test_model
    reference = " ".join(u.text for u in clip.utterances)
    hypothesis = " ".join(s.text for s in transcript.segments)
    # Same bound as the in-process whole-clip test in test_whisper_model.py.
    assert wer(reference, hypothesis) <= 0.25, hypothesis
