"""SttWorker runs an engine in a spawned child process.

FakeStt runs in the child for everything but the one model test, so no
model is needed. Spawn start-up is slow on Windows CI (a few seconds), so
waits have generous bounds, and every test checks that no child process is
left behind.
"""

import asyncio
import multiprocessing
import os
import signal
import time

import numpy as np
import pytest

from voiceqa.audio.wav import read_wav
from voiceqa.speech import Segment, Transcript
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


def run(coro):
    async def bounded():
        return await asyncio.wait_for(coro, WAIT)

    return asyncio.run(bounded())


def test_transcribes_a_clip_and_closes_cleanly(clip):
    worker = fake_worker().start()
    try:
        result = run(worker.transcribe(*clip))
    finally:
        worker.close()

    assert result == SCRIPTED
    assert worker.exitcode == 0
    assert not worker.is_alive()


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


def test_jobs_come_back_to_the_job_that_asked():
    async def main():
        with fake_worker() as worker:
            short = np.zeros(800, dtype=np.int16)
            return await asyncio.gather(*(worker.transcribe(short, 8000) for _ in range(5)))

    assert run(main()) == [SCRIPTED] * 5


def test_close_lets_queued_jobs_finish_within_the_timeout(clip):
    worker = fake_worker(delay=0.2).start()
    jobs = [worker.submit(*clip) for _ in range(3)]
    worker.close(timeout=WAIT)

    assert [job.result(timeout=0) for job in jobs] == [SCRIPTED] * 3
    assert worker.exitcode == 0


def test_close_fails_jobs_still_pending_when_the_timeout_runs_out(clip):
    worker = fake_worker(delay=60).start()
    jobs = [worker.submit(*clip) for _ in range(2)]
    worker.close(timeout=0.5)

    assert not worker.is_alive()
    for job in jobs:
        with pytest.raises(SttWorkerError, match="closed"):
            job.result(timeout=0)


def test_leaving_the_with_block_on_an_error_does_not_wait_for_jobs(clip):
    started = time.perf_counter()
    with pytest.raises(KeyboardInterrupt):
        with fake_worker(delay=60) as worker:
            run(worker.wait_ready())
            job = worker.submit(*clip)
            raise KeyboardInterrupt
    # close() would give the job DEFAULT_CLOSE_TIMEOUT (10 s) to finish on a
    # normal exit; on an error it stops the child straight away.
    assert time.perf_counter() - started < DEFAULT_CLOSE_TIMEOUT
    assert not worker.is_alive()
    with pytest.raises(SttWorkerError, match="closed"):
        job.result(timeout=0)


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


def test_an_engine_error_comes_back_for_that_job(clip):
    with fake_worker(error="no speech engine today") as worker:
        with pytest.raises(SttWorkerError, match="no speech engine today"):
            run(worker.transcribe(*clip))
        assert worker.is_alive()
    assert worker.exitcode == 0


def test_an_engine_that_cannot_be_built_fails_ready_and_jobs(clip):
    with SttWorker("no-such-engine") as worker:
        with pytest.raises(SttWorkerError, match="unknown STT engine 'no-such-engine'"):
            run(worker.wait_ready())
        with pytest.raises(SttWorkerError, match="unknown STT engine"):
            run(worker.transcribe(*clip))


def test_samples_are_checked_before_they_are_sent():
    with fake_worker() as worker:
        with pytest.raises(TypeError):
            worker.submit(np.zeros(10, dtype=np.float32), 8000)
        with pytest.raises(ValueError):
            worker.submit(np.zeros((10, 2), dtype=np.int16), 8000)


def test_submit_needs_a_started_open_worker(clip):
    worker = fake_worker()
    with pytest.raises(SttWorkerError, match="not started"):
        worker.submit(*clip)
    worker.start()
    worker.close()
    with pytest.raises(SttWorkerError, match="closed"):
        worker.submit(*clip)


@pytest.mark.model
def test_tiny_en_in_the_worker(test_model):
    clip = load_bench_clip(default_clip_dir())

    async def main():
        async with SttWorker("faster-whisper", model=test_model, cpu_threads=1) as worker:
            return await worker.transcribe(clip.samples, clip.rate)

    transcript = run(main())

    assert transcript.engine == "faster-whisper"
    assert transcript.model == test_model
    reference = " ".join(u.text for u in clip.utterances)
    hypothesis = " ".join(s.text for s in transcript.segments)
    # Same bound as the in-process whole-clip test in test_whisper_model.py.
    assert wer(reference, hypothesis) <= 0.25, hypothesis
