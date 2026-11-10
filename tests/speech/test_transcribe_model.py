"""A run folder transcribed by tiny.en in the worker: agent speech placed
2 s later in agent.wav comes back 2 s later.

The comparison is against the same speech transcribed on its own with 1 s
of silence before it, not against where the speech really starts:
faster-whisper puts the first word of this clip 0.25 to 0.3 s early after
leading silence (and at 0.0 with none), so only the shift is exact.
tests/engine/test_transcribe.py checks absolute times with a fake engine.
"""

import asyncio

import numpy as np
import pytest

from voiceqa.audio.wav import write_wav
from voiceqa.engine.transcribe import transcribe_run
from voiceqa.speech.bench import default_clip_dir, load_bench_clip
from voiceqa.speech.wer import wer
from voiceqa.speech.worker import SttWorker

pytestmark = pytest.mark.model

LEAD = 1.0
OFFSET = 2.0


def test_agent_speech_two_seconds_later_comes_back_two_seconds_later(test_model, tmp_path):
    clip = load_bench_clip(default_clip_dir())
    u = clip.utterances[1]
    speech = clip.samples[round(u.start * clip.rate) : round(u.end * clip.rate)]

    def silence(seconds):
        return np.zeros(round(seconds * clip.rate), dtype=np.int16)

    reference = np.concatenate([silence(LEAD), speech])
    agent = np.concatenate([silence(LEAD + OFFSET), speech, silence(1.0)])
    write_wav(tmp_path / "agent.wav", agent, clip.rate)
    write_wav(tmp_path / "caller.wav", np.zeros_like(agent), clip.rate)

    async def main():
        async with SttWorker("faster-whisper", model=test_model, cpu_threads=1) as worker:
            alone = await worker.transcribe(reference, clip.rate)
            run = await transcribe_run(tmp_path, worker)
            return alone, run

    alone, run = asyncio.run(asyncio.wait_for(main(), 60))

    assert run.segments, "tiny.en heard nothing"
    assert {s.speaker for s in run.segments} == {"agent"}
    shift = run.segments[0].words[0].start - alone.segments[0].words[0].start
    assert shift == pytest.approx(OFFSET, abs=0.2)
    hypothesis = " ".join(s.text for s in run.segments)
    assert wer(u.text, hypothesis) <= 0.6, hypothesis
