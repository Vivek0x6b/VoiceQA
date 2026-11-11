"""The RTP sender keeps its slots with Silero running in the call process.

Slot lateness (when a frame actually went out minus its slot) is compared
between calls with Silero VAD and calls with no VAD at all, in the same
test run on the same machine, and against the sender's own contract: a
frame more than MAX_LATE_FRAMES late makes it skip slots.
"""

import asyncio
from pathlib import Path

import numpy as np
import pytest

from voiceqa.audio.wav import read_wav
from voiceqa.engine import session as session_module
from voiceqa.engine.session import CallSession
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.media import SAMPLE_RATE, media_clock
from voiceqa.telephony.rtp import RtpSession
from voiceqa.telephony.rtp.session import FRAME_SECONDS

pytestmark = [pytest.mark.model, pytest.mark.usefixtures("silero_extra")]

CLIP = Path(__file__).parents[1] / "data" / "speech" / "magi_8k.wav"
CALL_SECONDS = 4.0
ROUNDS = 2


class TimedRtp(RtpSession):
    """Notes, for every frame, how late it went out against its slot."""

    sessions: list["TimedRtp"] = []

    def __init__(self):
        super().__init__()
        self.dues: list[float] = []
        self.lateness: list[float] = []
        TimedRtp.sessions.append(self)

    def _send_frame(self, due):
        self.lateness.append(media_clock() - due)
        self.dues.append(due)
        super()._send_frame(due)


async def one_call(record_dir, vad, speech):
    agent = await LocalAgent.start(AgentConfig(host="127.0.0.1", port=0, timer_t1=0.02, play=CLIP))
    try:
        session = CallSession(
            "agent", "127.0.0.1", agent.local_addr[1], "1001", "", record_dir, vad=vad
        )
        async with session:
            await session.say(speech[: round((CALL_SECONDS - 1) * SAMPLE_RATE)])
            await asyncio.sleep(1.0)
        return session
    finally:
        await agent.close()


def skipped_slots(dues):
    steps = np.round(np.diff(dues) / FRAME_SECONDS).astype(int)
    return int(np.sum(steps - 1))


def test_silero_vad_does_not_make_the_rtp_sender_late(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "RtpSession", TimedRtp)
    TimedRtp.sessions = []
    speech, _ = read_wav(CLIP)
    runs = {"none": [], "silero": []}
    sessions = []
    # Alternated, so drift in machine load hits both the same way.
    for i in range(ROUNDS):
        for vad in ("none", "silero"):
            s = asyncio.run(
                one_call(tmp_path / f"{vad}-{i}", None if vad == "none" else vad, speech)
            )
            sessions.append(s)
            runs[vad].append(TimedRtp.sessions[-1])

    silero_sessions = [s for s in sessions if s.vad_name == "silero"]
    # Silero really ran on the agent's speech.
    assert all(s._vad.window_count > 50 for s in silero_sessions)
    assert all(s.dropped_frames == 0 for s in silero_sessions)

    stats = {}
    for vad, rtps in runs.items():
        late = np.concatenate([r.lateness for r in rtps]) * 1000
        stats[vad] = {
            "frames": len(late),
            "p50": float(np.percentile(late, 50)),
            "p95": float(np.percentile(late, 95)),
            "p99": float(np.percentile(late, 99)),
            "max": float(late.max()),
            "skipped": sum(skipped_slots(r.dues) for r in rtps),
        }
    print(stats)

    # The sender's contract: no frame so late (over MAX_LATE_FRAMES) that it
    # skipped slots, unless the machine did that without the VAD as well.
    if stats["none"]["skipped"] == 0:
        assert stats["silero"]["skipped"] == 0
    # Same lateness, within half a frame at the tail. Timer resolution on
    # Windows before 3.13 is 15.6 ms, which both kinds of call share.
    half_frame_ms = FRAME_SECONDS * 1000 / 2
    assert stats["silero"]["p95"] <= stats["none"]["p95"] + half_frame_ms
