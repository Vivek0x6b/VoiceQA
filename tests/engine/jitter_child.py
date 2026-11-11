"""Child process for test_session_jitter.py: places calls with one VAD
setting and prints how late each sent frame was against its slot, as JSON.

    python jitter_child.py none|silero CALLS SECONDS RECORD_DIR

It prints "ready" once everything slow (imports, the Silero model) is done,
then waits for a line on stdin, so the parent can start both kinds of call
at the same moment.
"""

import asyncio
import json
import sys
from pathlib import Path

from voiceqa.audio.wav import read_wav
from voiceqa.engine import session as session_module
from voiceqa.engine.session import CallSession
from voiceqa.speech.vad import load_silero_session
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.media import SAMPLE_RATE, media_clock
from voiceqa.telephony.rtp import RtpSession

CLIP = Path(__file__).parents[1] / "data" / "speech" / "magi_8k.wav"


class TimedRtp(RtpSession):
    """Notes, for every frame, its slot and how late it went out."""

    last: "TimedRtp | None" = None

    def __init__(self):
        super().__init__()
        self.dues: list[float] = []
        self.lateness: list[float] = []
        TimedRtp.last = self

    def _send_frame(self, due):
        self.lateness.append(media_clock() - due)
        self.dues.append(due)
        super()._send_frame(due)


async def one_call(record_dir, vad, speech, seconds):
    agent = await LocalAgent.start(AgentConfig(host="127.0.0.1", port=0, timer_t1=0.02, play=CLIP))
    try:
        # freeze_heap as the CLI runs it: the check is about the call path
        # a user gets, which includes the pre-dial collect and freeze.
        session = CallSession(
            "agent",
            "127.0.0.1",
            agent.local_addr[1],
            "1001",
            "",
            record_dir,
            vad=vad,
            freeze_heap=True,
        )
        async with session:
            await session.say(speech[: round((seconds - 1) * SAMPLE_RATE)])
            await asyncio.sleep(1.0)
        rtp = TimedRtp.last
        return {
            "dues": rtp.dues,
            "lateness": rtp.lateness,
            "windows": getattr(session._vad, "window_count", None),
            "dropped": session.dropped_frames,
        }
    finally:
        await agent.close()


def main() -> None:
    mode, record_dir = sys.argv[1], sys.argv[4]
    calls, seconds = int(sys.argv[2]), float(sys.argv[3])
    session_module.RtpSession = TimedRtp
    vad = None if mode == "none" else mode
    if vad == "silero":
        load_silero_session()
    speech, _ = read_wav(CLIP)
    print("ready", flush=True)
    sys.stdin.readline()
    results = [
        asyncio.run(one_call(Path(record_dir) / f"call-{i}", vad, speech, seconds))
        for i in range(calls)
    ]
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
