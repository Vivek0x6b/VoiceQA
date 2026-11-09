"""Measuring VoiceQA's own round-trip latency against a local echo agent.

voiceqa calibrate sends a train of short tone pulses to the local echo
agent (voiceqa.telephony.local.agent), placed through the same call flow
as voiceqa call (engine.call.run_call), and times how long each pulse
takes to come back. The number this produces includes our own sender's
20 ms pacing, the local UDP hop out and back, the echo agent's 20 ms
flush and its own sender, and our receive path. It says nothing about how
fast a real voice agent thinks: it is VoiceQA's own floor, meant to be
subtracted from any latency measured against a real agent.
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from voiceqa.audio.analysis import pulse_round_trip_delays
from voiceqa.audio.dsp import generate_tone
from voiceqa.audio.wav import read_wav
from voiceqa.engine.call import run_call
from voiceqa.telephony.local.agent import AgentConfig, LocalAgent
from voiceqa.telephony.media import SAMPLE_RATE

BURST_MS = 40.0
PERIOD_MS = 500.0
BURST_HZ = 1000.0
# Extra time to keep the call up after the last pulse, so its echo has
# time to come back before we hang up.
TAIL_SECONDS = 1.0


def _pulse_starts(seconds: float, period_ms: float, sample_rate: int) -> range:
    total = round(seconds * sample_rate)
    period = max(round(period_ms / 1000 * sample_rate), 1)
    return range(0, total, period)


def generate_pulse_train(
    seconds: float,
    burst_ms: float = BURST_MS,
    period_ms: float = PERIOD_MS,
    freq_hz: float = BURST_HZ,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = 0.5,
) -> np.ndarray:
    """A burst_ms burst of freq_hz every period_ms, starting at 0, for seconds total."""
    total = round(seconds * sample_rate)
    burst = generate_tone(freq_hz, burst_ms / 1000, sample_rate, amplitude)
    out = np.zeros(total, dtype=np.int16)
    for start in _pulse_starts(seconds, period_ms, sample_rate):
        end = min(start + len(burst), total)
        out[start:end] = burst[: end - start]
    return out


def pulse_count(
    seconds: float, period_ms: float = PERIOD_MS, sample_rate: int = SAMPLE_RATE
) -> int:
    """How many pulses generate_pulse_train(seconds, period_ms, ...) produces.

    This is the authoritative "pulses sent" count: it comes straight from
    how the train was built, not from detecting onsets in the recording
    afterward, which can miss a pulse that starts too close to the edge of
    the file to have a full quiet gap in front of it.
    """
    return len(_pulse_starts(seconds, period_ms, sample_rate))


@dataclass(frozen=True)
class CalibrationResult:
    created_at: str
    samples: int
    median_ms: float | None
    p95_ms: float | None
    min_ms: float | None
    max_ms: float | None
    pulses_sent: int
    pulses_matched: int
    answered: bool
    status: int | None
    reason: str | None
    record_dir: Path


async def run_calibration(record_dir: str | Path, seconds: float = 10.0) -> CalibrationResult:
    """Call a local echo agent with a pulse train and measure the round trip.

    Starts a local echo agent in-process on 127.0.0.1 (port 0), places one
    call to it through run_call (the same flow voiceqa call uses) sending a
    pulse train instead of a tone, then finds each pulse's onset in
    caller.wav and agent.wav and pairs them up to get round-trip delays.
    Writes calibration.json into record_dir and returns the same numbers.
    """
    record_dir = Path(record_dir)
    agent = await LocalAgent.start(AgentConfig(host="127.0.0.1", port=0))
    try:
        call_result = await run_call(
            target="calibrate",
            server_host="127.0.0.1",
            server_port=agent.local_addr[1],
            username="voiceqa",
            password="",
            audio=generate_pulse_train(seconds),
            record_dir=record_dir,
            duration=seconds + TAIL_SECONDS,
        )
    finally:
        await agent.close()

    caller, _ = read_wav(record_dir / "caller.wav")
    agent_audio, _ = read_wav(record_dir / "agent.wav")
    timing = pulse_round_trip_delays(caller, agent_audio)

    if timing.delays_ms:
        median_ms = float(np.median(timing.delays_ms))
        p95_ms = float(np.percentile(timing.delays_ms, 95))
        min_ms = float(min(timing.delays_ms))
        max_ms = float(max(timing.delays_ms))
    else:
        median_ms = p95_ms = min_ms = max_ms = None

    result = CalibrationResult(
        created_at=datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        samples=len(timing.delays_ms),
        median_ms=median_ms,
        p95_ms=p95_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        # The onsets pulse_round_trip_delays finds in caller.wav are only
        # used to pair up echoes; how many pulses we actually built is
        # already known, and is not at the mercy of onset detection.
        pulses_sent=pulse_count(seconds),
        pulses_matched=timing.pulses_matched,
        answered=call_result.answered,
        status=call_result.status,
        reason=call_result.reason,
        record_dir=record_dir,
    )
    _write_calibration_json(result)
    return result


def _write_calibration_json(result: CalibrationResult) -> None:
    data = {
        "created_at": result.created_at,
        "samples": result.samples,
        "median_ms": result.median_ms,
        "p95_ms": result.p95_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "pulses_sent": result.pulses_sent,
        "pulses_matched": result.pulses_matched,
        "agent": "local echo",
    }
    path = result.record_dir / "calibration.json"
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
