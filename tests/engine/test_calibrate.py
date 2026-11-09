import asyncio
import json

import numpy as np

from voiceqa.engine.calibrate import generate_pulse_train, pulse_count, run_calibration


def test_generate_pulse_train_has_a_burst_at_the_start_of_each_period():
    train = generate_pulse_train(seconds=1.5, burst_ms=40, period_ms=500, sample_rate=8000)

    assert len(train) == 12000  # 1.5 s at 8 kHz
    assert np.any(train[0:320] != 0)  # the burst at 0 ms
    assert np.any(train[4000:4320] != 0)  # the burst at 500 ms
    assert np.all(train[400:4000] == 0)  # quiet in between


def test_run_calibration_against_the_local_echo_agent_writes_calibration_json(tmp_path):
    seconds = 2.0
    result = asyncio.run(run_calibration(tmp_path, seconds=seconds))

    # pulses_sent comes from how the train was built, not from detecting
    # onsets, so it must match exactly, not just be "enough".
    assert result.pulses_sent == pulse_count(seconds)
    assert result.answered is True
    # Same status run_call reports for an answered call: used to build the
    # same "answered/no (status reason)" summary line as voiceqa call.
    assert result.status == 200
    # Generous bound: a local loopback round trip should be nowhere near 1 s.
    assert result.pulses_matched >= 1
    assert result.median_ms is not None
    assert 0 <= result.median_ms < 1000

    path = tmp_path / "calibration.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["agent"] == "local echo"
    assert data["created_at"].endswith("Z")
    assert data["pulses_sent"] == result.pulses_sent
    assert data["pulses_matched"] == result.pulses_matched
    assert data["samples"] == result.pulses_matched
