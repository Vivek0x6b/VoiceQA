import numpy as np
import pytest

from voiceqa.audio.analysis import first_loud_frame, pulse_round_trip_delays
from voiceqa.audio.wav import write_wav
from voiceqa.telephony.media import SAMPLE_RATE

AMPLITUDE = 10000  # well above the -40 dBFS / RMS 328 threshold


def tone(n, amplitude=AMPLITUDE, freq_hz=1000):
    t = np.arange(n) / SAMPLE_RATE
    return np.round(amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


def pulses(total, positions, width=320, amplitude=AMPLITUDE, freq_hz=1000):
    out = np.zeros(total, dtype=np.int16)
    burst = tone(width, amplitude=amplitude, freq_hz=freq_hz)
    for pos in positions:
        out[pos : pos + width] = burst
    return out


def test_first_loud_frame_finds_a_frame_aligned_offset(tmp_path):
    path = tmp_path / "agent.wav"
    silence = np.zeros(480, dtype=np.int16)
    samples = np.concatenate([silence, tone(320)])
    write_wav(path, samples, SAMPLE_RATE)

    assert first_loud_frame(path) == 480


def test_first_loud_frame_returns_none_for_all_silence(tmp_path):
    path = tmp_path / "agent.wav"
    write_wav(path, np.zeros(1000, dtype=np.int16), SAMPLE_RATE)

    assert first_loud_frame(path) is None


def test_first_loud_frame_handles_a_frame_straddling_a_block_boundary(tmp_path):
    path = tmp_path / "agent.wav"
    # Frame index 1 spans samples 160-319. The tone starts at sample 200, so
    # that frame is mostly silence at the start and loud at the end. With a
    # read block size of 300 samples, the block boundary (at sample 300)
    # falls inside that very frame, so the function must stitch the two
    # blocks together to find it.
    samples = np.concatenate([np.zeros(200, dtype=np.int16), tone(600)])
    write_wav(path, samples, SAMPLE_RATE)

    assert first_loud_frame(path, block_frames=300) == 160


def test_first_loud_frame_respects_a_custom_threshold(tmp_path):
    path = tmp_path / "agent.wav"
    quiet = tone(320, amplitude=100)
    write_wav(path, quiet, SAMPLE_RATE)

    assert first_loud_frame(path) is None
    assert first_loud_frame(path, threshold_rms=50) == 0


def test_pulse_round_trip_delays_pairs_each_pulse_with_its_echo():
    caller = pulses(6000, [1000, 5000])
    agent = pulses(6000, [1080, 5080])  # 80 samples = 10 ms later, each time

    timing = pulse_round_trip_delays(caller, agent)

    assert timing.pulses_sent == 2
    assert timing.pulses_matched == 2
    assert timing.delays_ms == pytest.approx([10.0, 10.0], abs=0.2)


def test_pulse_round_trip_delays_leaves_a_missing_echo_unmatched():
    caller = pulses(6000, [1000, 5000])
    agent = pulses(6000, [5090])  # no echo at all for the first pulse

    timing = pulse_round_trip_delays(caller, agent)

    assert timing.pulses_sent == 2
    assert timing.pulses_matched == 1
    assert timing.delays_ms == pytest.approx([11.25], abs=0.2)


def test_pulse_round_trip_delays_ignores_a_spurious_click():
    # A click at 2300 that is not a response to anything must not be paired
    # with either pulse, and must not throw off the pairing that follows it.
    caller = pulses(6000, [1000, 5000])
    agent = pulses(6000, [1080, 2300, 5080])

    timing = pulse_round_trip_delays(caller, agent)

    assert timing.pulses_sent == 2
    assert timing.pulses_matched == 2
    assert timing.delays_ms == pytest.approx([10.0, 10.0], abs=0.2)


def test_pulse_round_trip_delays_counts_a_pulse_right_at_the_start():
    # There is no quiet before sample 0 to measure, but a pulse starting
    # there is still a real pulse and must still be counted.
    caller = pulses(6000, [0, 4000])
    agent = pulses(6000, [80, 4080])

    timing = pulse_round_trip_delays(caller, agent)

    assert timing.pulses_sent == 2
    assert timing.pulses_matched == 2
    assert timing.delays_ms == pytest.approx([10.0, 10.0], abs=0.2)


def test_pulse_round_trip_delays_counts_a_pulse_shortly_after_the_start():
    # Only 100 samples of quiet precede this pulse, well under quiet_ms's
    # 800 samples, but the start of the array still counts as quiet.
    caller = pulses(6000, [100, 4000])
    agent = pulses(6000, [180, 4080])

    timing = pulse_round_trip_delays(caller, agent)

    assert timing.pulses_sent == 2
    assert timing.pulses_matched == 2
    assert timing.delays_ms == pytest.approx([10.0, 10.0], abs=0.2)
