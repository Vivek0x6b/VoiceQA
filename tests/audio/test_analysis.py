import numpy as np

from voiceqa.audio.analysis import first_loud_frame
from voiceqa.audio.wav import write_wav
from voiceqa.telephony.media import SAMPLE_RATE

AMPLITUDE = 10000  # well above the -40 dBFS / RMS 328 threshold


def tone(n, amplitude=AMPLITUDE, freq_hz=1000):
    t = np.arange(n) / SAMPLE_RATE
    return np.round(amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.int16)


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
