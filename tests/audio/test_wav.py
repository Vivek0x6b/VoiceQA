import wave

import numpy as np
import pytest

from voiceqa.audio.wav import read_wav, write_wav


def test_write_read_mono_round_trip(tmp_path):
    path = tmp_path / "mono.wav"
    samples = np.arange(-32768, 32768, 256, dtype=np.int16)
    write_wav(path, samples, 8000)
    out, rate = read_wav(path)
    assert rate == 8000
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, samples)


def test_write_read_stereo_round_trip(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.stack([np.arange(100, dtype=np.int16), -np.arange(100, dtype=np.int16)], axis=1)
    write_wav(path, samples, 48000)
    out, rate = read_wav(path)
    assert rate == 48000
    assert out.shape == (100, 2)
    np.testing.assert_array_equal(out, samples)


def test_write_empty(tmp_path):
    path = tmp_path / "empty.wav"
    write_wav(path, np.zeros(0, dtype=np.int16), 8000)
    out, rate = read_wav(path)
    assert rate == 8000
    assert out.shape == (0,)


def test_read_rejects_non_16_bit(tmp_path):
    path = tmp_path / "8bit.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes(100))
    with pytest.raises(ValueError, match="16-bit"):
        read_wav(path)


def test_write_rejects_non_int16(tmp_path):
    with pytest.raises(TypeError):
        write_wav(tmp_path / "f.wav", np.zeros(10), 8000)
