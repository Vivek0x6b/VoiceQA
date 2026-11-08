import wave

import numpy as np
import pytest

from voiceqa.audio.wav import WavWriter, read_wav, read_wav_blocks, write_wav


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


def test_writer_file_is_complete_after_every_write(tmp_path):
    path = tmp_path / "stream.wav"
    w = WavWriter(path, 8000)
    try:
        assert read_wav(path)[0].shape == (0,)
        w.write(np.arange(10, dtype=np.int16))
        out, _ = read_wav(path)
        np.testing.assert_array_equal(out, np.arange(10))
        w.write(np.arange(10, 25, dtype=np.int16))
        out, rate = read_wav(path)
        assert rate == 8000
        np.testing.assert_array_equal(out, np.arange(25))
        assert w.frames == 25
    finally:
        w.close()
    w.close()
    np.testing.assert_array_equal(read_wav(path)[0], np.arange(25))


def test_writer_stereo_and_validation(tmp_path):
    path = tmp_path / "s.wav"
    with WavWriter(path, 16000, channels=2) as w:
        w.write(np.array([[1, -1], [2, -2]], dtype=np.int16))
        with pytest.raises(TypeError):
            w.write(np.zeros((2, 2)))
        with pytest.raises(ValueError):
            w.write(np.zeros(4, dtype=np.int16))
    out, rate = read_wav(path)
    assert rate == 16000
    np.testing.assert_array_equal(out, [[1, -1], [2, -2]])


def test_read_blocks(tmp_path):
    path = tmp_path / "b.wav"
    samples = np.arange(-500, 500, dtype=np.int16)
    write_wav(path, samples, 8000)
    blocks = list(read_wav_blocks(path, 300))
    assert [len(b) for b in blocks] == [300, 300, 300, 100]
    assert all(b.dtype == np.int16 for b in blocks)
    np.testing.assert_array_equal(np.concatenate(blocks), samples)

    write_wav(path, np.stack([samples, -samples], axis=1), 8000)
    blocks = list(read_wav_blocks(path, 400))
    assert blocks[0].shape == (400, 2)
    np.testing.assert_array_equal(np.concatenate(blocks)[:, 1], -samples)
