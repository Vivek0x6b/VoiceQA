import numpy as np
import pytest

from voiceqa.speech import Speech


def test_speech_holds_samples_and_rate():
    samples = np.zeros(10, dtype=np.int16)
    speech = Speech(samples=samples, rate=8000)
    assert speech.rate == 8000
    assert speech.samples is samples


def test_speech_rejects_non_int16():
    with pytest.raises(TypeError, match="int16"):
        Speech(samples=np.zeros(10, dtype=np.float32), rate=8000)


def test_speech_rejects_multichannel():
    with pytest.raises(ValueError, match="mono"):
        Speech(samples=np.zeros((10, 2), dtype=np.int16), rate=8000)


def test_speech_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="rate"):
        Speech(samples=np.zeros(10, dtype=np.int16), rate=0)


def test_speech_is_immutable():
    speech = Speech(samples=np.zeros(10, dtype=np.int16), rate=8000)
    with pytest.raises(AttributeError):
        speech.rate = 16000
