import textwrap
import wave
from pathlib import Path

import pytest


def _silent_wav(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\0\0" * 160)
    return path


@pytest.fixture
def write_scenario(tmp_path):
    """Writes YAML text (dedented) to a scenario file and returns its path."""

    def write(text: str, name: str = "scenario.yaml") -> Path:
        path = tmp_path / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        return path

    return write


@pytest.fixture
def silent_wav():
    return _silent_wav
