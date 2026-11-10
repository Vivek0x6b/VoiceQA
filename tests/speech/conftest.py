from pathlib import Path

import pytest


def _make_cached_model(cache: Path, repo: str, files: dict[str, bytes]) -> Path:
    """A hub cache entry laid out the way huggingface_hub writes it without symlinks."""
    org, name = repo.split("/")
    snapshot = cache / f"models--{org}--{name}" / "snapshots" / "0123abcd"
    snapshot.mkdir(parents=True)
    for filename, data in files.items():
        (snapshot / filename).write_bytes(data)
    return snapshot


@pytest.fixture
def make_cached_model():
    return _make_cached_model
