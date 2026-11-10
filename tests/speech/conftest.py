import os
from pathlib import Path

import pytest

from voiceqa.speech.model_cache import find_model, resolve_models_dir, stt_extra_installed

TEST_MODEL = "tiny.en"


def _model_skip_reason() -> str | None:
    if not stt_extra_installed():
        return "needs the [stt] extra: pip install voiceqa[stt]"
    if find_model(TEST_MODEL, resolve_models_dir()) is None:
        return (
            f"{TEST_MODEL} is not in {resolve_models_dir()}; "
            f"run: voiceqa models pull stt {TEST_MODEL}"
        )
    return None


@pytest.fixture(scope="session")
def test_model() -> str:
    """tiny.en's name, or a skip with the reason when it cannot run here."""
    reason = _model_skip_reason()
    if reason:
        # CI sets this so a missing model fails loudly instead of skipping.
        if os.environ.get("VOICEQA_REQUIRE_MODEL_TESTS"):
            pytest.fail(reason)
        pytest.skip(reason)
    return TEST_MODEL


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
