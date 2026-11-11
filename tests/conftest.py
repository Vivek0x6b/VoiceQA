import os

import pytest

from voiceqa.speech.model_cache import (
    STT_EXTRA_HINT,
    find_model,
    resolve_models_dir,
    stt_extra_installed,
)

TEST_MODEL = "tiny.en"


def _model_skip_reason() -> str | None:
    if not stt_extra_installed():
        return f"needs the [stt] extra: {STT_EXTRA_HINT}"
    if find_model(TEST_MODEL, resolve_models_dir()) is None:
        return (
            f"{TEST_MODEL} is not in {resolve_models_dir()}; "
            f"run: voiceqa models pull stt {TEST_MODEL}"
        )
    return None


def _skip_or_fail(reason: str | None) -> None:
    if reason:
        # CI sets this so a missing model fails loudly instead of skipping.
        if os.environ.get("VOICEQA_REQUIRE_MODEL_TESTS"):
            pytest.fail(reason)
        pytest.skip(reason)


@pytest.fixture(scope="session")
def test_model() -> str:
    """tiny.en's name, or a skip with the reason when it cannot run here."""
    _skip_or_fail(_model_skip_reason())
    return TEST_MODEL


@pytest.fixture(scope="session")
def silero_extra() -> None:
    """Skips when the [stt] extra is missing. The Silero model ships inside
    the faster-whisper wheel, so nothing else is needed."""
    _skip_or_fail(None if stt_extra_installed() else f"needs the [stt] extra: {STT_EXTRA_HINT}")
