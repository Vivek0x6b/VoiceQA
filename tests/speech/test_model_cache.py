from pathlib import Path

import pytest

from voiceqa.speech import model_cache
from voiceqa.speech.model_cache import (
    MODELS_DIR_ENV,
    cached_models,
    default_hf_cache,
    find_model,
    repo_id,
    resolve_models_dir,
)


@pytest.fixture
def clean_env(monkeypatch):
    for name in (MODELS_DIR_ENV, "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    return monkeypatch


def test_repo_id_maps_short_names_to_systran_repos():
    assert repo_id("tiny.en") == "Systran/faster-whisper-tiny.en"
    assert repo_id("base.en") == "Systran/faster-whisper-base.en"
    assert repo_id("small.en") == "Systran/faster-whisper-small.en"
    assert repo_id("large-v3") == "Systran/faster-whisper-large-v3"


def test_repo_id_passes_full_repo_ids_through():
    assert repo_id("someone/my-whisper-ct2") == "someone/my-whisper-ct2"


def test_repo_id_rejects_unknown_names_and_lists_known_ones():
    with pytest.raises(ValueError, match="base.en"):
        repo_id("enormous.en")


def test_default_hf_cache_without_env_is_under_home(clean_env):
    assert default_hf_cache() == Path.home() / ".cache" / "huggingface" / "hub"


def test_default_hf_cache_follows_hf_home(clean_env, tmp_path):
    clean_env.setenv("HF_HOME", str(tmp_path))
    assert default_hf_cache() == tmp_path / "hub"


def test_default_hf_cache_prefers_hf_hub_cache(clean_env, tmp_path):
    clean_env.setenv("HF_HOME", str(tmp_path / "home"))
    clean_env.setenv("HF_HUB_CACHE", str(tmp_path / "hub-cache"))
    assert default_hf_cache() == tmp_path / "hub-cache"


def test_default_hf_cache_matches_huggingface_hub(clean_env):
    constants = pytest.importorskip("huggingface_hub.constants")
    # The constant is computed at import time from the environment the test
    # run started with, so only compare when that had no overrides either.
    import os

    if any(os.environ.get(n) for n in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")):
        pytest.skip("Hugging Face cache env vars were set when the tests started")
    assert default_hf_cache() == Path(constants.HF_HUB_CACHE)


def test_resolve_models_dir_order(clean_env, tmp_path):
    assert resolve_models_dir() == default_hf_cache()
    clean_env.setenv(MODELS_DIR_ENV, str(tmp_path / "from-env"))
    assert resolve_models_dir() == tmp_path / "from-env"
    assert resolve_models_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_models_dir_source_names_where_the_dir_came_from(clean_env, tmp_path):
    assert model_cache.models_dir_source(None) == "Hugging Face cache"
    clean_env.setenv(MODELS_DIR_ENV, str(tmp_path))
    assert model_cache.models_dir_source(None) == MODELS_DIR_ENV
    assert model_cache.models_dir_source(tmp_path) == "--models-dir"


def test_find_model_returns_the_snapshot_holding_model_bin(tmp_path, make_cached_model):
    snapshot = make_cached_model(
        tmp_path, "Systran/faster-whisper-tiny.en", {"model.bin": b"x" * 10, "config.json": b"{}"}
    )
    assert find_model("tiny.en", tmp_path) == snapshot


def test_find_model_is_none_when_missing_or_incomplete(tmp_path, make_cached_model):
    assert find_model("tiny.en", tmp_path) is None
    make_cached_model(tmp_path, "Systran/faster-whisper-base.en", {"config.json": b"{}"})
    assert find_model("base.en", tmp_path) is None


def test_find_model_accepts_a_local_model_folder(tmp_path):
    folder = tmp_path / "my-model"
    folder.mkdir()
    (folder / "model.bin").write_bytes(b"x")
    assert find_model(str(folder), tmp_path / "unused") == folder


def test_cached_models_lists_whisper_models_with_size(tmp_path, make_cached_model):
    make_cached_model(
        tmp_path,
        "Systran/faster-whisper-tiny.en",
        {"model.bin": b"x" * 1000, "config.json": b"y" * 24},
    )
    make_cached_model(tmp_path, "Systran/faster-whisper-base.en", {"model.bin": b"z" * 50})
    make_cached_model(tmp_path, "bert/bert-base", {"model.bin": b"q" * 7})

    found = cached_models(tmp_path)

    assert [m.name for m in found] == ["base.en", "tiny.en"]
    tiny = found[1]
    assert tiny.repo_id == "Systran/faster-whisper-tiny.en"
    assert tiny.size_bytes == 1024
    assert tiny.path == tmp_path / "models--Systran--faster-whisper-tiny.en"


def test_cached_models_skips_entries_without_model_bin(tmp_path, make_cached_model):
    make_cached_model(tmp_path, "Systran/faster-whisper-small.en", {"config.json": b"{}"})
    assert cached_models(tmp_path) == []


def test_cached_models_of_a_missing_dir_is_empty(tmp_path):
    assert cached_models(tmp_path / "nope") == []


def test_stt_extra_installed_follows_find_spec(monkeypatch):
    monkeypatch.setattr(model_cache.importlib.util, "find_spec", lambda name: None)
    assert model_cache.stt_extra_installed() is False
    monkeypatch.setattr(model_cache.importlib.util, "find_spec", lambda name: object())
    assert model_cache.stt_extra_installed() is True


def test_pull_without_the_extra_names_it(monkeypatch, tmp_path):
    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: False)
    with pytest.raises(model_cache.SttExtraMissingError, match=r"voiceqa\[stt\]"):
        model_cache.pull_stt_model("tiny.en", tmp_path)


def test_pull_skips_a_model_that_is_already_cached(monkeypatch, tmp_path, make_cached_model):
    snapshot = make_cached_model(tmp_path, "Systran/faster-whisper-tiny.en", {"model.bin": b"x"})

    def no_network(*args, **kwargs):
        raise AssertionError("must not download a cached model")

    monkeypatch.setattr(model_cache, "_download", no_network)
    assert model_cache.pull_stt_model("tiny.en", tmp_path) == snapshot


def test_pull_downloads_into_the_models_dir(monkeypatch, tmp_path, make_cached_model):
    calls = []

    def fake_download(model, cache_dir):
        calls.append((model, cache_dir))
        return str(make_cached_model(cache_dir, repo_id(model), {"model.bin": b"x"}))

    monkeypatch.setattr(model_cache, "stt_extra_installed", lambda: True)
    monkeypatch.setattr(model_cache, "_download", fake_download)

    path = model_cache.pull_stt_model("base.en", tmp_path)

    assert calls == [("base.en", tmp_path)]
    assert path == find_model("base.en", tmp_path)
