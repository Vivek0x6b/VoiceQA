"""Where faster-whisper models live on disk, and which ones are there.

Models sit in a Hugging Face hub cache folder, the layout faster-whisper
and huggingface_hub use: models--<org>--<name>/snapshots/<revision>/. By
default that is the user's Hugging Face cache. VOICEQA_MODELS_DIR or an
explicit --models-dir points at another folder with the same layout.

This module never imports faster-whisper or huggingface_hub at import time,
so `voiceqa doctor` and `voiceqa models list` work without the [stt] extra.
"""

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR_ENV = "VOICEQA_MODELS_DIR"
STT_EXTRA_HINT = 'pip install -e ".[stt]" (from the VoiceQA checkout)'
# Here rather than in whisper.py, which cannot be imported without the extra.
DEFAULT_MODEL = "base.en"

# Whole-repo sizes from the Hugging Face API, checked 2026-11-09.
KNOWN_SIZES_MB = {"tiny.en": 78.1, "base.en": 147.8, "small.en": 486.1}

_SYSTRAN_MODELS = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
)
_SYSTRAN_PREFIX = "Systran/faster-whisper-"


class SttExtraMissingError(RuntimeError):
    pass


class ModelNotAvailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class CachedModel:
    name: str
    repo_id: str
    path: Path
    size_bytes: int


def repo_id(model: str) -> str:
    if "/" in model:
        return model
    if model in _SYSTRAN_MODELS:
        return _SYSTRAN_PREFIX + model
    known = ", ".join(_SYSTRAN_MODELS)
    raise ValueError(f"unknown faster-whisper model {model!r}, known models: {known}")


def _model_name(repo: str) -> str:
    return repo[len(_SYSTRAN_PREFIX) :] if repo.startswith(_SYSTRAN_PREFIX) else repo


def default_hf_cache() -> Path:
    """The hub cache folder huggingface_hub uses, worked out the same way it does."""
    for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(name):
            return Path(os.environ[name]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]).expanduser() / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def resolve_models_dir(models_dir: str | os.PathLike | None = None) -> Path:
    if models_dir is not None:
        return Path(models_dir)
    if os.environ.get(MODELS_DIR_ENV):
        return Path(os.environ[MODELS_DIR_ENV])
    return default_hf_cache()


def models_dir_source(models_dir: str | os.PathLike | None) -> str:
    if models_dir is not None:
        return "--models-dir"
    if os.environ.get(MODELS_DIR_ENV):
        return MODELS_DIR_ENV
    return "Hugging Face cache"


def _snapshot_with_model(repo_dir: Path) -> Path | None:
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = [s for s in snapshots.iterdir() if (s / "model.bin").is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda s: (s / "model.bin").stat().st_mtime)


def is_local_model(model: str) -> bool:
    """True when `model` reads as a folder path rather than a short name or
    a hub repo id ("org/name", exactly one forward slash)."""
    path = Path(model).expanduser()
    if (path / "model.bin").is_file():
        return True
    return path.is_absolute() or model.startswith(".") or "\\" in model or model.count("/") > 1


def find_model(model: str, models_dir: str | os.PathLike | None = None) -> Path | None:
    """The folder holding `model`'s model.bin, or None if it is not on disk.

    `model` is a short name (base.en), a hub repo id, or a local folder that
    already holds a converted model. A local folder that does not exist or
    has no model.bin raises ModelNotAvailableError, since pulling cannot fix
    it.
    """
    if is_local_model(model):
        local = Path(model).expanduser()
        if not local.is_dir():
            raise ModelNotAvailableError(f"local model folder {model} does not exist")
        if not (local / "model.bin").is_file():
            raise ModelNotAvailableError(
                f"local model folder {model} has no model.bin, so it is not a converted "
                "faster-whisper model"
            )
        return local
    org, name = repo_id(model).split("/", 1)
    return _snapshot_with_model(resolve_models_dir(models_dir) / f"models--{org}--{name}")


def usable_model(model: str, models_dir: str | os.PathLike | None = None) -> Path:
    """The folder of `model`, checked to hold everything faster-whisper needs
    to load it without the network. Raises ModelNotAvailableError naming the
    fix otherwise."""
    cache = resolve_models_dir(models_dir)
    path = find_model(model, cache)
    if path is None:
        raise ModelNotAvailableError(
            f"faster-whisper model {model} is not in {cache}. "
            f"Download it first: voiceqa models pull stt {model}"
        )
    # Without it faster-whisper fetches a tokenizer from the hub, which must
    # never happen during a run.
    if not (path / "tokenizer.json").is_file():
        raise ModelNotAvailableError(
            f"faster-whisper model folder {path} has no tokenizer.json. Copy it in "
            "from the original model, or delete the folder and pull the model again."
        )
    return path


def _size_on_disk(folder: Path) -> int:
    # Snapshot files are often symlinks into blobs/, and newer huggingface_hub
    # versions link those blobs again into a shared store at the cache root.
    # Following every snapshot file to its real target and counting each
    # target once gives the model's real size in all of these layouts.
    targets = set()
    for root, _dirs, files in os.walk(folder / "snapshots"):
        for f in files:
            target = (Path(root) / f).resolve()
            if target.is_file():
                targets.add(target)
    return sum(t.stat().st_size for t in targets)


def cached_models(models_dir: str | os.PathLike | None = None) -> list[CachedModel]:
    cache = resolve_models_dir(models_dir)
    if not cache.is_dir():
        return []
    found = []
    for entry in sorted(cache.glob("models--*")):
        parts = entry.name.split("--", 2)
        if len(parts) != 3 or "whisper" not in parts[2].lower():
            continue
        if _snapshot_with_model(entry) is None:
            continue
        repo = f"{parts[1]}/{parts[2]}"
        found.append(CachedModel(_model_name(repo), repo, entry, _size_on_disk(entry)))
    return sorted(found, key=lambda m: m.name)


def stt_extra_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def _download(model: str, cache_dir: Path) -> str:
    from faster_whisper.utils import download_model

    return download_model(repo_id(model), cache_dir=str(cache_dir))


def pull_stt_model(model: str, models_dir: str | os.PathLike | None = None) -> Path:
    """Download `model` into the models dir unless it is already there."""
    cache = resolve_models_dir(models_dir)
    existing = find_model(model, cache)
    if existing is not None:
        return existing
    repo_id(model)
    if not stt_extra_installed():
        raise SttExtraMissingError(
            f"downloading STT models needs the [stt] extra: {STT_EXTRA_HINT}"
        )
    _download(model, cache)
    path = find_model(model, cache)
    if path is None:
        raise ModelNotAvailableError(f"download of {model} finished but no model.bin in {cache}")
    return path
