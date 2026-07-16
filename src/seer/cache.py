"""Fail-closed resolution of pinned Hugging Face snapshots from a local cache."""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QWEN3_REPOSITORY = "Qwen/Qwen3-0.6B"
QWEN3_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MIN_TRANSFORMERS_VERSION = (4, 51)


class ModelCacheError(RuntimeError):
    """The requested model cannot be safely loaded from the verified local cache."""


@dataclass(frozen=True, slots=True)
class ResolvedSnapshot:
    repository_id: str
    revision: str
    snapshot_path: Path
    cache_dir: Path | None
    metadata_hashes: dict[str, str]


SnapshotResolver = Callable[..., str]


def require_transformers_version(version: str | None = None) -> None:
    """Reject Transformers releases older than the Qwen3-compatible floor."""
    if version is None:
        try:
            import transformers
        except ImportError as error:
            raise ModelCacheError(
                "Transformers >=4.51 is required to load Qwen3; run `uv sync`."
            ) from error
        version = transformers.__version__
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < MIN_TRANSFORMERS_VERSION:
        raise ModelCacheError(
            f"Transformers >=4.51 is required to load Qwen3; found {version!r}. "
            "Update the environment with `uv sync`."
        )


def format_download_command(
    repository_id: str, revision: str, cache_dir: str | Path | None = None
) -> str:
    """Return an explicit opt-in command; this module never executes it."""
    parts = [
        "huggingface-cli",
        "download",
        repository_id,
        "--revision",
        revision,
    ]
    if cache_dir is not None:
        parts.extend(("--cache-dir", str(cache_dir)))
    return " ".join(shlex.quote(part) for part in parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic(
    repository_id: str,
    revision: str,
    cache_dir: str | Path | None,
    detail: str,
) -> str:
    searched = str(cache_dir) if cache_dir is not None else "the default Hugging Face cache"
    command = format_download_command(repository_id, revision, cache_dir)
    return (
        f"Cannot use cached model {repository_id} at revision {revision}; searched {searched}. "
        f"{detail}. To explicitly opt into a download, run: {command}"
    )


def verify_snapshot_files(
    snapshot_path: str | Path,
    repository_id: str,
    revision: str,
    cache_dir: str | Path | None = None,
) -> ResolvedSnapshot:
    """Verify snapshot identity and essential metadata without hashing model weights."""
    snapshot = Path(snapshot_path).expanduser().absolute()
    if not snapshot.is_dir():
        raise ModelCacheError(
            _diagnostic(
                repository_id,
                revision,
                cache_dir,
                f"resolved path is not a directory: {snapshot}",
            )
        )
    if snapshot.name != revision:
        raise ModelCacheError(
            _diagnostic(
                repository_id,
                revision,
                cache_dir,
                f"resolved snapshot commit {snapshot.name!r} does not match the requested commit",
            )
        )

    groups = {
        "configuration": ("config.json",),
        "tokenizer": ("tokenizer.json", "tokenizer_config.json"),
        "weights": (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "pytorch_model.bin",
        ),
    }
    missing = [
        label
        for label, names in groups.items()
        if not any((snapshot / name).is_file() for name in names)
    ]
    if missing:
        raise ModelCacheError(
            _diagnostic(
                repository_id,
                revision,
                cache_dir,
                f"snapshot is missing required {', '.join(missing)} files",
            )
        )

    metadata_names = {
        name
        for label, names in groups.items()
        if label != "weights"
        for name in names
        if (snapshot / name).is_file()
    }
    metadata_names.update(
        name
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")
        if (snapshot / name).is_file()
    )
    hashes = {name: _sha256(snapshot / name) for name in sorted(metadata_names)}
    return ResolvedSnapshot(
        repository_id=repository_id,
        revision=revision,
        snapshot_path=snapshot,
        cache_dir=Path(cache_dir).expanduser().absolute() if cache_dir is not None else None,
        metadata_hashes=hashes,
    )


def resolve_cached_snapshot(
    repository_id: str,
    revision: str,
    *,
    cache_dir: str | Path | None = None,
    resolver: SnapshotResolver | None = None,
) -> ResolvedSnapshot:
    """Resolve exactly one local snapshot and never retry with network access."""
    if resolver is None:
        from huggingface_hub import snapshot_download

        resolver = snapshot_download
    arguments: dict[str, Any] = {"revision": revision, "local_files_only": True}
    if cache_dir is not None:
        arguments["cache_dir"] = str(cache_dir)
    try:
        resolved = resolver(repository_id, **arguments)
    except Exception as error:
        raise ModelCacheError(
            _diagnostic(
                repository_id,
                revision,
                cache_dir,
                f"local-only resolution failed: {error}",
            )
        ) from error
    return verify_snapshot_files(resolved, repository_id, revision, cache_dir)
