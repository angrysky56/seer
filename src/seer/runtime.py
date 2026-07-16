"""Transactional run storage, provenance, and deterministic checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from seer.config import ExperimentConfig, config_digest

MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_bytes(
    destination: str | Path,
    data: bytes,
    *,
    before_replace: Callable[[], None] | None = None,
) -> None:
    """Durably replace a file using a temporary sibling and directory fsync."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if before_replace is not None:
            before_replace()
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(destination: str | Path, value: Any) -> None:
    atomic_write_bytes(destination, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: str
    bytes: int
    sha256: str
    media_type: str = "application/octet-stream"
    schema_type: str | None = None


def inventory_artifact(
    run_dir: str | Path,
    artifact: str | Path,
    media_type: str = "application/octet-stream",
    schema_type: str | None = None,
) -> ArtifactRecord:
    root = Path(run_dir).resolve()
    path = Path(artifact).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact {path} is outside run directory {root}") from error
    if not path.is_file():
        raise ValueError(f"missing artifact: {relative}")
    mutable_names = {"manifest.json", "state.json", "COMPLETE", ".lock"}
    if relative.name in mutable_names or ".tmp" in relative.name:
        raise ValueError(f"mutable or self-referential file cannot be an artifact: {relative}")
    return ArtifactRecord(
        path=relative.as_posix(),
        bytes=path.stat().st_size,
        sha256=sha256_file(path),
        media_type=media_type,
        schema_type=schema_type,
    )


def validate_artifacts(run_dir: str | Path, records: Iterable[ArtifactRecord]) -> None:
    root = Path(run_dir).resolve()
    for record in records:
        path = (root / record.path).resolve()
        if root not in path.parents:
            raise ValueError(f"artifact path escapes run directory: {record.path}")
        if not path.is_file():
            raise ValueError(f"missing artifact: {record.path}")
        if path.stat().st_size != record.bytes:
            raise ValueError(f"size mismatch for artifact: {record.path}")
        if sha256_file(path) != record.sha256:
            raise ValueError(f"hash mismatch for artifact: {record.path}")


def _git_facts(root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    head = git("rev-parse", "HEAD")
    diff = git("diff", "--binary", "HEAD") if head else None
    return {
        "head": head,
        "dirty": bool(diff) if diff is not None else None,
        "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
    }


def collect_provenance(
    *,
    command: Sequence[str],
    model: Mapping[str, Any],
    datasets: Sequence[Mapping[str, Any]],
    seeds: Mapping[str, Any],
    git_root: str | Path = ".",
) -> dict[str, Any]:
    dependencies: dict[str, str] = {}
    for package in ("seer", "numpy", "torch", "transformers", "scikit-learn"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "command": list(command),
        "git": _git_facts(Path(git_root)),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        },
        "model": dict(model),
        "datasets": [dict(dataset) for dataset in datasets],
        "seeds": dict(seeds),
        "timestamps": {"started_at": _utc_now()},
    }


@dataclass(frozen=True, slots=True)
class Manifest:
    schema_version: int
    run_id: str
    status: str
    config_digest: str
    command: list[str]
    git: dict[str, Any]
    environment: dict[str, Any]
    hardware: dict[str, Any]
    model: dict[str, Any]
    datasets: list[dict[str, Any]]
    seeds: dict[str, Any]
    timestamps: dict[str, Any]
    artifacts: list[ArtifactRecord]
    events: list[dict[str, Any]] = field(default_factory=list)


def finalize_run(
    run_dir: str | Path,
    config: ExperimentConfig,
    artifacts: Sequence[ArtifactRecord],
    *,
    provenance: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]] = (),
) -> Manifest:
    root = Path(run_dir)
    if (root / "COMPLETE").exists():
        raise ValueError(f"run is already complete and immutable: {root}")
    validate_artifacts(root, artifacts)
    digest = config_digest(config)
    timestamps = dict(provenance.get("timestamps", {}))
    timestamps["completed_at"] = _utc_now()
    manifest = Manifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        run_id=root.name,
        status="complete",
        config_digest=digest,
        command=list(provenance.get("command", [])),
        git=dict(provenance.get("git", {})),
        environment=dict(provenance.get("environment", {})),
        hardware=dict(provenance.get("hardware", {})),
        model=dict(provenance.get("model", {})),
        datasets=list(provenance.get("datasets", [])),
        seeds=dict(provenance.get("seeds", {})),
        timestamps=timestamps,
        artifacts=list(artifacts),
        events=[dict(event) for event in events],
    )
    atomic_write_json(root / "manifest.json", asdict(manifest))
    validate_artifacts(root, artifacts)
    atomic_write_bytes(root / "COMPLETE", f"{digest}\n".encode())
    return manifest
