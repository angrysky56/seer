"""Transactional run storage, provenance, and deterministic checkpoints."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import socket
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from seer.config import ExperimentConfig, config_digest, config_to_dict

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


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
    COMPLETE = "complete"


@dataclass(slots=True)
class RunState:
    schema_version: int
    config_digest: str
    status: RunStatus
    stages: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)


class RunLock:
    """An atomic exclusive lock that can only be recovered by explicit request."""

    def __init__(
        self,
        path: str | Path,
        *,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.event_sink = event_sink
        self.acquired = False

    def acquire(self, *, recover_stale: bool = False) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "started_at": _utc_now(),
        }
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError as error:
            if not recover_stale:
                raise RuntimeError(
                    f"run is already locked at {self.path}; pass recover_stale=True "
                    "only after confirming its writer is gone"
                ) from error
            recovered = self.path.with_name(
                f"{self.path.name}.recovered.{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
            )
            os.replace(self.path, recovered)
            event = {
                "type": "lock_recovered",
                "timestamp": _utc_now(),
                "previous_lock": recovered.name,
            }
            if self.event_sink is not None:
                self.event_sink(event)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        self.acquired = True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> RunLock:
        self.acquire()
        return self

    def __exit__(self, *_error: object) -> None:
        self.release()


@dataclass(slots=True)
class CheckpointState:
    config_digest: str
    global_step: int
    next_epoch: int
    next_batch: int
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    rng_state: dict[str, Any]
    sampler_state: dict[str, Any]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION


def capture_rng_state(
    *, generators: Mapping[str, torch.Generator] | None = None
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "generators": {
            name: generator.get_state() for name, generator in (generators or {}).items()
        },
    }


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    generators: Mapping[str, torch.Generator] | None = None,
) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda", [])
    if cuda_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    available = generators or {}
    for name, generator_state in state.get("generators", {}).items():
        if name not in available:
            raise ValueError(f"checkpoint requires missing generator: {name}")
        available[name].set_state(generator_state)


class RunStore:
    """Config-addressed run directory with immutable completion semantics."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.digest = config_digest(config)
        slug = "-".join(config.output.run_name.lower().split())
        slug = "".join(character for character in slug if character.isalnum() or character in "-_")
        self.run_dir = config.output.root / f"{slug}-{self.digest[:12]}"
        self.replaced_run: Path | None = None

    def _read_state(self) -> RunState:
        try:
            payload = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot resume invalid state: {error}") from error
        if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ValueError("cannot resume incompatible state schema")
        if payload.get("config_digest") != self.digest:
            raise ValueError("cannot resume: config digest does not match")
        return RunState(
            schema_version=payload["schema_version"],
            config_digest=payload["config_digest"],
            status=RunStatus(payload["status"]),
            stages=payload.get("stages", {}),
            events=payload.get("events", []),
        )

    def _write_state(self, state: RunState) -> None:
        payload = asdict(state)
        payload["status"] = state.status.value
        atomic_write_json(self.run_dir / "state.json", payload)

    def prepare(self, *, resume: bool = False, replace: bool = False) -> str:
        if resume and replace:
            raise ValueError("resume and replace are mutually exclusive")
        complete = self.run_dir / "COMPLETE"
        if self.run_dir.exists() and complete.exists():
            if resume:
                raise ValueError("cannot resume a complete run")
            if not replace:
                return "complete"
        replacement_event: dict[str, Any] | None = None
        if self.run_dir.exists() and replace:
            suffix = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            self.replaced_run = self.run_dir.with_name(f"{self.run_dir.name}.replaced.{suffix}")
            os.replace(self.run_dir, self.replaced_run)
            replacement_event = {
                "type": "run_replaced",
                "timestamp": _utc_now(),
                "previous_run": self.replaced_run.name,
            }
        elif self.run_dir.exists():
            if not resume:
                raise ValueError("incomplete run exists; use resume=True or replace=True")
            self._read_state()
            return "resumed"

        self.run_dir.mkdir(parents=True)
        for directory in ("checkpoints", "artifacts", "logs"):
            (self.run_dir / directory).mkdir()
        atomic_write_json(self.run_dir / "config.json", config_to_dict(self.config))
        state = RunState(
            schema_version=MANIFEST_SCHEMA_VERSION,
            config_digest=self.digest,
            status=RunStatus.CREATED,
            events=[replacement_event] if replacement_event else [],
        )
        self._write_state(state)
        return "created"

    @staticmethod
    def save_checkpoint(path: str | Path, checkpoint: CheckpointState) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            torch.save(asdict(checkpoint), temporary)
            with temporary.open("rb+") as stream:
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def load_checkpoint(path: str | Path, *, config_digest: str) -> CheckpointState:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("checkpoint schema is incompatible")
        if payload.get("config_digest") != config_digest:
            raise ValueError("checkpoint config digest does not match")
        return CheckpointState(**payload)
