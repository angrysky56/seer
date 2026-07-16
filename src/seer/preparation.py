"""Explicit-consent, fail-closed staging of pinned dataset sources."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from seer.adapters import BabiAdapter, Gsm8kAdapter, ProofWriterAdapter
from seer.config import DatasetSpec
from seer.evidence import TaskExample, decode_jsonl, encode_jsonl
from seer.runtime import atomic_write_bytes, atomic_write_json, sha256_file


class PreparationError(RuntimeError):
    """A dataset source cannot be staged without weakening its declared contract."""


@dataclass(frozen=True, slots=True)
class DatasetSourceFile:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    repository_id: str
    requested_revision: str
    resolved_revision: str
    config_name: str
    splits: dict[str, int]
    features: dict[str, Any]
    license_id: str
    fingerprint: str
    files: tuple[DatasetSourceFile, ...] = ()
    builder_name: str | None = None
    builder_hash: str | None = None
    archive_hash: str | None = None


@dataclass(frozen=True, slots=True)
class LockedDataset:
    repository_id: str
    requested_revision: str
    resolved_revision: str
    config_name: str
    splits: dict[str, int]
    features: dict[str, Any]
    license_id: str
    fingerprint: str
    files: tuple[DatasetSourceFile, ...]
    builder_name: str | None
    builder_hash: str | None
    archive_hash: str | None
    normalized_shards: dict[str, str]


@dataclass(frozen=True, slots=True)
class DatasetLock:
    schema_version: int
    datasets_library_version: str
    datasets: tuple[LockedDataset, ...]


class DatasetResolver(Protocol):
    def resolve(self, spec: DatasetSpec) -> ResolvedDataset: ...


class DatasetLoader(Protocol):
    def load(self, resolved: ResolvedDataset, split: str) -> Iterable[Mapping[str, Any]]: ...


def _validate_resolution(spec: DatasetSpec, resolved: ResolvedDataset) -> None:
    if resolved.repository_id != spec.repository_id or resolved.config_name != spec.config_name:
        raise PreparationError("unsafe source substitution")
    if resolved.requested_revision != spec.requested_revision:
        raise PreparationError("requested revision mismatch")
    if len(resolved.resolved_revision) != 40 or any(
        char not in "0123456789abcdef" for char in resolved.resolved_revision
    ):
        raise PreparationError("revision did not resolve unambiguously to a full commit")
    if not resolved.resolved_revision.startswith(spec.requested_revision):
        raise PreparationError("resolved revision does not match requested pin")
    if set(resolved.splits) != set(spec.splits):
        raise PreparationError("official split mismatch")
    for split, expected in spec.expected_counts.items():
        if resolved.splits.get(split) != expected:
            raise PreparationError(f"source count mismatch for {split}")
    if resolved.license_id != spec.expected_license:
        raise PreparationError("source license mismatch")
    if not resolved.features or not resolved.fingerprint:
        raise PreparationError("missing source schema or fingerprint")
    for source_file in resolved.files:
        if len(source_file.sha256) != 64:
            raise PreparationError("invalid selected source file hash")


def _adapter(spec: DatasetSpec):
    return {"gsm8k": Gsm8kAdapter, "proofwriter": ProofWriterAdapter,
            "babi": BabiAdapter}[spec.domain]()


def _lock_dict(lock: DatasetLock) -> dict[str, Any]:
    return asdict(lock)


def load_and_verify_staging(root: str | Path) -> tuple[TaskExample, ...]:
    """Verify every normalized shard against the lock before returning records."""
    staging = Path(root) / "staging"
    try:
        payload = json.loads((staging / "dataset-lock.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise PreparationError(f"invalid dataset lock: {error}") from error
    records: list[TaskExample] = []
    for dataset in payload.get("datasets", []):
        for relative, expected in dataset.get("normalized_shards", {}).items():
            path = staging / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise PreparationError(f"normalized shard hash mismatch: {relative}")
            decoded = decode_jsonl(path.read_bytes())
            if not all(isinstance(item, TaskExample) for item in decoded):
                raise PreparationError(f"non-example record in source shard: {relative}")
            records.extend(decoded)  # type: ignore[arg-type]
    return tuple(records)


def stage_dataset_sources(
    specs: Iterable[DatasetSpec], root: str | Path, *, allow_download: bool = False,
    resolver: DatasetResolver | None = None, loader: DatasetLoader | None = None,
    datasets_version: str | None = None,
) -> DatasetLock:
    """Resolve and stage sources atomically; the boundary is inert without explicit consent."""
    if not allow_download:
        raise PreparationError("dataset download is disabled; pass --allow-download explicitly")
    if resolver is None or loader is None:
        backend = HuggingFaceDatasetBackend()
        resolver = resolver or backend
        loader = loader or backend
    destination = Path(root)
    if (destination / "COMPLETE").exists():
        raise PreparationError("source staging must not replace a completed prepared corpus")
    if (destination / "staging" / "dataset-lock.json").exists():
        load_and_verify_staging(destination)
        raise PreparationError("source staging already exists and is immutable")
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".staging-", dir=destination))
    locked: list[LockedDataset] = []
    try:
        for spec in specs:
            resolved = resolver.resolve(spec)
            _validate_resolution(spec, resolved)
            hashes: dict[str, str] = {}
            adapter = _adapter(spec)
            for split in spec.splits:
                rows = loader.load(resolved, split)
                cap = spec.sample_caps.get(split)
                examples: list[TaskExample] = []
                for item in adapter.adapt(rows, spec, split):
                    examples.append(item)
                    if cap is not None and len(examples) >= cap:
                        break
                relative = f"examples/{spec.domain}-{spec.config_name}-{split}.jsonl"
                path = temporary / relative
                atomic_write_bytes(path, encode_jsonl(tuple(examples)))
                hashes[relative] = sha256_file(path)
            locked.append(LockedDataset(
                repository_id=resolved.repository_id,
                requested_revision=resolved.requested_revision,
                resolved_revision=resolved.resolved_revision,
                config_name=resolved.config_name,
                splits=resolved.splits,
                features=resolved.features,
                license_id=resolved.license_id,
                fingerprint=resolved.fingerprint,
                files=resolved.files,
                builder_name=resolved.builder_name,
                builder_hash=resolved.builder_hash,
                archive_hash=resolved.archive_hash,
                normalized_shards=hashes,
            ))
        version = datasets_version or importlib.metadata.version("datasets")
        lock = DatasetLock(1, version, tuple(locked))
        atomic_write_json(temporary / "dataset-lock.json", _lock_dict(lock))
        temporary.replace(destination / "staging")
        return lock
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class HuggingFaceDatasetBackend:
    """Production backend, imported and invoked only beyond the consent boundary."""

    def __init__(self) -> None:
        self._datasets: dict[tuple[str, str, str], Any] = {}

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        from datasets import get_dataset_config_info, load_dataset
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(spec.repository_id, revision=spec.requested_revision,
                                    files_metadata=True)
        commit = info.sha
        if not commit:
            raise PreparationError("dataset revision did not resolve")
        config_info = get_dataset_config_info(spec.repository_id, spec.config_name, revision=commit)
        loaded = load_dataset(spec.repository_id, spec.config_name, revision=commit)
        self._datasets[(spec.repository_id, spec.config_name, commit)] = loaded
        files = tuple(DatasetSourceFile(item.rfilename, item.lfs.sha256)
                      for item in (info.siblings or []) if item.lfs and item.lfs.sha256)
        splits = {name: int(value.num_examples) for name, value in config_info.splits.items()}
        features = config_info.features.to_dict()
        fingerprint = "|".join(str(loaded[name]._fingerprint) for name in sorted(loaded))
        return ResolvedDataset(spec.repository_id, spec.requested_revision, commit,
                               spec.config_name, splits, features, spec.expected_license,
                               fingerprint, files, config_info.builder_name)

    def load(self, resolved: ResolvedDataset, split: str) -> Iterable[Mapping[str, Any]]:
        key = (resolved.repository_id, resolved.config_name, resolved.resolved_revision)
        return self._datasets[key][split]
