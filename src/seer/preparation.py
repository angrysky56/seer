"""Explicit-consent, fail-closed staging of pinned dataset sources."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from seer.adapters import BabiAdapter, Gsm8kAdapter, ProofWriterAdapter
from seer.config import DatasetSpec
from seer.corruptions import CorruptionRecord, encode_corruptions
from seer.evidence import TaskExample, canonical_json_bytes, decode_jsonl, encode_jsonl
from seer.partitions import (
    PartitionError,
    assign_partitions,
    audit_and_deduplicate,
    build_partition_manifest,
    manifest_dict,
)
from seer.runtime import (
    atomic_write_bytes,
    atomic_write_json,
    inventory_artifact,
    sha256_file,
    validate_artifacts,
)


class PreparationError(RuntimeError):
    """A dataset source cannot be staged without weakening its declared contract."""


BABI_REPOSITORY = "facebook/babi_qa"
BABI_REVISION = "ab3777b46c6c0d9a4513cd3b82ea6562293837a8"
BABI_ARCHIVE_SHA256 = "0364ebde659f14d11bc21744516c5ec49d3d06cb692733f66680771244998898"
KAGGLE_ZIP_SHA256 = "50e529ad9c7144a2b176802f7496f5878d53b4e95d93cb0de957517fa84b09b2"
KAGGLE_METADATA_SHA256 = "e12c94afee4985d580eba8f299065059d990bf65119e41d78a346c937f24790d"
BABI_BUILDER_SHA256 = "d73dcc678a09f5a1bba25bbb319cab04ef1fc78af9730771dbeef25d59a73fc1"
BABI_LICENSE_SHA256 = "d12f09a636365a040fd581ebab6bf018d6fe61973c6d4862f841a6aca2f53efa"
BABI_README_SHA256 = "187382307f23056d1cf6b24c4b583be61513570c1d9d473bffd65f5a59e7eca3"
BABI_MEMBERS = {
    ("en-valid-10k-qa1", "train"): (
        "qa1_train.txt", "6f3dad60b5001427caf740404d6106aaea7257dc8e75a73afd4289475bed705a",
        1800, 9000),
    ("en-valid-10k-qa1", "validation"): (
        "qa1_valid.txt", "a7a41a89ca009b00cd5869bb949f47bb21c3b5c4df424cd1795a5e6067f0562e",
        200, 1000),
    ("en-valid-10k-qa1", "test"): (
        "qa1_test.txt", "55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b",
        200, 1000),
    ("en-valid-10k-qa2", "train"): (
        "qa2_train.txt", "e074f5119e67bca02338882e5fc50616a271d824dfde977f9c643fd3fa8748c5",
        1800, 9000),
    ("en-valid-10k-qa2", "validation"): (
        "qa2_valid.txt", "cf3cb2877eada0bff7e84fe0cbbcc70fb45ea855c583fe963c6eb25c426726cc",
        200, 1000),
    ("en-valid-10k-qa2", "test"): (
        "qa2_test.txt", "eff93fb9cd81ba02eb765772cf7cc269769ed44ec71cbbbc19cfb16ee5eba0f6",
        200, 1000),
    ("en-valid-10k-qa3", "train"): (
        "qa3_train.txt", "c1994f75f434f58115bcbfedfa728917c748e80f2191ffbf21e56cd52851304a",
        1800, 9000),
    ("en-valid-10k-qa3", "validation"): (
        "qa3_valid.txt", "8da018e1b079299c25b34c62a509cde6b9820735c398357e69d1d72a1cc54537",
        200, 1000),
    ("en-valid-10k-qa3", "test"): (
        "qa3_test.txt", "17795c977100baf8188f386522ae301b62d6c1a13efc01b3aea781e588b4d57f",
        200, 1000),
}


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
    source_chain: dict[str, Any] | None = None


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
    source_chain: dict[str, Any] | None
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


def _canonical_pretty(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True).encode() + b"\n"


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
                source_chain=resolved.source_chain,
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


def prepare_data(
    specs: Iterable[DatasetSpec], root: str | Path, *, allow_download: bool = False,
    resolver: DatasetResolver | None = None, loader: DatasetLoader | None = None,
    datasets_version: str | None = None,
    corruptions: Iterable[CorruptionRecord] = (),
    before_complete: Any | None = None,
    babi_archive: str | Path | None = None,
    babi_metadata: str | Path | None = None,
) -> None:
    """Stage sources, audit them, and publish a generation-eligible corpus atomically."""
    destination = Path(root)
    if (babi_archive is None) != (babi_metadata is None):
        raise PreparationError("--babi-archive and --babi-metadata must be provided together")
    if babi_archive is not None and not allow_download:
        raise PreparationError("local bAbI source requires explicit --allow-download")
    if babi_archive is not None and resolver is None and loader is None:
        backend = LocalBabiBackend(Path(babi_archive), Path(babi_metadata))
        resolver = loader = backend
    if (destination / "COMPLETE").exists():
        raise PreparationError("prepared corpus is already complete and immutable")
    if not (destination / "staging" / "dataset-lock.json").exists():
        stage_dataset_sources(specs, destination, allow_download=allow_download,
                              resolver=resolver, loader=loader,
                              datasets_version=datasets_version)
    staged = load_and_verify_staging(destination)
    try:
        assigned = assign_partitions(staged)
        kept, quarantined, audit = audit_and_deduplicate(assigned)
    except PartitionError as error:
        raise PreparationError(str(error)) from error
    temporary = Path(tempfile.mkdtemp(prefix=".publication-", dir=destination))
    try:
        shutil.copy2(destination / "staging" / "dataset-lock.json",
                     temporary / "dataset-lock.json")
        shards: dict[tuple[str, str], list[TaskExample]] = defaultdict(list)
        for item in kept:
            shards[(item.domain, item.partition)].append(item)
        for (domain, partition), records in sorted(shards.items()):
            atomic_write_bytes(temporary / "examples" / f"{domain}-{partition}.jsonl",
                               encode_jsonl(tuple(records)))
        atomic_write_bytes(temporary / "quarantine" / "conflicting-gold.jsonl",
                           encode_jsonl(quarantined))
        corruption_records = tuple(corruptions)
        atomic_write_bytes(temporary / "corruptions" / "fixtures.jsonl",
                           encode_corruptions(corruption_records))
        artifact_hashes = {
            path.relative_to(temporary).as_posix(): sha256_file(path)
            for path in sorted(temporary.rglob("*")) if path.is_file()
        }
        manifest = build_partition_manifest(kept, artifact_hashes)
        atomic_write_json(temporary / "partition-manifest.json", manifest_dict(manifest))
        atomic_write_json(temporary / "leakage-audit.json", manifest_dict(audit))
        artifacts = [inventory_artifact(temporary, path, schema_type=path.name)
                     for path in sorted(temporary.rglob("*")) if path.is_file()]
        atomic_write_json(temporary / "manifest.json", {
            "schema_version": 1,
            "status": "complete",
            "counts": {"examples": len(kept), "quarantined": len(quarantined),
                       "corruptions": len(corruption_records)},
            "artifacts": [asdict(item) for item in artifacts],
            "leakage_audit_sha256": sha256_file(temporary / "leakage-audit.json"),
        })
        validate_artifacts(temporary, artifacts)
        if audit.content_overlaps or audit.group_overlaps:
            raise PreparationError("protected overlap survived publication validation")
        if before_complete is not None:
            before_complete(temporary)
        validate_artifacts(temporary, artifacts)
        # Promote payloads individually only after the complete temporary tree validates.
        for source in sorted(temporary.rglob("*")):
            if source.is_file():
                relative = source.relative_to(temporary)
                atomic_write_bytes(destination / relative, source.read_bytes())
        # Re-read final payload hashes before exposing generation eligibility.
        final_manifest = json.loads((destination / "manifest.json").read_text())
        from seer.runtime import ArtifactRecord
        validate_artifacts(destination, [ArtifactRecord(**item)
                                         for item in final_manifest["artifacts"]])
        atomic_write_bytes(destination / "COMPLETE",
                           canonical_json_bytes({"dataset_lock": sha256_file(
                               destination / "dataset-lock.json"),
                               "manifest": sha256_file(destination / "manifest.json")}) + b"\n")
    except Exception:
        # COMPLETE is the sole eligibility boundary; partial payloads remain ineligible.
        (destination / "COMPLETE").unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


class HuggingFaceDatasetBackend:
    """Production backend, imported and invoked only beyond the consent boundary."""

    def __init__(self) -> None:
        self._datasets: dict[tuple[str, str, str], Any] = {}

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        from datasets import get_dataset_config_info, load_dataset
        from huggingface_hub import HfApi, hf_hub_download

        if spec.repository_id == BABI_REPOSITORY and spec.requested_revision != BABI_REVISION:
            raise PreparationError("bAbI custom loader is allowed only at the frozen exact commit")

        info = HfApi().dataset_info(spec.repository_id, revision=spec.requested_revision,
                                    files_metadata=True)
        commit = info.sha
        if not commit:
            raise PreparationError("dataset revision did not resolve")
        trusted_babi = spec.repository_id == BABI_REPOSITORY and commit == BABI_REVISION
        config_kwargs = {"trust_remote_code": True} if trusted_babi else {}
        builder_hash = None
        if trusted_babi:
            builder_path = Path(hf_hub_download(
                spec.repository_id, "babi_qa.py", repo_type="dataset", revision=commit,
            ))
            builder_hash = sha256_file(builder_path)
            if len(builder_hash) != 64:
                raise PreparationError("bAbI builder code hash is unavailable")
        config_info = get_dataset_config_info(
            spec.repository_id, spec.config_name, revision=commit, **config_kwargs)
        loaded = load_dataset(
            spec.repository_id, spec.config_name, revision=commit, **config_kwargs)
        archive_hash = None
        if trusted_babi:
            checksums = getattr(next(iter(loaded.values())).info, "download_checksums", None)
            available = {
                str(fact.get("checksum"))
                for fact in (checksums or {}).values() if isinstance(fact, Mapping)
            }
            if BABI_ARCHIVE_SHA256 not in available:
                raise PreparationError("bAbI upstream archive SHA-256 is unavailable or mismatched")
            archive_hash = BABI_ARCHIVE_SHA256
        self._datasets[(spec.repository_id, spec.config_name, commit)] = loaded
        files = tuple(DatasetSourceFile(item.rfilename, item.lfs.sha256)
                      for item in (info.siblings or []) if item.lfs and item.lfs.sha256)
        splits = {
            name: int(value["num_examples"] if isinstance(value, Mapping)
                      else value.num_examples)
            for name, value in config_info.splits.items()
        }
        features = config_info.features.to_dict()
        fingerprint = "|".join(str(loaded[name]._fingerprint) for name in sorted(loaded))
        return ResolvedDataset(spec.repository_id, spec.requested_revision, commit,
                               spec.config_name, splits, features, spec.expected_license,
                               fingerprint, files, config_info.builder_name,
                               builder_hash, archive_hash)

    def load(self, resolved: ResolvedDataset, split: str) -> Iterable[Mapping[str, Any]]:
        key = (resolved.repository_id, resolved.config_name, resolved.resolved_revision)
        return self._datasets[key][split]


class LocalBabiBackend:
    """Hash-locked local bAbI repack parser; never executes archive code."""

    def __init__(self, archive: Path, metadata: Path) -> None:
        if sha256_file(archive) != KAGGLE_ZIP_SHA256:
            raise PreparationError("local bAbI archive hash mismatch")
        if sha256_file(metadata) != KAGGLE_METADATA_SHA256:
            raise PreparationError("local bAbI metadata hash mismatch")
        facts = json.loads(metadata.read_text())
        if facts.get("version") != 1 or "CC BY 3.0" not in facts.get("license", {}).get("name", ""):
            raise PreparationError("local bAbI metadata version/license mismatch")
        self.archive, self.metadata, self.delegate = archive, metadata, HuggingFaceDatasetBackend()
        self._rows: dict[tuple[str, str], tuple[Mapping[str, Any], ...]] = {}
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)):
                raise PreparationError("local bAbI archive contains duplicate members")
            if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
                raise PreparationError("local bAbI archive contains unsafe member paths")
            self._verify_member(bundle, "tasks_1-20_v1-2/LICENSE.txt", BABI_LICENSE_SHA256)
            self._verify_member(bundle, "tasks_1-20_v1-2/README.txt", BABI_README_SHA256)
            for (config, split), (filename, digest, stories, questions) in BABI_MEMBERS.items():
                path = f"tasks_1-20_v1-2/en-valid-10k/{filename}"
                payload = self._verify_member(bundle, path, digest)
                rows = self._parse(payload, config)
                if len(rows) != stories or sum(
                    bool(line.get("answer")) for row in rows for line in row["story"]
                ) != questions:
                    raise PreparationError(f"local bAbI member count mismatch: {path}")
                self._rows[(config, split)] = rows

    @staticmethod
    def _verify_member(bundle: zipfile.ZipFile, path: str, expected: str) -> bytes:
        try:
            payload = bundle.read(path)
        except KeyError as error:
            raise PreparationError(f"local bAbI member missing: {path}") from error
        if hashlib.sha256(payload).hexdigest() != expected:
            raise PreparationError(f"local bAbI member hash mismatch: {path}")
        return payload

    @staticmethod
    def _parse(payload: bytes, config: str) -> tuple[Mapping[str, Any], ...]:
        try:
            lines = payload.decode("utf-8").splitlines()
        except UnicodeDecodeError as error:
            raise PreparationError("local bAbI member is not UTF-8") from error
        rows, story = [], []
        task_no = int(config.rsplit("qa", 1)[1])
        for raw in lines:
            fields = raw.split("\t")
            head = fields[0].split(" ", 1)
            if len(head) != 2 or not head[0].isdigit() or len(fields) not in (1, 3):
                raise PreparationError("malformed local bAbI line")
            line_id, text = int(head[0]), head[1]
            if line_id == 1 and story:
                rows.append({"id": str(len(rows)), "task_no": task_no, "story": story})
                story = []
            if fields[1:]:
                supporting = fields[2].split()
                if any(not item.isdigit() or int(item) >= line_id for item in supporting):
                    raise PreparationError("invalid local bAbI supporting IDs")
                story.append({"id": str(line_id), "text": text, "answer": fields[1],
                              "supporting_ids": supporting})
            else:
                story.append({"id": str(line_id), "text": text, "answer": "",
                              "supporting_ids": []})
        if story:
            rows.append({"id": str(len(rows)), "task_no": task_no, "story": story})
        return tuple(rows)

    def resolve(self, spec: DatasetSpec) -> ResolvedDataset:
        if spec.repository_id != BABI_REPOSITORY:
            return self.delegate.resolve(spec)
        if spec.requested_revision != BABI_REVISION:
            raise PreparationError("local bAbI source revision mismatch")
        splits = {split: BABI_MEMBERS[(spec.config_name, split)][2] for split in spec.splits}
        member_facts = {split: {"path": BABI_MEMBERS[(spec.config_name, split)][0],
                                "sha256": BABI_MEMBERS[(spec.config_name, split)][1],
                                "story_count": BABI_MEMBERS[(spec.config_name, split)][2],
                                "question_count": BABI_MEMBERS[(spec.config_name, split)][3]}
                        for split in spec.splits}
        chain = {"source_kind": "kaggle-local-repack", "kaggle_dataset_version": 1,
                 "metadata_sha256": KAGGLE_METADATA_SHA256,
                 "repack_archive_sha256": KAGGLE_ZIP_SHA256,
                 "upstream_declared_hash": BABI_ARCHIVE_SHA256,
                 "license_sha256": BABI_LICENSE_SHA256, "readme_sha256": BABI_README_SHA256,
                 "members": member_facts}
        fingerprint = hashlib.sha256(canonical_json_bytes(chain)).hexdigest()
        return ResolvedDataset(BABI_REPOSITORY, BABI_REVISION, BABI_REVISION, spec.config_name,
                               splits, {"story": "structured-list"}, "CC-BY-3.0", fingerprint,
                               (), "babi_qa", BABI_BUILDER_SHA256, KAGGLE_ZIP_SHA256, chain)

    def load(self, resolved: ResolvedDataset, split: str) -> Iterable[Mapping[str, Any]]:
        if resolved.repository_id != BABI_REPOSITORY:
            return self.delegate.load(resolved, split)
        return self._rows[(resolved.config_name, split)]
