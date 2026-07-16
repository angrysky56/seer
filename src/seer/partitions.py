"""Deterministic protected partitions and fail-closed evidence leakage audits."""

from __future__ import annotations

import hashlib
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, replace
from typing import Any

from seer.evidence import Partition, TaskExample, canonical_json_bytes

PARTITION_ALGORITHM_ID = "seer-partition-v1"
UINT64_SPACE = 1 << 64
THRESHOLDS = {"gsm8k": (70 * UINT64_SPACE // 100, 85 * UINT64_SPACE // 100),
              "development_50_50": (UINT64_SPACE // 2,)}


class PartitionError(ValueError):
    """Prepared evidence violates a protected partition invariant."""


def audit_normalize(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("\r\n", "\n")
    text = " ".join(text.split())
    return text.rstrip(" .!?;:")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def content_fingerprint(example: TaskExample, *, include_gold: bool = True) -> str:
    value: dict[str, Any] = {"domain": example.domain,
                             "inputs": audit_normalize(example.prompt_text)}
    if include_gold:
        value["gold"] = audit_normalize(example.gold_normalized)
    return _fingerprint(value)


def group_fingerprint(example: TaskExample) -> str:
    return _fingerprint({"domain": example.domain,
                         "dataset_config": example.dataset_config,
                         "group": audit_normalize(example.group_id)})


def partition_hash(example: TaskExample) -> int:
    material = (f"{PARTITION_ALGORITHM_ID}\0{example.dataset_revision}\0{example.domain}\0"
                f"{group_fingerprint(example)}").encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big", signed=False)


def assign_partition(example: TaskExample) -> Partition:
    split = example.source_split.casefold()
    if split == "test":
        return "confirmatory_test"
    value = partition_hash(example)
    if example.domain == "gsm8k" and split == "train":
        low, high = THRESHOLDS["gsm8k"]
        if value < low:
            return "signal_train"
        return "model_selection" if value < high else "calibration"
    if example.domain in {"proofwriter", "babi"} and split in {"validation", "valid"}:
        return "model_selection" if value < THRESHOLDS["development_50_50"][0] else "calibration"
    if split == "train":
        return "signal_train"
    policy = f"{example.domain}/{example.source_split}"
    raise PartitionError(f"unsupported official split policy: {policy}")


def assign_partitions(examples: Iterable[TaskExample]) -> tuple[TaskExample, ...]:
    assigned = [replace(item, partition=assign_partition(item),
                        content_fingerprint=content_fingerprint(item),
                        group_fingerprint=group_fingerprint(item)) for item in examples]
    return tuple(sorted(assigned, key=lambda item: item.example_id))


@dataclass(frozen=True, slots=True)
class DuplicateFact:
    kept_example_id: str
    example_ids: tuple[str, ...]
    multiplicity: int


@dataclass(frozen=True, slots=True)
class ConflictFact:
    input_fingerprint: str
    example_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeakageAudit:
    schema_version: int
    duplicates: tuple[DuplicateFact, ...]
    conflicts: tuple[ConflictFact, ...]
    content_overlaps: tuple[str, ...]
    group_overlaps: tuple[str, ...]
    duplicate_source_ids: tuple[str, ...]
    cross_domain_prompt_overlaps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PartitionManifest:
    schema_version: int
    algorithm_id: str
    thresholds: dict[str, tuple[int, ...]]
    counts: dict[str, int]
    group_hashes: dict[str, tuple[str, ...]]
    artifact_hashes: dict[str, str]


def audit_and_deduplicate(examples: Iterable[TaskExample]) -> tuple[
        tuple[TaskExample, ...], tuple[TaskExample, ...], LeakageAudit]:
    items = tuple(sorted(examples, key=lambda item: (item.dataset_id, item.dataset_config,
                                                      item.source_split, item.source_row_id,
                                                      item.example_id)))
    by_input: dict[str, list[TaskExample]] = defaultdict(list)
    by_content: dict[str, list[TaskExample]] = defaultdict(list)
    by_group: dict[str, list[TaskExample]] = defaultdict(list)
    by_source: dict[tuple[str, str, str, str], list[TaskExample]] = defaultdict(list)
    by_prompt: dict[str, list[TaskExample]] = defaultdict(list)
    for item in items:
        by_input[content_fingerprint(item, include_gold=False)].append(item)
        by_content[item.content_fingerprint].append(item)
        by_group[item.group_fingerprint].append(item)
        by_source[(item.dataset_id, item.dataset_config, item.source_split,
                   item.source_row_id)].append(item)
        by_prompt[_fingerprint(audit_normalize(item.prompt_text))].append(item)
    conflicts = {key: values for key, values in by_input.items()
                 if len({item.gold_normalized for item in values}) > 1}
    quarantined_ids = {item.example_id for values in conflicts.values() for item in values}
    duplicates: list[DuplicateFact] = []
    kept: list[TaskExample] = []
    for values in by_content.values():
        usable = [item for item in values if item.example_id not in quarantined_ids]
        if not usable:
            continue
        partitions = {item.partition for item in usable}
        if len(partitions) > 1:
            continue
        winner = min(usable, key=lambda item: (item.dataset_id, item.dataset_config,
                                               item.source_split, item.source_row_id,
                                               item.example_id))
        kept.append(winner)
        if len(usable) > 1:
            duplicates.append(DuplicateFact(winner.example_id,
                                             tuple(sorted(item.example_id for item in usable)),
                                             len(usable)))
    content_overlaps = tuple(sorted(key for key, values in by_content.items()
                                    if len({item.partition for item in values}) > 1))
    group_overlaps = tuple(sorted(key for key, values in by_group.items()
                                  if len({item.partition for item in values}) > 1))
    audit = LeakageAudit(
        1, tuple(sorted(duplicates, key=lambda item: item.kept_example_id)),
        tuple(ConflictFact(key, tuple(sorted(item.example_id for item in values)))
              for key, values in sorted(conflicts.items())), content_overlaps, group_overlaps,
        tuple(sorted("|".join(key) for key, values in by_source.items() if len(values) > 1)),
        tuple(sorted(key for key, values in by_prompt.items()
                     if len({item.domain for item in values}) > 1)),
    )
    if content_overlaps or group_overlaps:
        raise PartitionError("protected partition content/group overlap")
    quarantined = tuple(sorted((item for item in items if item.example_id in quarantined_ids),
                               key=lambda item: item.example_id))
    return tuple(sorted(kept, key=lambda item: item.example_id)), quarantined, audit


def build_partition_manifest(examples: Iterable[TaskExample],
                             artifact_hashes: dict[str, str] | None = None) -> PartitionManifest:
    counts: dict[str, int] = defaultdict(int)
    groups: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        counts[item.partition] += 1
        groups[item.partition].add(item.group_fingerprint)
    return PartitionManifest(1, PARTITION_ALGORITHM_ID, THRESHOLDS,
                             dict(sorted(counts.items())),
                             {key: tuple(sorted(value)) for key, value in sorted(groups.items())},
                             dict(sorted((artifact_hashes or {}).items())))


@dataclass(frozen=True, slots=True)
class ProtectedExample:
    example_id: str
    domain: str
    partition: Partition
    prompt_text: str
    prompt_payload: dict[str, Any]
    metadata: dict[str, Any]


class GoldScorer:
    def __init__(self, examples: Iterable[TaskExample]) -> None:
        self.__gold = {item.example_id: item.gold_normalized for item in examples}

    def gold_for(self, example_id: str) -> str:
        return self.__gold[example_id]


def protected_generation_view(examples: Iterable[TaskExample]) -> Iterator[ProtectedExample]:
    for item in examples:
        yield ProtectedExample(item.example_id, item.domain, item.partition, item.prompt_text,
                               item.prompt_payload, dict(item.adapter_metadata))


def manifest_dict(value: PartitionManifest | LeakageAudit) -> dict[str, Any]:
    return asdict(value)
