"""Explicitly separated, provenance-rich constructed evidence fixtures."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Any, Literal

from seer.evidence import TaskExample, canonical_json_bytes

CorruptionUse = Literal["signal_training", "ablation"]


class CorruptionError(ValueError):
    """Constructed evidence was used outside its explicit capability."""


@dataclass(frozen=True, slots=True)
class CorruptionRecord:
    corruption_id: str
    base_example_id: str
    strategy: str
    strategy_version: int
    seed: int
    parameters: dict[str, Any]
    before_hash: str
    after_hash: str
    intended_use: CorruptionUse
    generator_code_revision: str
    validation_status: Literal["validated", "invalid"]
    corrupted_prompt_text: str
    schema_version: int = 1


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def make_corruption(example: TaskExample, *, strategy: str, seed: int,
                    intended_use: CorruptionUse = "signal_training",
                    generator_code_revision: str = "fixture-v1") -> CorruptionRecord:
    if example.partition == "confirmatory_test":
        raise CorruptionError("confirmatory_test corruptions are forbidden")
    rng = random.Random(seed)
    if strategy == "context-line-shuffle":
        lines = example.prompt_text.splitlines()
        rng.shuffle(lines)
        changed = "\n".join(lines)
        parameters: dict[str, Any] = {"line_count": len(lines)}
    elif strategy == "answer-replacement":
        replacement = f"fixture-answer-{rng.randrange(1_000_000)}"
        changed = example.prompt_text + f"\nFINAL: {replacement}"
        parameters = {"replacement": replacement}
    else:
        raise CorruptionError(f"unsupported fixture strategy: {strategy}")
    identity = {"base_example_id": example.example_id, "strategy": strategy,
                "strategy_version": 1, "seed": seed, "parameters": parameters,
                "intended_use": intended_use, "generator_code_revision": generator_code_revision}
    corruption_id = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    return CorruptionRecord(corruption_id, example.example_id, strategy, 1, seed, parameters,
                            _hash(example.prompt_text), _hash(changed), intended_use,
                            generator_code_revision, "validated", changed)


def encode_corruptions(records: Iterable[CorruptionRecord]) -> bytes:
    ordered = sorted(records, key=lambda item: item.corruption_id)
    return b"".join(canonical_json_bytes({"record_type": "corruption", **asdict(item)}) + b"\n"
                    for item in ordered)


def decode_corruptions(data: bytes | str) -> tuple[CorruptionRecord, ...]:
    text = data.decode() if isinstance(data, bytes) else data
    result = []
    expected = set(CorruptionRecord.__dataclass_fields__)
    for line in text.splitlines():
        payload = json.loads(line)
        if payload.pop("record_type", None) != "corruption" or set(payload) != expected:
            raise CorruptionError("invalid corruption record")
        result.append(CorruptionRecord(**payload))
    return tuple(result)


def natural_examples(records: Iterable[TaskExample | CorruptionRecord], *,
                     include_corruptions: bool = False) -> Iterator[TaskExample | CorruptionRecord]:
    for item in records:
        if isinstance(item, CorruptionRecord):
            if not include_corruptions:
                raise CorruptionError("corruption rejected from natural iterator")
            yield item
        elif item.corruption is not None:
            raise CorruptionError("natural example has non-null corruption provenance")
        else:
            yield item
