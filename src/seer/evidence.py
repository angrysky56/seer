"""Strict canonical scientific evidence records for real-task experiments."""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import MISSING, asdict, dataclass, fields
from typing import Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

Domain = Literal["gsm8k", "proofwriter", "babi"]
Partition = Literal["signal_train", "model_selection", "calibration", "confirmatory_test"]
Regime = Literal["non_thinking", "thinking"]
ScoreStatus = Literal["scored", "invalid", "ambiguous"]
FailureStage = Literal["dataset", "prompt", "generation", "scoring"]
FailureCode = Literal[
    "dataset_resolution_failed", "schema_mismatch", "source_gold_invalid",
    "prompt_over_budget", "generation_error", "empty_generation", "missing_final_answer",
    "multiple_conflicting_answers", "invalid_answer_type", "thinking_parse_error",
    "token_budget_exhausted", "scoring_error",
]
SCHEMA_VERSION = 1


class EvidenceError(ValueError):
    """Evidence wire-format error with a field-specific diagnostic."""


@dataclass(frozen=True, slots=True)
class TaskExample:
    example_id: str
    domain: Domain
    dataset_id: str
    dataset_revision: str
    dataset_config: str
    source_split: str
    source_row_id: str
    group_id: str
    partition: Partition
    prompt_template_id: str
    prompt_payload: dict[str, Any]
    prompt_text: str
    gold_raw: str
    gold_normalized: str
    answer_type: str
    adapter_metadata: dict[str, Any]
    content_fingerprint: str
    group_fingerprint: str
    license_id: str
    corruption: None = None
    schema_version: int = SCHEMA_VERSION
    record_type: ClassVar[str] = "task_example"


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    generation_id: str
    example_id: str
    model_id: str
    model_revision: str
    tokenizer_revision: str
    chat_template_hash: str
    prompt_token_ids_hash: str
    prompt_token_count: int
    regime: Regime
    thinking_enabled: bool
    do_sample: bool
    seed: int
    temperature: float | None
    top_p: float | None
    top_k: int | None
    min_p: float | None
    repetition_penalty: float | None
    presence_penalty: float | None
    max_new_tokens: int
    stopping_token_ids: tuple[int, ...]
    padding_token_id: int | None
    eos_token_ids: tuple[int, ...]
    dtype: str
    device: str
    raw_generation: str
    raw_generated_token_ids: tuple[int, ...] | None
    generated_token_artifact: str | None
    generated_token_count: int
    finish_reason: str
    truncated: bool
    thinking_text: str | None
    answer_text: str | None
    failure_id: str | None
    schema_version: int = SCHEMA_VERSION
    record_type: ClassVar[str] = "generation"


@dataclass(frozen=True, slots=True)
class ScoredResult:
    generation_id: str
    prediction_raw: str
    prediction_normalized: str | None
    gold_normalized: str | None
    is_correct: bool | None
    score_status: ScoreStatus
    normalizer_id: str
    normalizer_version: int
    candidate_spans: tuple[str, ...]
    failure_reason: FailureCode | None
    schema_version: int = SCHEMA_VERSION
    record_type: ClassVar[str] = "scored_result"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    record_id: str
    stage: FailureStage
    code: FailureCode
    message: str
    retryable: bool
    example_id: str | None
    generation_id: str | None
    context: dict[str, Any]
    schema_version: int = SCHEMA_VERSION
    record_type: ClassVar[str] = "failure"


EvidenceRecord = TaskExample | GenerationRecord | ScoredResult | FailureRecord
_RECORD_TYPES = {item.record_type: item for item in (TaskExample, GenerationRecord, ScoredResult, FailureRecord)}


def _canonical_identity(kind: str, identity: dict[str, Any]) -> str:
    material = canonical_json_bytes({"identity_type": kind, **identity})
    return hashlib.sha256(material).hexdigest()


def example_id(**source_identity: str) -> str:
    """Hash a canonical dataset source identity, independent of mapping order."""
    required = {"dataset_id", "dataset_revision", "dataset_config", "source_split", "source_row_id"}
    if set(source_identity) != required:
        raise EvidenceError(f"example identity fields: expected {sorted(required)}")
    return _canonical_identity("example-v1", source_identity)


def generation_id(**generation_identity: Any) -> str:
    """Hash canonical prompt, model, regime, seed, and decoding identities."""
    required = {"example_id", "model_id", "model_revision", "tokenizer_revision", "regime", "seed",
                "prompt_token_ids_hash", "generation_parameters"}
    if set(generation_identity) != required:
        raise EvidenceError(f"generation identity fields: expected {sorted(required)}")
    return _canonical_identity("generation-v1", generation_identity)


def record_id(*, stage: str, code: str, example_id: str | None,
              generation_id: str | None, context: dict[str, Any]) -> str:
    return _canonical_identity("failure-v1", {"stage": stage, "code": code,
                                               "example_id": example_id,
                                               "generation_id": generation_id, "context": context})


def _json_value(value: Any) -> Any:
    if hasattr(value, "record_type"):
        return {"record_type": value.record_type, **asdict(value)}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a record or JSON value in the project's canonical compact form."""
    return json.dumps(_json_value(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _decode_value(value: Any, annotation: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        choices = [choice for choice in args if choice is not type(None)]
        if len(choices) == 1:
            return _decode_value(value, choices[0], path)
    if origin is Literal:
        if value not in args:
            raise EvidenceError(f"{path}: expected one of {args}, got {value!r}")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise EvidenceError(f"{path}: expected array")
        return tuple(_decode_value(item, args[0], f"{path}[{index}]")
                     for index, item in enumerate(value))
    if origin is dict:
        if not isinstance(value, dict):
            raise EvidenceError(f"{path}: expected object")
        return value
    if annotation in (str, int, bool):
        if type(value) is not annotation:
            raise EvidenceError(f"{path}: expected {annotation.__name__}")
    if annotation is float and (not isinstance(value, int | float) or isinstance(value, bool)):
        raise EvidenceError(f"{path}: expected number")
    return float(value) if annotation is float else value


def decode_record(data: bytes | str) -> EvidenceRecord:
    try:
        payload = json.loads(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise EvidenceError(f"record: invalid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceError("record: expected object")
    record_type = payload.pop("record_type", None)
    if record_type not in _RECORD_TYPES:
        raise EvidenceError(f"record_type: unsupported value {record_type!r}")
    cls = _RECORD_TYPES[record_type]
    definitions = {field.name: field for field in fields(cls)}
    unknown = set(payload) - set(definitions)
    if unknown:
        raise EvidenceError(f"{sorted(unknown)[0]}: unknown field")
    hints = get_type_hints(cls)
    decoded: dict[str, Any] = {}
    for name, definition in definitions.items():
        if name not in payload:
            if definition.default is MISSING and definition.default_factory is MISSING:
                raise EvidenceError(f"{name}: missing required field")
            continue
        decoded[name] = _decode_value(payload[name], hints[name], name)
    if decoded.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise EvidenceError(f"schema_version: unsupported version {decoded['schema_version']}")
    for identifier in (name for name in ("example_id", "generation_id", "record_id")
                       if name in decoded and decoded[name] is not None):
        value = decoded[identifier]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise EvidenceError(f"{identifier}: expected lowercase SHA-256")
    return cls(**decoded)


def encode_jsonl(records: list[EvidenceRecord] | tuple[EvidenceRecord, ...]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def decode_jsonl(data: bytes | str) -> tuple[EvidenceRecord, ...]:
    lines = data.decode() if isinstance(data, bytes) else data
    return tuple(decode_record(line) for line in lines.splitlines() if line.strip())
