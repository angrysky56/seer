"""Strict adapters from pinned public-dataset rows to canonical evidence records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from seer.config import DatasetSpec
from seer.evidence import TaskExample, canonical_json_bytes, example_id
from seer.normalization import normalize_babi, normalize_gsm8k, normalize_proofwriter

SYSTEM_PROMPT = (
    "You solve the task using only the information in the user message. Follow the requested "
    "final-answer format exactly. Do not add facts."
)


class AdapterError(ValueError):
    """A source row does not match the pinned adapter contract."""


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(row: Mapping[str, Any], fields: set[str]) -> None:
    missing = fields - set(row)
    if missing:
        raise AdapterError(f"schema mismatch: missing {sorted(missing)}")


def _make(spec: DatasetSpec, split: str, row_id: str, group_id: str, payload: dict[str, Any],
          prompt: str, gold: str, normalized: str, answer_type: str,
          metadata: dict[str, Any]) -> TaskExample:
    identity = dict(dataset_id=spec.repository_id, dataset_revision=spec.requested_revision,
                    dataset_config=spec.config_name, source_split=split, source_row_id=row_id)
    qualified_group_id = f"{split}:{group_id}"
    return TaskExample(
        example_id=example_id(**identity), domain=spec.domain, dataset_id=spec.repository_id,
        dataset_revision=spec.requested_revision, dataset_config=spec.config_name,
        source_split=split, source_row_id=row_id, group_id=qualified_group_id,
        partition="signal_train", prompt_template_id=f"{spec.domain}-v1",
        prompt_payload={"system": SYSTEM_PROMPT, **payload}, prompt_text=prompt,
        gold_raw=gold, gold_normalized=normalized, answer_type=answer_type,
        adapter_metadata=metadata,
        content_fingerprint=_hash({"prompt": prompt, "gold": normalized}),
        group_fingerprint=_hash(qualified_group_id), license_id=spec.expected_license,
    )


class Gsm8kAdapter:
    def adapt(self, rows: Iterable[Mapping[str, Any]], spec: DatasetSpec,
              split: str) -> Iterator[TaskExample]:
        for index, row in enumerate(rows):
            _require(row, {"question", "answer"})
            question, gold = row["question"], row["answer"]
            if not isinstance(question, str) or not isinstance(gold, str):
                raise AdapterError("schema mismatch: question and answer must be strings")
            result = normalize_gsm8k(gold, source_gold=True)
            if result.status != "valid" or result.canonical_value is None:
                raise AdapterError("invalid GSM8K source gold")
            row_id = str(row.get("id", index))
            prompt = (f"{question}\n\nReason step by step. End with exactly "
                      "`FINAL: \\boxed{answer}` where `answer` is only the final number.")
            yield _make(spec, split, row_id, row_id, {"question": question}, prompt, gold,
                        result.canonical_value, "rational", {"gold_marker": "####"})


class ProofWriterAdapter:
    def adapt(self, rows: Iterable[Mapping[str, Any]], spec: DatasetSpec,
              split: str) -> Iterator[TaskExample]:
        for index, row in enumerate(rows):
            _require(row, {"theory", "question", "answer"})
            theory, question, gold = row["theory"], row["question"], row["answer"]
            if not all(isinstance(item, str) for item in (theory, question, gold)):
                raise AdapterError("schema mismatch: ProofWriter text fields must be strings")
            result = normalize_proofwriter(gold, source_gold=True)
            if result.status != "valid" or result.canonical_value is None:
                raise AdapterError("invalid ProofWriter source label")
            row_id = str(row.get("id", index))
            group = str(row.get("theory_id", _hash(theory)))
            prompt = (f"Facts and rules:\n{theory}\n\nClaim:\n{question}\n\nUsing open-world "
                      "reasoning, answer whether the claim is entailed, contradicted, or unknown. "
                      "End with exactly `FINAL: true`, `FINAL: false`, or `FINAL: unknown`.")
            metadata = {key: row[key] for key in ("depth", "config") if key in row}
            yield _make(spec, split, row_id, group, {"theory": theory, "question": question},
                        prompt, gold, result.canonical_value, "categorical", metadata)


class BabiAdapter:
    def adapt(self, rows: Iterable[Mapping[str, Any]], spec: DatasetSpec,
              split: str) -> Iterator[TaskExample]:
        for row_index, row in enumerate(rows):
            _require(row, {"story"})
            story = row["story"]
            if not isinstance(story, list):
                raise AdapterError("schema mismatch: story must be a list")
            context: list[str] = []
            group = str(row.get("id", row_index))
            for line_index, line in enumerate(story):
                if not isinstance(line, Mapping):
                    raise AdapterError("schema mismatch: story line must be an object")
                text = line.get("text")
                if not isinstance(text, str):
                    raise AdapterError("schema mismatch: story text must be a string")
                answer = line.get("answer", "")
                if answer:
                    if not isinstance(answer, str):
                        raise AdapterError("invalid bAbI source gold")
                    result = normalize_babi(answer, source_gold=True)
                    if result.status != "valid" or result.canonical_value is None:
                        raise AdapterError("invalid bAbI source gold")
                    row_id = f"{group}:{line_index}"
                    prompt = ("Story:\n" + "\n".join(context) + f"\n\nQuestion:\n{text}\n\n"
                              "Answer from the story. End with exactly `FINAL: answer` and no "
                              "other text after it.")
                    metadata = {"task_no": row.get("task_no"),
                                "supporting_ids": line.get("supporting_ids", [])}
                    yield _make(spec, split, row_id, group,
                                {"context": list(context), "question": text}, prompt, answer,
                                result.canonical_value, "entity", metadata)
                else:
                    context.append(text)
