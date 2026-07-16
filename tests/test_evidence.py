from __future__ import annotations

import json
from dataclasses import fields

import pytest

from seer.data import SyntheticStateTrackingDataset
from seer.evidence import (
    EvidenceError,
    FailureRecord,
    GenerationRecord,
    ScoredResult,
    TaskExample,
    canonical_json_bytes,
    decode_record,
    encode_jsonl,
    example_id,
    generation_id,
)


def example() -> TaskExample:
    return TaskExample(
        example_id=example_id(
            dataset_id="openai/gsm8k", dataset_revision="a" * 40,
            dataset_config="main", source_split="test", source_row_id="17",
        ),
        domain="gsm8k", dataset_id="openai/gsm8k", dataset_revision="a" * 40,
        dataset_config="main", source_split="test", source_row_id="17", group_id="17",
        partition="confirmatory_test", prompt_template_id="gsm8k-v1",
        prompt_payload={"question": "What is 1 + 1?"}, prompt_text="What is 1 + 1?",
        gold_raw="#### 2", gold_normalized="2", answer_type="rational",
        adapter_metadata={"extraction": "hash-marker"}, content_fingerprint="b" * 64,
        group_fingerprint="c" * 64, license_id="MIT", corruption=None,
    )


def generation(item: TaskExample) -> GenerationRecord:
    identity = {
        "example_id": item.example_id, "model_id": "Qwen/Qwen3-0.6B",
        "model_revision": "d" * 40, "tokenizer_revision": "d" * 40,
        "regime": "non_thinking", "seed": 7, "prompt_token_ids_hash": "e" * 64,
        "generation_parameters": {"do_sample": False, "max_new_tokens": 256},
    }
    return GenerationRecord(
        generation_id=generation_id(**identity), example_id=item.example_id,
        model_id="Qwen/Qwen3-0.6B", model_revision="d" * 40,
        tokenizer_revision="d" * 40, chat_template_hash="f" * 64,
        prompt_token_ids_hash="e" * 64, prompt_token_count=21,
        regime="non_thinking", thinking_enabled=False, do_sample=False, seed=7,
        temperature=None, top_p=None, top_k=None, min_p=None, repetition_penalty=1.0,
        presence_penalty=None, max_new_tokens=256, stopping_token_ids=(151643,),
        padding_token_id=151643, eos_token_ids=(151643,), dtype="float16", device="cuda:0",
        raw_generation="FINAL: \\boxed{2}", raw_generated_token_ids=(1, 2, 3),
        generated_token_artifact=None, generated_token_count=3, finish_reason="eos",
        truncated=False, thinking_text=None, answer_text="FINAL: \\boxed{2}", failure_id=None,
    )


def test_all_records_round_trip_with_canonical_byte_stability() -> None:
    item = example()
    output = generation(item)
    records = [
        item,
        output,
        ScoredResult(output.generation_id, output.raw_generation, "2", "2", True,
                     "scored", "gsm8k", 1, ("2",), None),
        FailureRecord("a" * 64, "generation", "generation_error", "failed", True,
                      item.example_id, output.generation_id, {"attempt": 1}),
    ]
    for record in records:
        wire = canonical_json_bytes(record)
        assert canonical_json_bytes(decode_record(wire)) == wire
    assert encode_jsonl(records).endswith(b"\n")


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda value: value.pop("domain"), "domain"),
        (lambda value: value.update({"surprise": 1}), "surprise"),
        (lambda value: value.update({"schema_version": 99}), "schema_version"),
        (lambda value: value.update({"partition": "secret"}), "partition"),
        (lambda value: value.update({"example_id": "not-a-hash"}), "example_id"),
    ],
)
def test_decoder_rejects_invalid_wire_fields(mutation, field: str) -> None:
    payload = json.loads(canonical_json_bytes(example()))
    mutation(payload)
    with pytest.raises(EvidenceError, match=field):
        decode_record(json.dumps(payload).encode())


def test_ids_are_order_independent_and_identity_sensitive() -> None:
    identity = dict(dataset_id="d", dataset_revision="r", dataset_config="c",
                    source_split="s", source_row_id="1")
    assert example_id(**identity) == example_id(**dict(reversed(list(identity.items()))))
    assert example_id(**identity) != example_id(**(identity | {"source_row_id": "2"}))
    item = example()
    first = generation(item)
    changed = generation_id(
        example_id=item.example_id, model_id=first.model_id, model_revision=first.model_revision,
        tokenizer_revision=first.tokenizer_revision, regime=first.regime, seed=8,
        prompt_token_ids_hash=first.prompt_token_ids_hash,
        generation_parameters={"do_sample": False, "max_new_tokens": 256},
    )
    assert changed != first.generation_id


def test_records_exclude_later_scientific_fields_and_preserve_tensor_smoke_interface() -> None:
    forbidden = {"energy", "hidden_state", "baseline", "calibration", "metric", "auroc"}
    for record_type in (TaskExample, GenerationRecord, ScoredResult, FailureRecord):
        assert forbidden.isdisjoint({field.name for field in fields(record_type)})
    sample = SyntheticStateTrackingDataset("synthetic", 1, seq_len=4, seed=3)[0]
    assert sample.input_ids.shape == sample.step_targets.shape == sample.step_mask.shape
