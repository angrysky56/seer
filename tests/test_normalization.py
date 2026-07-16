from __future__ import annotations

import pytest

from seer.evidence import GenerationRecord, TaskExample
from seer.normalization import (
    NormalizationResult,
    normalize_babi,
    normalize_gsm8k,
    normalize_proofwriter,
    score_generation,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#### 1,200", "1200"),
        ("FINAL: \\boxed{-3}", "-3"),
        ("FINAL: 0.50", "1/2"),
        ("FINAL: \\boxed{3/4}", "3/4"),
        ("FINAL: 25%", "1/4"),
        ("work\r\nFINAL: 2\r\nFINAL: 2.0", "2"),
        ("ＦＩＮＡＬ： ２", "2"),
    ],
)
def test_gsm8k_normalizes_exact_rational_forms(text: str, expected: str) -> None:
    result = normalize_gsm8k(text)
    assert result.status == "valid"
    assert result.canonical_value == expected


@pytest.mark.parametrize(
    ("text", "status", "failure"),
    [
        ("FINAL: 2\nFINAL: 3", "ambiguous", "multiple_conflicting_answers"),
        ("FINAL: 1/0", "invalid", "invalid_answer_type"),
        ("FINAL: NaN", "invalid", "invalid_answer_type"),
        ("FINAL: infinity", "invalid", "invalid_answer_type"),
        ("FINAL: 2+2", "invalid", "invalid_answer_type"),
        ("The last number happens to be 7", "invalid", "missing_final_answer"),
        ("FINAL: \\boxed{2", "invalid", "invalid_answer_type"),
        ("", "invalid", "empty_generation"),
        ("<think>work</think>", "invalid", "missing_final_answer"),
        ("<think>work FINAL: 2", "invalid", "thinking_parse_error"),
    ],
)
def test_gsm8k_rejects_ambiguous_or_malformed_forms(
    text: str, status: str, failure: str
) -> None:
    result = normalize_gsm8k(text)
    assert (result.status, result.failure_code) == (status, failure)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("true", "true"), ("ENTAILED", "true"), ("yes", "true"),
     ("false", "false"), ("contradiction", "false"), ("No", "false"),
     ("unknown", "unknown")],
)
def test_proofwriter_uses_only_declared_label_vocabulary(text: str, expected: str) -> None:
    result = normalize_proofwriter(text, source_gold=True)
    assert result.canonical_value == expected
    generated = normalize_proofwriter(f"reason\nFINAL: {text}")
    assert generated.canonical_value == expected


def test_proofwriter_distinguishes_unknown_and_conflicts() -> None:
    assert normalize_proofwriter("FINAL: unknown").canonical_value == "unknown"
    conflict = normalize_proofwriter("FINAL: true\nFINAL: false")
    assert conflict.status == "ambiguous"
    assert conflict.failure_code == "multiple_conflicting_answers"
    no_final = normalize_proofwriter("Probably true, but no final")
    assert no_final.failure_code == "missing_final_answer"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("FINAL: Kitchen.", "kitchen"), (" FINAL:\tMARY  \n", "mary"),
     ("FINAL: north-east", "north-east")],
)
def test_babi_normalizes_one_explicit_entity(text: str, expected: str) -> None:
    assert normalize_babi(text).canonical_value == expected


@pytest.mark.parametrize("text", ["FINAL: kitchen, garden", "FINAL: mary and john", "FINAL:"])
def test_babi_rejects_multi_answer_or_empty_output(text: str) -> None:
    assert normalize_babi(text).status == "invalid"


def test_result_is_typed_and_scorer_preserves_traceability() -> None:
    assert isinstance(normalize_gsm8k("FINAL: 2"), NormalizationResult)
    generation = GenerationRecord(
        "a" * 64, "b" * 64, "model", "rev", "rev", "c" * 64, "d" * 64, 1,
        "non_thinking", False, False, 0, None, None, None, None, 1.0, None, 10,
        (1,), 1, (1,), "float32", "cpu", "work\nFINAL: 2", (2,), None, 1,
        "eos", False, None, "FINAL: 2", None,
    )
    example = TaskExample(
        "b" * 64, "gsm8k", "dataset", "rev", "main", "test", "1", "1",
        "confirmatory_test", "v1", {}, "prompt", "#### 2", "2", "rational", {},
        "c" * 64, "d" * 64, "MIT", None,
    )
    score = score_generation(generation, example)
    assert score.prediction_raw == generation.raw_generation
    assert score.candidate_spans == ("2",)
    assert score.prediction_normalized == score.gold_normalized == "2"
    assert score.is_correct is True


def test_scorer_reserves_none_for_invalid_source_gold() -> None:
    result = score_generation("FINAL: 2", "not a gold marker", domain="gsm8k")
    assert result.is_correct is None
    assert result.failure_reason == "source_gold_invalid"
