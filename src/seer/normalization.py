"""Pure, deterministic answer normalization for Phase 2 reasoning domains."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from seer.evidence import Domain, FailureCode, GenerationRecord, ScoredResult, TaskExample

NormalizationStatus = Literal["valid", "invalid", "ambiguous"]
NORMALIZER_VERSION = 1


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    candidate_spans: tuple[str, ...]
    canonical_value: str | None
    status: NormalizationStatus
    normalizer_id: str
    normalizer_version: int = NORMALIZER_VERSION
    failure_code: FailureCode | None = None


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def _answer_text(text: str) -> tuple[str | None, FailureCode | None]:
    cleaned = _clean(text)
    if not cleaned:
        return None, "empty_generation"
    has_tag = "<think" in cleaned.lower() or "</think>" in cleaned.lower()
    if has_tag:
        blocks = re.findall(r"<think>.*?</think>", cleaned, flags=re.IGNORECASE | re.DOTALL)
        starts = len(re.findall(r"<think>", cleaned, flags=re.IGNORECASE))
        ends = len(re.findall(r"</think>", cleaned, flags=re.IGNORECASE))
        if starts != ends or starts != len(blocks):
            return None, "thinking_parse_error"
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned,
                         flags=re.IGNORECASE | re.DOTALL).strip()
    return cleaned, None


def _markers(text: str, *, source_gold: bool) -> tuple[tuple[str, ...], FailureCode | None]:
    answer, failure = _answer_text(text)
    if failure is not None:
        return (), failure
    assert answer is not None
    if source_gold:
        hash_matches = re.findall(r"####\s*([^\n]+)", answer)
        spans = tuple(span.strip() for span in hash_matches) if hash_matches else (answer.strip(),)
        return spans, None
    matches = re.findall(r"(?:^|\n)\s*(?:FINAL\s*:|####)\s*([^\n]*)",
                         answer, flags=re.IGNORECASE)
    if not matches:
        return (), "missing_final_answer"
    return tuple(span.strip() for span in matches), None


def _invalid(normalizer_id: str, spans: tuple[str, ...], code: FailureCode,
             *, ambiguous: bool = False) -> NormalizationResult:
    return NormalizationResult(spans, None, "ambiguous" if ambiguous else "invalid",
                               normalizer_id, failure_code=code)


_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/[+-]?\d+)?%?")


def _rational(span: str) -> Fraction | None:
    candidate = span.strip()
    boxed = re.fullmatch(r"\\boxed\{([^{}]+)\}", candidate)
    if boxed:
        candidate = boxed.group(1).strip()
    elif "\\boxed" in candidate:
        return None
    if not _NUMBER.fullmatch(candidate):
        return None
    percent = candidate.endswith("%")
    if percent:
        candidate = candidate[:-1]
    candidate = candidate.replace(",", "")
    try:
        value = Fraction(candidate)
    except (ValueError, ZeroDivisionError):
        return None
    return value / 100 if percent else value


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def normalize_gsm8k(text: str, *, source_gold: bool = False) -> NormalizationResult:
    spans, failure = _markers(text, source_gold=source_gold)
    if failure is not None:
        return _invalid("gsm8k", spans, failure)
    values = [_rational(span) for span in spans]
    if any(value is None for value in values):
        return _invalid("gsm8k", spans, "invalid_answer_type")
    unique = set(values)
    if len(unique) != 1:
        return _invalid("gsm8k", spans, "multiple_conflicting_answers", ambiguous=True)
    value = next(iter(unique))
    assert value is not None
    return NormalizationResult(spans, _fraction_text(value), "valid", "gsm8k")


_PROOF_LABELS = {
    "true": "true", "entailed": "true", "entailment": "true", "yes": "true",
    "false": "false", "contradicted": "false", "contradiction": "false", "no": "false",
    "unknown": "unknown",
}


def normalize_proofwriter(text: str, *, source_gold: bool = False) -> NormalizationResult:
    spans, failure = _markers(text, source_gold=source_gold)
    if failure is not None:
        return _invalid("proofwriter", spans, failure)
    values = [_PROOF_LABELS.get(span.strip().casefold().rstrip(".")) for span in spans]
    if any(value is None for value in values):
        return _invalid("proofwriter", spans, "invalid_answer_type")
    unique = set(values)
    if len(unique) != 1:
        return _invalid("proofwriter", spans, "multiple_conflicting_answers", ambiguous=True)
    return NormalizationResult(spans, next(iter(unique)), "valid", "proofwriter")


def normalize_babi(text: str, *, source_gold: bool = False) -> NormalizationResult:
    spans, failure = _markers(text, source_gold=source_gold)
    if failure is not None:
        return _invalid("babi", spans, failure)
    values: list[str] = []
    for span in spans:
        value = " ".join(span.casefold().strip().rstrip(".!?").split())
        if not value or "," in value or re.search(r"\s+(?:and|or)\s+", value):
            return _invalid("babi", spans, "invalid_answer_type")
        if not re.fullmatch(r"[\w-]+(?: [\w-]+)*", value):
            return _invalid("babi", spans, "invalid_answer_type")
        values.append(value)
    unique = set(values)
    if len(unique) != 1:
        return _invalid("babi", spans, "multiple_conflicting_answers", ambiguous=True)
    return NormalizationResult(spans, next(iter(unique)), "valid", "babi")


def _normalizer(domain: Domain):
    return {"gsm8k": normalize_gsm8k, "proofwriter": normalize_proofwriter,
            "babi": normalize_babi}[domain]


def score_generation(
    generation: GenerationRecord | str,
    example: TaskExample | str,
    *,
    domain: Domain | None = None,
) -> ScoredResult:
    """Normalize a generation and source gold into a traceable score record."""
    if isinstance(generation, GenerationRecord):
        generation_id = generation.generation_id
        prediction_raw = generation.raw_generation
    else:
        prediction_raw = generation
        generation_id = hashlib.sha256(_clean(generation).encode()).hexdigest()
    if isinstance(example, TaskExample):
        selected_domain = example.domain
        gold_raw = example.gold_raw
    else:
        if domain is None:
            raise ValueError("domain is required when scoring raw strings")
        selected_domain, gold_raw = domain, example
    normalize = _normalizer(selected_domain)
    gold = normalize(gold_raw, source_gold=True)
    prediction = normalize(prediction_raw)
    if gold.status != "valid":
        return ScoredResult(generation_id, prediction_raw, prediction.canonical_value, None, None,
                            "invalid", prediction.normalizer_id, prediction.normalizer_version,
                            prediction.candidate_spans, "source_gold_invalid")
    status = "scored" if prediction.status == "valid" else prediction.status
    correct = prediction.status == "valid" and prediction.canonical_value == gold.canonical_value
    return ScoredResult(generation_id, prediction_raw, prediction.canonical_value,
                        gold.canonical_value, correct, status, prediction.normalizer_id,
                        prediction.normalizer_version, prediction.candidate_spans,
                        prediction.failure_code)
