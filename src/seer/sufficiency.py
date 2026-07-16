"""Natural class sufficiency reporting without adaptive repair."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Literal

from seer.evidence import FailureRecord, GenerationRecord, ScoredResult


@dataclass(frozen=True, slots=True)
class SufficiencyCounts:
    domain: str
    regime: str
    partition: str
    correct: int
    incorrect: int
    ambiguous: int
    invalid_source: int
    generation_failure: int
    truncation: int
    status: Literal["eligible", "underpowered"]


@dataclass(frozen=True, slots=True)
class SufficiencyReport:
    schema_version: int
    groups: tuple[SufficiencyCounts, ...]
    status: Literal["eligible", "underpowered"]
    recommendation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_sufficiency_report(
    generations: Iterable[GenerationRecord], scores: Iterable[ScoredResult],
    failures: Iterable[FailureRecord], metadata: dict[str, tuple[str, str]],
) -> SufficiencyReport:
    """Count only natural confirmatory scores; corruptions are absent by construction."""
    generations = tuple(generations)
    scores = tuple(scores)
    failures = tuple(failures)
    generations_by_id = {item.generation_id: item for item in generations}
    scores_by_id = {item.generation_id: item for item in scores}
    failure_ids = {item.generation_id for item in failures if item.stage == "generation"}
    grouped: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    for generation_id, generation in generations_by_id.items():
        domain, partition = metadata[generation.example_id]
        key = (domain, generation.regime, partition)
        if generation.truncated:
            grouped[key]["truncation"] += 1
        score = scores_by_id.get(generation_id)
        if score is None:
            continue
        if score.score_status == "ambiguous":
            grouped[key]["ambiguous"] += 1
        elif score.failure_reason == "source_gold_invalid":
            grouped[key]["invalid_source"] += 1
        elif score.score_status == "scored":
            grouped[key]["correct" if score.is_correct else "incorrect"] += 1
    for failure_id in failure_ids:
        if failure_id is None:
            continue
        failure = next(item for item in failures if item.generation_id == failure_id)
        if failure.example_id in metadata:
            domain, partition = metadata[failure.example_id]
            regime = str(failure.context.get("regime", "non_thinking"))
            grouped[(domain, regime, partition)]["generation_failure"] += 1
    groups = []
    for (domain, regime, partition), counts in sorted(grouped.items()):
        eligible = (partition == "confirmatory_test" and regime == "non_thinking" and
                    counts["correct"] >= 100 and counts["incorrect"] >= 100)
        groups.append(SufficiencyCounts(
            domain, regime, partition, counts["correct"], counts["incorrect"],
            counts["ambiguous"], counts["invalid_source"], counts["generation_failure"],
            counts["truncation"], "eligible" if eligible else "underpowered"))
    status = "eligible" if groups and all(
        item.status == "eligible" for item in groups
        if item.regime == "non_thinking" and item.partition == "confirmatory_test"
    ) and any(item.regime == "non_thinking" and item.partition == "confirmatory_test"
              for item in groups) else "underpowered"
    return SufficiencyReport(1, tuple(groups), status,
                             "proceed without protocol changes" if status == "eligible" else
                             "record underpowered; do not adaptively repair")
