from dataclasses import replace

import pytest

from seer.evidence import TaskExample
from seer.partitions import (
    GoldScorer,
    PartitionError,
    assign_partitions,
    audit_and_deduplicate,
    build_partition_manifest,
    protected_generation_view,
)


def example(row: str, *, domain="gsm8k", split="train", group=None, prompt=None, gold="1"):
    return TaskExample(row.zfill(64), domain, f"source/{domain}", "a" * 40, "main", split,
                       row, group or row, "signal_train", "v1", {"question": prompt or row},
                       prompt or row, gold, gold, "rational", {}, "0" * 64, "0" * 64, "MIT")


def test_partition_is_order_seed_process_and_regime_independent():
    rows = [example(str(i)) for i in range(20)]
    first = assign_partitions(rows)
    second = assign_partitions(reversed(rows))
    assert first == second
    assert build_partition_manifest(first) == build_partition_manifest(second)
    assert {item.partition for item in first} <= {"signal_train", "model_selection", "calibration"}


def test_official_partition_policies_and_group_assignment():
    test = assign_partitions([example("1", split="test")])[0]
    proof = assign_partitions([example("2", domain="proofwriter", split="train")])[0]
    assert test.partition == "confirmatory_test" and proof.partition == "signal_train"
    pair = assign_partitions([example("3", domain="babi", split="validation", group="story"),
                              example("4", domain="babi", split="validation", group="story")])
    assert pair[0].partition == pair[1].partition


def test_duplicate_collapse_conflict_quarantine_and_overlap_failure():
    same_a = example("1", prompt="same")
    same_b = example("2", prompt="same")
    assigned = assign_partitions([same_a, same_b])
    # Force the fixture into one partition to isolate deterministic collapse.
    assigned = tuple(replace(item, partition="signal_train") for item in assigned)
    kept, quarantine, audit = audit_and_deduplicate(assigned)
    assert len(kept) == 1 and not quarantine and audit.duplicates[0].multiplicity == 2
    conflict = assign_partitions([example("3", prompt="conflict", gold="1"),
                                  example("4", prompt="conflict", gold="2")])
    conflict = tuple(replace(item, partition="signal_train") for item in conflict)
    kept, quarantine, audit = audit_and_deduplicate(conflict)
    assert not kept and len(quarantine) == 2 and len(audit.conflicts) == 1
    leaked = [replace(assigned[0], partition="signal_train"),
              replace(assigned[1], partition="confirmatory_test")]
    with pytest.raises(PartitionError, match="overlap"):
        audit_and_deduplicate(leaked)


def test_protected_generation_view_has_no_gold_and_scorer_is_scoped():
    item = assign_partitions([example("1")])[0]
    view = next(protected_generation_view([item]))
    assert not hasattr(view, "gold_normalized") and not hasattr(view, "gold_raw")
    assert GoldScorer([item]).gold_for(item.example_id) == "1"
