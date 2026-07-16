import json
from pathlib import Path

import pytest

from seer.adapters import AdapterError, BabiAdapter, Gsm8kAdapter, ProofWriterAdapter
from seer.config import DatasetSpec, load_config
from seer.preparation import (
    DatasetSourceFile,
    PreparationError,
    ResolvedDataset,
    load_and_verify_staging,
    stage_dataset_sources,
)


def spec(domain: str, config: str = "main") -> DatasetSpec:
    return DatasetSpec(
        domain, f"source/{domain}", config, "a" * 40, ["train"], "test", {"train": 1}
    )


def test_gsm8k_adapter_preserves_identity_and_normalizes_gold() -> None:
    item = next(Gsm8kAdapter().adapt(
        [{"id": "7", "question": "What is 2+2?", "answer": "work\n#### 4"}],
        spec("gsm8k"), "train"))
    assert item.source_row_id == "7" and item.gold_normalized == "4"
    assert item.prompt_template_id == "gsm8k-v1" and "\\boxed{answer}" in item.prompt_text


@pytest.mark.parametrize("label, expected", [("true", "true"), ("false", "false"),
                                               ("unknown", "unknown")])
def test_proofwriter_fixture_verifies_labels(label: str, expected: str) -> None:
    item = next(ProofWriterAdapter().adapt(
        [{"id": "p", "theory_id": "t", "theory": "Blue(a).", "question": "Blue(a)?",
          "answer": label, "depth": 1}], spec("proofwriter", "default"), "test"))
    assert item.gold_normalized == expected and item.group_id == "t"


def test_babi_uses_only_prior_context_and_groups_story() -> None:
    rows = [{"id": "s1", "task_no": 1, "story": [
        {"text": "Mary went to kitchen.", "answer": ""},
        {"text": "Where is Mary?", "answer": "kitchen", "supporting_ids": [1]},
        {"text": "Mary went to garden.", "answer": ""},
        {"text": "Where is Mary now?", "answer": "garden", "supporting_ids": [3]},
    ]}]
    first, second = BabiAdapter().adapt(rows, spec("babi", "en-valid-10k-qa1"), "train")
    assert first.group_id == second.group_id == "s1"
    assert "garden" not in first.prompt_text
    assert second.prompt_payload["context"][-1] == "Mary went to garden."


def test_schema_and_invalid_gold_fail_closed() -> None:
    with pytest.raises(AdapterError, match="schema mismatch"):
        next(Gsm8kAdapter().adapt([{"question": "x"}], spec("gsm8k"), "train"))
    with pytest.raises(AdapterError, match="invalid"):
        next(ProofWriterAdapter().adapt(
            [{"theory": "x", "question": "y", "answer": "maybe"}],
            spec("proofwriter"), "train"))


def test_evidence_config_has_exact_research_pins() -> None:
    config = load_config(Path(__file__).parents[1] / "examples/evidence.json")
    assert [item.requested_revision for item in config.datasets] == [
        "740312a", "761ca6eedf37f1c27a4eeb88cd5107ada469a4ec",
        "ab3777b46c6c0d9a4513cd3b82ea6562293837a8",
        "ab3777b46c6c0d9a4513cd3b82ea6562293837a8",
        "ab3777b46c6c0d9a4513cd3b82ea6562293837a8",
    ]
    assert json.loads((Path(__file__).parents[1] / "examples/evidence.json").read_text())


class FakeBackend:
    def __init__(self, resolved, rows):
        self.resolved, self.rows, self.resolve_calls, self.load_calls = resolved, rows, 0, []

    def resolve(self, spec):
        self.resolve_calls += 1
        return self.resolved

    def load(self, resolved, split):
        self.load_calls.append(split)
        yield from self.rows[split]


def resolved_for(item: DatasetSpec) -> ResolvedDataset:
    return ResolvedDataset(item.repository_id, item.requested_revision, item.requested_revision,
                           item.config_name, {"train": 1}, {"question": "string",
                           "answer": "string"}, item.expected_license, "fingerprint",
                           (DatasetSourceFile("train.parquet", "b" * 64),))


def test_prepare_refuses_before_calling_resolver(tmp_path: Path) -> None:
    item = spec("gsm8k")
    backend = FakeBackend(resolved_for(item), {})
    with pytest.raises(PreparationError, match="allow-download"):
        stage_dataset_sources([item], tmp_path, resolver=backend, loader=backend)
    assert backend.resolve_calls == 0


def test_prepare_stages_lock_hashes_caps_and_never_completes(tmp_path: Path) -> None:
    item = spec("gsm8k")
    item.sample_caps["train"] = 1
    backend = FakeBackend(resolved_for(item), {"train": [
        {"id": "1", "question": "1+1?", "answer": "#### 2"},
        {"id": "2", "question": "2+2?", "answer": "#### 4"},
    ]})
    lock = stage_dataset_sources([item], tmp_path, allow_download=True,
                                 resolver=backend, loader=backend, datasets_version="3.6.0")
    assert len(load_and_verify_staging(tmp_path)) == 1
    assert lock.datasets_library_version == "3.6.0"
    assert lock.datasets[0].resolved_revision == "a" * 40
    assert not (tmp_path / "COMPLETE").exists()
    shard = next((tmp_path / "staging/examples").iterdir())
    shard.write_text(shard.read_text() + "{}\n")
    with pytest.raises(PreparationError, match="hash mismatch"):
        load_and_verify_staging(tmp_path)


def test_prepare_rejects_ambiguous_revision_and_split_drift(tmp_path: Path) -> None:
    item = spec("gsm8k")
    bad = ResolvedDataset(item.repository_id, item.requested_revision, "abc", item.config_name,
                          {"other": 1}, {"x": "string"}, item.expected_license, "fp")
    backend = FakeBackend(bad, {})
    with pytest.raises(PreparationError, match="full commit"):
        stage_dataset_sources([item], tmp_path, allow_download=True,
                              resolver=backend, loader=backend, datasets_version="3.6.0")
