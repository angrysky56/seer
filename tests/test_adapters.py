import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from seer.adapters import AdapterError, BabiAdapter, Gsm8kAdapter, ProofWriterAdapter
from seer.config import DatasetSpec, load_config
from seer.preparation import (
    BABI_ARCHIVE_SHA256,
    BABI_REVISION,
    DatasetSourceFile,
    HuggingFaceDatasetBackend,
    LocalBabiBackend,
    PreparationError,
    ResolvedDataset,
    load_and_verify_staging,
    prepare_data,
    stage_dataset_sources,
)


def test_dataset_backend_accepts_mapping_shaped_split_metadata(monkeypatch):
    """Datasets 3.6 exposes config-info split facts as plain mappings."""
    class Info:
        sha = "a" * 40
        siblings = []

    class ConfigInfo:
        splits = {"train": {"num_examples": 2}, "test": {"num_examples": 1}}
        features = type("Features", (), {"to_dict": lambda self: {"question": "string"}})()
        builder_name = "parquet"

    loaded = {
        "train": type("Split", (), {"_fingerprint": "train-fp"})(),
        "test": type("Split", (), {"_fingerprint": "test-fp"})(),
    }
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", lambda *args, **kwargs: Info())
    monkeypatch.setattr("datasets.get_dataset_config_info", lambda *args, **kwargs: ConfigInfo())
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: loaded)
    spec = DatasetSpec("gsm8k", "openai/gsm8k", "main", "a" * 40,
                       ["train", "test"], "MIT", {"train": 2, "test": 1})
    resolved = HuggingFaceDatasetBackend().resolve(spec)
    assert resolved.splits == {"train": 2, "test": 1}


def test_babi_remote_code_is_exact_commit_scoped_and_hash_locked(monkeypatch, tmp_path):
    builder = tmp_path / "babi_qa.py"
    builder.write_text("# exact pinned loader\n")
    calls = []

    class Info:
        sha = BABI_REVISION
        siblings = []

    class ConfigInfo:
        splits = {"train": {"num_examples": 1}}
        features = type("Features", (), {"to_dict": lambda self: {"story": "list"}})()
        builder_name = "babi_qa"

    split_info = type("SplitInfo", (), {
        "download_checksums": {"archive": {"checksum": BABI_ARCHIVE_SHA256}}})()
    loaded = {"train": type("Split", (), {"_fingerprint": "fp", "info": split_info})()}
    monkeypatch.setattr("huggingface_hub.HfApi.dataset_info", lambda *args, **kwargs: Info())
    monkeypatch.setattr("huggingface_hub.hf_hub_download",
                        lambda *args, **kwargs: str(builder))

    def config_info(*args, **kwargs):
        calls.append(("info", kwargs))
        return ConfigInfo()

    def load(*args, **kwargs):
        calls.append(("load", kwargs))
        return loaded

    monkeypatch.setattr("datasets.get_dataset_config_info", config_info)
    monkeypatch.setattr("datasets.load_dataset", load)
    spec = DatasetSpec("babi", "facebook/babi_qa", "en-valid-10k-qa1", BABI_REVISION,
                       ["train"], "CC-BY-3.0", {"train": 1})
    resolved = HuggingFaceDatasetBackend().resolve(spec)
    assert all(call[1]["trust_remote_code"] is True for call in calls)
    assert resolved.builder_hash == hashlib.sha256(builder.read_bytes()).hexdigest()
    assert resolved.archive_hash == BABI_ARCHIVE_SHA256


def test_babi_remote_code_rejects_every_other_revision():
    spec = DatasetSpec("babi", "facebook/babi_qa", "en-valid-10k-qa1", "a" * 40,
                       ["train"], "CC-BY-3.0", {})
    with pytest.raises(PreparationError, match="only at the frozen exact commit"):
        HuggingFaceDatasetBackend().resolve(spec)


def test_local_babi_backend_verifies_and_parses_without_custom_code(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"version": 1, "license": {"name": "CC BY 3.0"}}))
    payloads = {
        "tasks_1-20_v1-2/LICENSE.txt": b"license",
        "tasks_1-20_v1-2/README.txt": b"readme",
        "tasks_1-20_v1-2/en-valid-10k/qa1_train.txt":
            b"1 Mary went to the hall.\n2 Where is Mary?\thall\t1\n",
    }
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, value in payloads.items():
            bundle.writestr(name, value)
    monkeypatch.setattr("seer.preparation.KAGGLE_ZIP_SHA256",
                        hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr("seer.preparation.KAGGLE_METADATA_SHA256",
                        hashlib.sha256(metadata.read_bytes()).hexdigest())
    monkeypatch.setattr("seer.preparation.BABI_LICENSE_SHA256",
                        hashlib.sha256(payloads["tasks_1-20_v1-2/LICENSE.txt"]).hexdigest())
    monkeypatch.setattr("seer.preparation.BABI_README_SHA256",
                        hashlib.sha256(payloads["tasks_1-20_v1-2/README.txt"]).hexdigest())
    member = payloads["tasks_1-20_v1-2/en-valid-10k/qa1_train.txt"]
    monkeypatch.setattr("seer.preparation.BABI_MEMBERS", {
        ("en-valid-10k-qa1", "train"):
            ("qa1_train.txt", hashlib.sha256(member).hexdigest(), 1, 1)})
    backend = LocalBabiBackend(archive, metadata)
    resolved = backend.resolve(DatasetSpec(
        "babi", "facebook/babi_qa", "en-valid-10k-qa1", BABI_REVISION,
        ["train"], "CC-BY-3.0", {"train": 1}))
    rows = tuple(backend.load(resolved, "train"))
    assert rows[0]["story"][1]["answer"] == "hall"
    assert resolved.source_chain["source_kind"] == "kaggle-local-repack"


def test_local_babi_backend_rejects_unsafe_archive_member(monkeypatch, tmp_path):
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"version": 1, "license": {"name": "CC BY 3.0"}}))
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape", b"bad")
    monkeypatch.setattr("seer.preparation.KAGGLE_ZIP_SHA256",
                        hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr("seer.preparation.KAGGLE_METADATA_SHA256",
                        hashlib.sha256(metadata.read_bytes()).hexdigest())
    with pytest.raises(PreparationError, match="unsafe member"):
        LocalBabiBackend(archive, metadata)


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
    assert item.gold_normalized == expected and item.group_id == "test:t"


def test_babi_uses_only_prior_context_and_groups_story() -> None:
    rows = [{"id": "s1", "task_no": 1, "story": [
        {"text": "Mary went to kitchen.", "answer": ""},
        {"text": "Where is Mary?", "answer": "kitchen", "supporting_ids": [1]},
        {"text": "Mary went to garden.", "answer": ""},
        {"text": "Where is Mary now?", "answer": "garden", "supporting_ids": [3]},
    ]}]
    first, second = BabiAdapter().adapt(rows, spec("babi", "en-valid-10k-qa1"), "train")
    assert first.group_id == second.group_id == "train:s1"
    assert "garden" not in first.prompt_text
    assert second.prompt_payload["context"][-1] == "Mary went to garden."


def test_source_local_group_ids_are_qualified_by_official_split() -> None:
    row = {"id": "s1", "task_no": 1, "story": [
        {"text": "Mary went to kitchen.", "answer": ""},
        {"text": "Where is Mary?", "answer": "kitchen", "supporting_ids": [1]},
    ]}
    adapter = BabiAdapter()
    train = next(adapter.adapt([row], spec("babi", "en-valid-10k-qa1"), "train"))
    test = next(adapter.adapt([row], spec("babi", "en-valid-10k-qa1"), "test"))
    assert train.group_id == "train:s1" and test.group_id == "test:s1"
    assert train.group_fingerprint != test.group_fingerprint


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


def test_prepare_data_publication_is_complete_and_transactional(tmp_path: Path) -> None:
    item = spec("gsm8k")
    backend = FakeBackend(resolved_for(item), {"train": [
        {"id": "1", "question": "1+1?", "answer": "#### 2"},
    ]})
    prepare_data([item], tmp_path, allow_download=True, resolver=backend, loader=backend,
                 datasets_version="3.6.0")
    assert (tmp_path / "COMPLETE").is_file()
    assert (tmp_path / "dataset-lock.json").is_file()
    assert (tmp_path / "partition-manifest.json").is_file()
    assert (tmp_path / "leakage-audit.json").is_file()
    assert list((tmp_path / "examples").glob("*.jsonl"))
    assert (tmp_path / "quarantine/conflicting-gold.jsonl").is_file()
    assert (tmp_path / "corruptions/fixtures.jsonl").is_file()


def test_prepare_data_interruption_never_marks_complete(tmp_path: Path) -> None:
    item = spec("gsm8k")
    backend = FakeBackend(resolved_for(item), {"train": [
        {"id": "1", "question": "1+1?", "answer": "#### 2"},
    ]})
    def interrupt(_):
        raise RuntimeError("interrupted")
    with pytest.raises(RuntimeError, match="interrupted"):
        prepare_data([item], tmp_path, allow_download=True, resolver=backend, loader=backend,
                     datasets_version="3.6.0", before_complete=interrupt)
    assert not (tmp_path / "COMPLETE").exists()
