from __future__ import annotations

import json
from pathlib import Path

import pytest

from seer.config import (
    DatasetConfig,
    EnergyConfig,
    ExperimentConfig,
    ModelConfig,
    OutputConfig,
    SeedConfig,
    config_digest,
)
from seer.runtime import (
    atomic_write_bytes,
    collect_provenance,
    finalize_run,
    inventory_artifact,
    validate_artifacts,
)


def make_config(root: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="runtime-test",
        model=ModelConfig(base_model_name="fake/model", revision="abc123"),
        dataset=DatasetConfig(
            name="fake/data",
            revision="data123",
            train_split="train",
            calibration_split="validation",
            test_splits=["test"],
        ),
        seeds=SeedConfig(training=[1, 2, 3], generation=4, bootstrap=5),
        energy=EnergyConfig(negative_strategies=["shuffled"]),
        output=OutputConfig(root=root, run_name="runtime"),
    )


def test_atomic_failure_never_promotes_partial_file(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.bin"

    with pytest.raises(RuntimeError, match="injected"):
        atomic_write_bytes(
            destination,
            b"partial",
            before_replace=lambda: (_ for _ in ()).throw(RuntimeError("injected")),
        )

    assert not destination.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_hash_inventory_detects_tamper_and_deletion(tmp_path: Path) -> None:
    artifact = tmp_path / "artifacts" / "results.json"
    atomic_write_bytes(artifact, b'{}\n')
    record = inventory_artifact(tmp_path, artifact, media_type="application/json", schema_type="result")
    validate_artifacts(tmp_path, [record])

    artifact.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_artifacts(tmp_path, [record])
    artifact.unlink()
    with pytest.raises(ValueError, match="missing artifact"):
        validate_artifacts(tmp_path, [record])


def test_finalize_manifest_has_complete_provenance_and_relative_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    run_dir = tmp_path / f"runtime-{config_digest(config)[:12]}"
    run_dir.mkdir()
    artifact = run_dir / "artifacts" / "results.json"
    atomic_write_bytes(artifact, b'{"accuracy":1}\n')
    record = inventory_artifact(run_dir, artifact, "application/json", "result")
    provenance = collect_provenance(
        command=["seer", "smoke"],
        model={"identifier": config.model.base_model_name, "revision": config.model.revision},
        datasets=[{"identifier": config.dataset.name, "revision": config.dataset.revision}],
        seeds={"training": config.seeds.training, "generation": config.seeds.generation},
        git_root=tmp_path,
    )
    manifest = finalize_run(run_dir, config, [record], provenance=provenance)

    payload = json.loads((run_dir / "manifest.json").read_text())
    assert payload["status"] == "complete"
    assert payload["config_digest"] == config_digest(config)
    assert payload["command"] == ["seer", "smoke"]
    assert payload["git"]["head"] is None
    assert payload["environment"]["python"]
    assert payload["environment"]["platform"]
    assert payload["environment"]["dependencies"]
    assert payload["hardware"]["cpu"]
    assert payload["model"]["revision"] == "abc123"
    assert payload["datasets"][0]["revision"] == "data123"
    assert payload["seeds"]["training"] == [1, 2, 3]
    assert payload["timestamps"]["completed_at"]
    assert payload["artifacts"][0]["path"] == "artifacts/results.json"
    assert payload["artifacts"][0]["sha256"] == record.sha256
    assert (run_dir / "COMPLETE").exists()
    assert manifest.status == "complete"


def test_tamper_blocks_finalization(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run_dir = tmp_path / "run"
    artifact = run_dir / "artifacts" / "results.json"
    atomic_write_bytes(artifact, b"original")
    record = inventory_artifact(run_dir, artifact)
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="hash mismatch"):
        finalize_run(run_dir, config, [record], provenance={})
    assert not (run_dir / "COMPLETE").exists()
