from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

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
    capture_rng_state,
    CheckpointState,
    collect_provenance,
    finalize_run,
    inventory_artifact,
    restore_rng_state,
    RunLock,
    RunStore,
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
    record = inventory_artifact(
        tmp_path, artifact, media_type="application/json", schema_type="result"
    )
    validate_artifacts(tmp_path, [record])

    artifact.write_bytes(b'[]\n')
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
    artifact.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        finalize_run(run_dir, config, [record], provenance={})
    assert not (run_dir / "COMPLETE").exists()


def test_lock_contention_and_deliberate_recovery_are_auditable(tmp_path: Path) -> None:
    events: list[dict[str, object]] = []
    first = RunLock(tmp_path / ".lock", event_sink=events.append)
    first.acquire()
    with pytest.raises(RuntimeError, match="already locked"):
        RunLock(tmp_path / ".lock").acquire()
    first.release()

    lock_path = tmp_path / ".lock"
    lock_path.write_text('{"pid": 99999999, "hostname": "stale", "started_at": "old"}')
    recovered = RunLock(lock_path, event_sink=events.append)
    with pytest.raises(RuntimeError, match="recover_stale"):
        recovered.acquire()
    recovered.acquire(recover_stale=True)
    recovered.release()

    assert any(event["type"] == "lock_recovered" for event in events)
    assert list(tmp_path.glob(".lock.recovered.*"))


def test_completed_run_noop_replace_and_resume_compatibility(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    store = RunStore(config)
    assert store.prepare() == "created"
    original = (store.run_dir / "config.json").read_bytes()
    atomic_write_bytes(store.run_dir / "COMPLETE", f"{config_digest(config)}\n".encode())

    assert store.prepare() == "complete"
    assert (store.run_dir / "config.json").read_bytes() == original
    with pytest.raises(ValueError, match="complete"):
        store.prepare(resume=True)

    assert store.prepare(replace=True) == "created"
    assert store.replaced_run is not None and store.replaced_run.exists()
    events = json.loads((store.run_dir / "state.json").read_text())["events"]
    assert events[0]["type"] == "run_replaced"

    resumable = RunStore(config)
    assert resumable.prepare(resume=True) == "resumed"
    state = json.loads((resumable.run_dir / "state.json").read_text())
    state["config_digest"] = "wrong"
    (resumable.run_dir / "state.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="config digest"):
        resumable.prepare(resume=True)


def test_checkpoint_rng_and_position_round_trip(tmp_path: Path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    generator = torch.Generator().manual_seed(19)
    rng = capture_rng_state(generators={"sampler": generator})
    checkpoint = CheckpointState(
        config_digest="digest",
        global_step=12,
        next_epoch=3,
        next_batch=4,
        model_state={"weight": torch.tensor([1.0])},
        optimizer_state={"step": 12},
        rng_state=rng,
        sampler_state={"permutation": [2, 0, 1], "offset": 2},
    )
    path = tmp_path / "checkpoints" / "step-12.pt"
    RunStore.save_checkpoint(path, checkpoint)

    expected = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    loaded = RunStore.load_checkpoint(path, config_digest="digest")
    restore_rng_state(loaded.rng_state, generators={"sampler": generator})
    actual = (random.random(), np.random.random(), torch.rand(1), torch.rand(1, generator=generator))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert torch.equal(actual[3], expected[3])
    assert (loaded.next_epoch, loaded.next_batch) == (3, 4)
    assert loaded.sampler_state["offset"] == 2

    with pytest.raises(ValueError, match="config digest"):
        RunStore.load_checkpoint(path, config_digest="other")
