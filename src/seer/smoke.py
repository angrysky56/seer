"""Offline synthetic experiment composed through the production run contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from seer.config import ExperimentConfig
from seer.data import Domain, SyntheticStateTrackingDataset
from seer.model import SeerPathAModel
from seer.optim import build_optimizer
from seer.runtime import (
    CheckpointState,
    RunLock,
    RunStore,
    atomic_write_bytes,
    atomic_write_json,
    collect_provenance,
    finalize_run,
    inventory_artifact,
    restore_rng_state,
    validate_artifacts,
)
from seer.train import TrainCheckpoint, TrainCursor, seed_everything, train_loop

RESULT_SCHEMA_VERSION = 1


class SmokeInterrupted(RuntimeError):
    """Internal test seam representing process interruption after durable checkpointing."""


class _SyntheticConfig:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class _SyntheticOutput:
    def __init__(self, logits: Tensor, hidden_states: tuple[Tensor, ...]) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class SyntheticBaseLM(nn.Module):
    """Tiny package-owned causal-LM-shaped object; never imports Transformers."""

    def __init__(self, vocab_size: int = 2, hidden_size: int = 8) -> None:
        super().__init__()
        self.config = _SyntheticConfig(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> _SyntheticOutput:
        del attention_mask, output_hidden_states
        embedded = self.embed_tokens(input_ids)
        hidden = torch.tanh(self.proj(embedded))
        return _SyntheticOutput(self.lm_head(hidden), (embedded, hidden))


@dataclass(frozen=True, slots=True)
class ResultEnvelope:
    schema_version: int
    stage: str
    status: str
    records: list[dict[str, int | float]]
    diagnostics: dict[str, Any]


def canonical_scientific_hash(records: list[dict[str, int | float]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_complete(store: RunStore) -> None:
    manifest_path = store.run_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot validate completed run manifest: {error}") from error
    if manifest.get("config_digest") != store.digest:
        raise ValueError("completed run config digest does not match")
    from seer.runtime import ArtifactRecord

    validate_artifacts(
        store.run_dir, [ArtifactRecord(**record) for record in manifest.get("artifacts", [])]
    )
    marker = (store.run_dir / "COMPLETE").read_text(encoding="utf-8").strip()
    if marker != store.digest:
        raise ValueError("COMPLETE marker config digest does not match")


def _write_results(run_dir: Path, records: list[dict[str, int | float]], status: str) -> None:
    envelope = ResultEnvelope(
        schema_version=RESULT_SCHEMA_VERSION,
        stage="smoke",
        status=status,
        records=records,
        diagnostics={"scientific_sha256": canonical_scientific_hash(records)},
    )
    atomic_write_json(run_dir / "artifacts" / "results.json", asdict(envelope))
    lines = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in records
    )
    atomic_write_bytes(run_dir / "artifacts" / "steps.jsonl", lines)


def run_smoke(
    config: ExperimentConfig,
    *,
    resume: bool = False,
    replace: bool = False,
    interrupt_after_checkpoint: int | None = None,
) -> str:
    """Run or resume the deterministic synthetic CPU experiment."""
    if config.runtime.backend != "synthetic":
        raise ValueError("smoke requires runtime.backend='synthetic'")
    store = RunStore(config)
    disposition = store.prepare(resume=resume, replace=replace)
    if disposition == "complete":
        _validate_complete(store)
        return "complete"

    records: list[dict[str, int | float]] = []
    cursor = TrainCursor()
    data_order_state: dict[str, Any] | None = None
    seed = config.train.seeds[0]
    seed_everything(seed)
    model = SeerPathAModel(SyntheticBaseLM(), config.model)
    optimizer = build_optimizer(model, **asdict(config.optim))
    checkpoint_path = store.run_dir / "checkpoints" / "latest.pt"
    if disposition == "resumed":
        checkpoint = store.load_checkpoint(checkpoint_path, config_digest=store.digest)
        model.load_state_dict(checkpoint.model_state)
        optimizer.load_state_dict(checkpoint.optimizer_state)
        restore_rng_state(checkpoint.rng_state)
        cursor = TrainCursor(
            checkpoint.global_step, checkpoint.next_epoch, checkpoint.next_batch
        )
        data_order_state = checkpoint.sampler_state
        payload = json.loads((store.run_dir / "artifacts" / "results.json").read_text())
        records = payload["records"]

    dataset = SyntheticStateTrackingDataset(
        Domain("synthetic"), num_examples=8, seq_len=8, seed=config.seeds.generation
    )
    lock = RunLock(store.run_dir / ".lock")
    lock.acquire()
    try:
        def checkpoint_callback(training: TrainCheckpoint) -> None:
            checkpoint = CheckpointState(
                config_digest=store.digest,
                global_step=training.cursor.global_step,
                next_epoch=training.cursor.next_epoch,
                next_batch=training.cursor.next_batch,
                model_state=training.model_state,
                optimizer_state=training.optimizer_state,
                rng_state=training.rng_state,
                sampler_state=training.data_order_state,
            )
            store.save_checkpoint(checkpoint_path, checkpoint)
            checkpoint_records = records + training.scientific_records
            _write_results(store.run_dir, checkpoint_records, "interrupted")
            if interrupt_after_checkpoint == training.cursor.global_step:
                raise SmokeInterrupted(f"interrupted after step {training.cursor.global_step}")

        new_records = train_loop(
            model,
            optimizer,
            dataset,
            config.train,
            seed=seed,
            cursor=cursor,
            data_order_state=data_order_state,
            checkpoint_interval=config.runtime.checkpoint_interval,
            checkpoint_callback=checkpoint_callback,
        )
        records.extend(record.to_dict() for record in new_records)
        # The callback runs before train_loop returns, so persist the final accumulated records.
        _write_results(store.run_dir, records, "complete")
        artifacts = [
            inventory_artifact(store.run_dir, store.run_dir / "config.json", "application/json"),
            inventory_artifact(
                store.run_dir,
                store.run_dir / "artifacts" / "results.json",
                "application/json",
                "seer.result-envelope.v1",
            ),
            inventory_artifact(
                store.run_dir,
                store.run_dir / "artifacts" / "steps.jsonl",
                "application/x-ndjson",
                "seer.scientific-step.v1",
            ),
            inventory_artifact(store.run_dir, checkpoint_path, "application/x-pytorch"),
        ]
        provenance = collect_provenance(
            command=["seer", "smoke"],
            model={"identifier": "package-owned-synthetic", "revision": "builtin-v1"},
            datasets=[{"identifier": "synthetic-parity", "revision": "builtin-v1"}],
            seeds={"training": seed, "generation": config.seeds.generation},
        )
        finalize_run(store.run_dir, config, artifacts, provenance=provenance)
        return "complete"
    finally:
        lock.release()
