"""Joint task + self-certainty training loop (TRAINING.md section 4).

The single most expensive lesson in TRAINING.md: training a confidence/energy
head ONLY on its own objective, with gradient flowing into the model,
catastrophically forgets the task (measured: task accuracy 0.931 -> 0.059).
``joint_loss`` always sums a task loss anchored on clean data with the
self-certainty loss — there is deliberately no code path here that trains the
self-certainty head alone.

This module is runnable end-to-end today against
``data.SyntheticStateTrackingDataset`` and a tiny fake base model (see
``tests/test_train.py``) as a smoke test. It is NOT yet wired to a real
downloaded base model or a real domain dataset — that is experiment 1 itself
(ROADMAP.md), deliberately left for when a base model and task are chosen.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from seer.config import TrainConfig
from seer.data import DomainDataset, StateTrackingBatch, collate_state_tracking_examples
from seer.optim import SingleDeviceMuonWithAuxAdam


@dataclass(slots=True)
class LossBreakdown:
    """Components of the joint objective, kept separate for logging.

    ``task_loss`` and ``certainty_loss`` are reported individually because a
    single combined scalar hides exactly the failure mode TRAINING.md section 4
    warns about (certainty loss improving while task loss silently explodes).
    """

    total: Tensor
    task_loss: Tensor
    certainty_loss: Tensor
    task_accuracy: Tensor


def joint_loss(
    task_logits: Tensor,
    step_targets: Tensor,
    step_mask: Tensor,
    self_certainty: Tensor,
    task_loss_weight: float,
    energy_loss_weight: float,
) -> LossBreakdown:
    """Task loss (anchored) + self-certainty loss (stop-gradient target), summed.

    Args:
        task_logits: Shape ``(batch, seq_len, vocab)``.
        step_targets: Class indices, shape ``(batch, seq_len)``.
        step_mask: 1 where supervised, shape ``(batch, seq_len)``.
        self_certainty: Predicted p(correct) per position, shape ``(batch, seq_len)``.
        task_loss_weight: Weight on the task loss (the anchor).
        energy_loss_weight: Weight on the self-certainty loss.

    Returns:
        A :class:`LossBreakdown` with the combined loss and its components.
    """
    mask = step_mask.bool()
    flat_logits = task_logits[mask]
    flat_targets = step_targets[mask]
    task_loss = F.cross_entropy(flat_logits, flat_targets)

    # Stop-gradient correctness target: the model's own argmax vs. ground truth.
    # TRAINING.md section 4: targets are (argmax == truth), never backprop through this.
    with torch.no_grad():
        predicted_class = flat_logits.argmax(dim=-1)
        correctness_target = (predicted_class == flat_targets).float()

    flat_certainty = self_certainty[mask]
    certainty_loss = F.binary_cross_entropy(flat_certainty, correctness_target)

    total = task_loss_weight * task_loss + energy_loss_weight * certainty_loss
    return LossBreakdown(
        total=total,
        task_loss=task_loss.detach(),
        certainty_loss=certainty_loss.detach(),
        task_accuracy=correctness_target.mean().detach(),
    )


@dataclass(slots=True)
class TrainStepResult:
    step: int
    loss: LossBreakdown

    def to_dict(self) -> dict[str, int | float]:
        """Return the stable scientific fields, excluding tensors and timings."""
        return {
            "step": self.step,
            "total_loss": float(self.loss.total.detach().cpu()),
            "task_loss": float(self.loss.task_loss.cpu()),
            "certainty_loss": float(self.loss.certainty_loss.cpu()),
            "task_accuracy": float(self.loss.task_accuracy.cpu()),
        }


@dataclass(frozen=True, slots=True)
class TrainCursor:
    """The next logical position to execute in a deterministic training stream."""

    global_step: int = 0
    next_epoch: int = 0
    next_batch: int = 0


@dataclass(frozen=True, slots=True)
class TrainCheckpoint:
    """Complete continuation payload delivered at configured step boundaries."""

    cursor: TrainCursor
    model_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    rng_state: dict[str, Any]
    data_order_state: dict[str, Any]
    scientific_records: list[dict[str, int | float]]


def seed_everything(seed: int) -> None:
    """Seed every RNG used by the supported deterministic CPU path."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _permutation(dataset_size: int, seed: int, epoch: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()


def train_loop(
    model: nn.Module,
    optimizer: SingleDeviceMuonWithAuxAdam,
    dataset: DomainDataset,
    config: TrainConfig,
    device: torch.device | str = "cpu",
    *,
    seed: int | None = None,
    cursor: TrainCursor | None = None,
    data_order_state: dict[str, Any] | None = None,
    checkpoint_interval: int | None = None,
    checkpoint_callback: Any | None = None,
    stop_after: int | None = None,
) -> list[TrainStepResult]:
    """Run the joint-objective training loop for ``config.max_steps`` steps.

    Precision note: ``config.precision == "bf16"`` enables autocast for the
    forward/backward, per TRAINING.md section 2. This does NOT apply to any
    future JVP / double-backward spectral-measurement code (not yet built) —
    that must run in fp32 with eager attention, a separate code path.

    Args:
        model: A module whose ``forward(input_ids, attention_mask)`` returns a
            dict with ``"logits"`` and ``"self_certainty"`` keys (see
            ``model.SeerPathAModel.forward``).
        optimizer: Built via ``optim.build_optimizer``.
        dataset: Single-domain dataset (dense per-step targets required).
        config: Training config; ``dense_supervision`` must be True.
        device: Device to run on. bf16 autocast is a no-op benefit-wise on CPU
            but still runs correctly — real speed gains need a CUDA device.

    Returns:
        Per-step loss breakdowns, in order, for logging/inspection.
    """
    if not config.dense_supervision:
        raise ValueError(
            "dense_supervision=False is not supported: TRAINING.md section 3 — a "
            "single terminal target does not train state-tracking."
        )

    training_seed = config.seeds[0] if seed is None else seed
    position = cursor or TrainCursor()
    if position.global_step < 0 or position.next_epoch < 0 or position.next_batch < 0:
        raise ValueError("training cursor positions must be non-negative")
    if position.global_step > config.max_steps:
        raise ValueError("training cursor is beyond max_steps")
    if cursor is None:
        seed_everything(training_seed)

    model.to(device)
    model.train()

    use_bf16 = config.precision == "bf16"
    results: list[TrainStepResult] = []
    step = position.global_step
    epoch = position.next_epoch
    batch_index = position.next_batch
    executed = 0
    batches_per_epoch = (len(dataset) + config.batch_size - 1) // config.batch_size
    if batches_per_epoch == 0:
        raise ValueError("training dataset must not be empty")
    while step < config.max_steps and (stop_after is None or executed < stop_after):
        permutation = _permutation(len(dataset), training_seed, epoch)
        if data_order_state and data_order_state.get("epoch") == epoch:
            persisted = data_order_state.get("epoch_permutation")
            if persisted != permutation:
                raise ValueError("persisted data order does not match deterministic permutation")
        while batch_index < batches_per_epoch:
            if step >= config.max_steps or (stop_after is not None and executed >= stop_after):
                break
            start = batch_index * config.batch_size
            indices = permutation[start : start + config.batch_size]
            batch: StateTrackingBatch = collate_state_tracking_examples(
                [dataset[index] for index in indices]
            )
            input_ids = batch.input_ids.to(device)
            step_targets = batch.step_targets.to(device)
            step_mask = batch.step_mask.to(device)

            with torch.autocast(
                device_type=torch.device(device).type, dtype=torch.bfloat16, enabled=use_bf16
            ):
                outputs = model(input_ids=input_ids)
                loss = joint_loss(
                    task_logits=outputs["logits"],
                    step_targets=step_targets,
                    step_mask=step_mask,
                    self_certainty=outputs["self_certainty"],
                    task_loss_weight=config.task_loss_weight,
                    energy_loss_weight=config.energy_loss_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.total.backward()
            optimizer.step()

            results.append(TrainStepResult(step=step, loss=loss))
            step += 1
            executed += 1
            batch_index += 1
            next_epoch, next_batch = epoch, batch_index
            if next_batch >= batches_per_epoch:
                next_epoch, next_batch = epoch + 1, 0
            if (
                checkpoint_callback is not None
                and checkpoint_interval is not None
                and step % checkpoint_interval == 0
            ):
                from seer.runtime import capture_rng_state

                checkpoint_callback(
                    TrainCheckpoint(
                        cursor=TrainCursor(step, next_epoch, next_batch),
                        model_state=copy.deepcopy(model.state_dict()),
                        optimizer_state=copy.deepcopy(optimizer.state_dict()),
                        rng_state=capture_rng_state(),
                        data_order_state={
                            "seed": training_seed,
                            "epoch": epoch,
                            "epoch_permutation": permutation,
                            "next_epoch": next_epoch,
                            "next_batch": next_batch,
                        },
                        scientific_records=[record.to_dict() for record in results],
                    )
                )
        if batch_index >= batches_per_epoch:
            epoch += 1
            batch_index = 0

    return results
