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

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

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


def train_loop(
    model: nn.Module,
    optimizer: SingleDeviceMuonWithAuxAdam,
    dataset: DomainDataset,
    config: TrainConfig,
    device: torch.device | str = "cpu",
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

    model.to(device)
    model.train()

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_state_tracking_examples,
    )

    use_bf16 = config.precision == "bf16"
    results: list[TrainStepResult] = []
    step = 0
    while step < config.max_steps:
        for batch in loader:
            if step >= config.max_steps:
                break
            batch: StateTrackingBatch
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

    return results
