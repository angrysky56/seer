"""Experiment configuration for SEER's Path A energy-transfer experiment (ROADMAP.md, step 1).

Dataclasses only — no argument parsing here. A thin CLI/YAML loader can be added
once an experiment actually runs; until then this is the single source of truth
for what a run needs to specify.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Precision = Literal["bf16", "fp32"]


@dataclass(slots=True)
class ModelConfig:
    """Path A model: a pretrained causal LM + concept projection + energy head.

    Attributes:
        base_model_name: HF hub id or local path of the pretrained causal LM
            (e.g. a small Qwen2 or Gemma checkpoint). Not resolved at config
            construction time — resolution happens in ``SeerPathAModel.from_pretrained``.
        concept_dim: Dimensionality of the projected concept space z. Independent
            of the base model's hidden size so the energy head can be sized
            separately from the trunk.
        freeze_base: If True, only the concept projection and energy head are
            trained (cheapest, matches the "head-only" ~0.955 AUROC rung in
            TRAINING.md section 4). If False, the base model's own weights are
            also trainable (the "baked in" rung, ~0.968 AUROC, forgets without
            the joint task anchor — see TrainConfig.task_loss_weight).
        commit_layer: Index of the decoder layer whose residual stream is read
            as the concept, following ARCHITECTURE.md section 5 ("reading the
            residual state at commit positions"). -1 means the final layer.
    """

    base_model_name: str
    concept_dim: int = 256
    freeze_base: bool = True
    commit_layer: int = -1


@dataclass(slots=True)
class OptimConfig:
    """Muon (hidden matmuls) + AdamW (everything else), per TRAINING.md section 1.

    Muon and AdamW learning rates live in different units (spectral-norm-per-update
    vs the usual gradient scale) and must be swept independently — never share a
    single ``lr`` field between them.
    """

    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.0
    adamw_lr: float = 3e-4
    adamw_betas: tuple[float, float] = (0.9, 0.95)
    adamw_eps: float = 1e-10
    adamw_weight_decay: float = 0.0


@dataclass(slots=True)
class TrainConfig:
    """Joint task + self-certainty objective, per TRAINING.md section 4.

    Attributes:
        task_loss_weight: Weight on the task (LM / state-tracking) loss. This is
            the anchor — TRAINING.md measured task accuracy collapse 0.931 -> 0.059
            when the self-certainty head is trained without it.
        energy_loss_weight: Weight on the self-certainty / energy loss.
        precision: "bf16" for the training forward/backward. Any future
            JVP / double-backward spectral measurement code must run in fp32
            (TRAINING.md section 2) — that is a separate, not-yet-built code path.
        dense_supervision: Must stay True. TRAINING.md section 3: a single
            terminal target does not train state-tracking; supervise every
            state-dependent step.
        seed: Single-seed effects on a small model are not to be trusted
            (TRAINING.md section 7) — always run seeds as a list and report
            mean +/- sd, never one number.
    """

    task_loss_weight: float = 1.0
    energy_loss_weight: float = 1.0
    precision: Precision = "bf16"
    dense_supervision: bool = True
    max_steps: int = 1000
    batch_size: int = 8
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


@dataclass(slots=True)
class EvalConfig:
    """Eval harness settings, per TRAINING.md section 7.

    Attributes:
        train_domain: The domain the self-certainty head is trained on.
        transfer_domains: Out-of-domain evaluation targets — this is the whole
            point of experiment 1 (ROADMAP.md H-energy-transfer): does the energy
            AUROC survive transfer where the token probe collapsed to ~0.50?
        ece_bins: Adaptive/equal-mass bins, not equal-width (TRAINING.md: equal-width
            manufactures artifacts).
        run_shuffle_control: Must stay True for any reported result. AUROC on
            permuted labels must fall to ~0.5, else the pipeline leaks.
    """

    train_domain: str
    transfer_domains: list[str] = field(default_factory=list)
    ece_bins: int = 10
    run_shuffle_control: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    """Top-level config composing model/optim/train/eval for one experiment run."""

    name: str
    model: ModelConfig
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig | None = None
