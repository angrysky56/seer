"""The energy / self-certainty channel (ARCHITECTURE.md section 2).

Two ways to realize the channel, in increasing ambition:

- ``PredictorDisagreement`` — cheap, no target needed, available at inference:
  K stochastic forward passes, variance of the predicted concept is the
  uncertainty estimate.
- ``EnergyHead`` (+ ``contrastive_energy_loss``) — a learned EBM: in-distribution
  concepts get low energy, corrupted/off-manifold concepts get high energy. This
  is the channel with a shot at domain-general transfer (ROADMAP.md
  H-energy-transfer), because it scores representation plausibility rather than
  token-surface features.

Neither is wired to a real base model yet — that happens in ``model.py``. This
module only defines the head and its losses so they can be unit-tested in
isolation on synthetic concept vectors.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class EnergyHead(nn.Module):
    """Scores how plausible a concept vector z is under the model's learned manifold.

    A small MLP mapping z -> scalar energy. Lower energy = more plausible. Kept
    deliberately simple (LayerNorm + 2-layer MLP) — TRAINING.md section 1 flags
    this exact kind of "2-D-ish but not doing y = x @ W" parameter as the easy
    miss for the Muon/AdamW router, so it must never be routed to Muon (see
    ``optim.classify_named_parameters``, which excludes anything under an
    ``EnergyHead`` by role, not shape).
    """

    def __init__(self, concept_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or concept_dim
        self.norm = nn.LayerNorm(concept_dim)
        self.net = nn.Sequential(
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Compute energy for a batch of concept vectors.

        Args:
            z: Concept vectors, shape ``(..., concept_dim)``.

        Returns:
            Energy, shape ``(...,)`` — one scalar per concept vector.
        """
        return self.net(self.norm(z)).squeeze(-1)


class SelfCertainty(nn.Module):
    """Calibrated monotone map of -E(z) to p(correct), per ARCHITECTURE.md section 2.

    A single learnable per-domain temperature. TRAINING.md section 5 and
    section 9 both expect a *per-domain* temperature rather than one global
    scalar once the transfer ladder is run across domains — this module holds
    exactly one temperature, so a domain-gated setup should construct one
    instance per domain rather than trying to share a single instance.
    """

    def __init__(self, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(init_temperature))))

    def forward(self, energy: Tensor) -> Tensor:
        """Map energy to a calibrated p(correct) in (0, 1).

        Args:
            energy: Shape ``(...,)``.

        Returns:
            p(correct), shape ``(...,)``.
        """
        temperature = self.log_temperature.exp()
        return torch.sigmoid(-energy / temperature)


def contrastive_energy_loss(
    energy_pos: Tensor,
    energy_neg: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """Hinge contrastive loss for training the EBM (TRAINING.md section 6).

    Pushes in-distribution ("positive") concepts to low energy and
    corrupted/off-manifold ("negative") concepts to at least ``margin`` higher
    energy. Kept off the target-encoder path — this loss trains only the energy
    head (and, if unfrozen, the concept projection), never the EMA target
    encoder itself (collapse accelerant per ARCHITECTURE.md section 1).

    Args:
        energy_pos: Energy of in-distribution concepts, any shape.
        energy_neg: Energy of corrupted/negative concepts, broadcastable to
            ``energy_pos``.
        margin: Minimum energy gap enforced between negatives and positives.

    Returns:
        Scalar loss, mean hinge violation.
    """
    return torch.clamp(margin - (energy_neg - energy_pos), min=0.0).mean()


@torch.no_grad()
def predictor_disagreement(
    predictor: nn.Module,
    z: Tensor,
    k: int = 8,
) -> Tensor:
    """Cheap uncertainty estimate: variance of K stochastic predictor passes.

    No target needed, available at inference. The caller is responsible for
    ensuring ``predictor`` actually has stochasticity to sample (e.g. dropout
    left in train mode, or an ensemble of predictor heads called in a loop) —
    this function does not toggle ``predictor.train()`` itself since that is a
    model-wide switch the caller should own.

    Args:
        predictor: Callable concept predictor, ``z -> predicted_z``.
        z: Input concept(s) to the predictor, shape ``(batch, concept_dim)``.
        k: Number of stochastic passes.

    Returns:
        Per-example uncertainty (mean variance across concept dims), shape
        ``(batch,)``. High value = the model is unsure.
    """
    if k < 2:
        raise ValueError(f"k must be >= 2 to estimate variance, got {k}")
    samples = torch.stack([predictor(z) for _ in range(k)], dim=0)  # (k, batch, concept_dim)
    return samples.var(dim=0, unbiased=True).mean(dim=-1)
