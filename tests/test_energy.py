import torch

from seer.energy import (
    EnergyHead,
    SelfCertainty,
    contrastive_energy_loss,
    predictor_disagreement,
)


def test_energy_head_output_shape() -> None:
    head = EnergyHead(concept_dim=8)
    z = torch.randn(4, 6, 8)  # (batch, seq, concept_dim)
    energy = head(z)
    assert energy.shape == (4, 6)


def test_self_certainty_is_in_unit_interval() -> None:
    certainty = SelfCertainty()
    energy = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    p = certainty(energy)
    assert torch.all(p > 0) and torch.all(p < 1)
    # lower energy (more plausible) must map to higher p(correct)
    assert bool(torch.all(p[:-1] >= p[1:]))


def test_contrastive_energy_loss_zero_when_margin_satisfied() -> None:
    energy_pos = torch.zeros(4)
    energy_neg = torch.full((4,), 5.0)
    loss = contrastive_energy_loss(energy_pos, energy_neg, margin=1.0)
    assert loss.item() == 0.0


def test_contrastive_energy_loss_positive_when_margin_violated() -> None:
    energy_pos = torch.zeros(4)
    energy_neg = torch.zeros(4)  # no gap at all
    loss = contrastive_energy_loss(energy_pos, energy_neg, margin=1.0)
    assert loss.item() == 1.0


def test_predictor_disagreement_shape_and_nonnegative() -> None:
    torch.manual_seed(0)

    class NoisyPredictor(torch.nn.Module):
        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return z + torch.randn_like(z) * 0.1

    predictor = NoisyPredictor()
    z = torch.randn(5, 8)
    disagreement = predictor_disagreement(predictor, z, k=8)
    assert disagreement.shape == (5,)
    assert torch.all(disagreement >= 0)


def test_predictor_disagreement_rejects_k_less_than_2() -> None:
    predictor = torch.nn.Identity()
    z = torch.randn(2, 4)
    try:
        predictor_disagreement(predictor, z, k=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for k < 2")
