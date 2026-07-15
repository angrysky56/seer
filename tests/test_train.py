import torch

from seer.config import ModelConfig, TrainConfig
from seer.data import Domain, SyntheticStateTrackingDataset
from seer.model import SeerPathAModel
from seer.optim import build_optimizer
from seer.train import joint_loss, train_loop
from fakes import FakeBaseLM


def test_joint_loss_combines_task_and_certainty_terms() -> None:
    torch.manual_seed(0)
    batch, seq_len, vocab = 2, 4, 2
    logits = torch.randn(batch, seq_len, vocab, requires_grad=True)
    targets = torch.randint(0, vocab, (batch, seq_len))
    mask = torch.ones(batch, seq_len, dtype=torch.bool)
    certainty = torch.rand(batch, seq_len)

    result = joint_loss(
        task_logits=logits,
        step_targets=targets,
        step_mask=mask,
        self_certainty=certainty,
        task_loss_weight=1.0,
        energy_loss_weight=1.0,
    )

    assert torch.isfinite(result.total)
    assert torch.allclose(result.total, result.task_loss + result.certainty_loss, atol=1e-5)
    assert result.total.requires_grad  # must be backprop-able through task_logits
    assert 0.0 <= result.task_accuracy.item() <= 1.0


def test_joint_loss_weighting_scales_components() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 2, requires_grad=True)
    targets = torch.randint(0, 2, (2, 4))
    mask = torch.ones(2, 4, dtype=torch.bool)
    certainty = torch.rand(2, 4)

    result = joint_loss(logits, targets, mask, certainty, task_loss_weight=2.0, energy_loss_weight=0.5)
    expected = 2.0 * result.task_loss + 0.5 * result.certainty_loss
    assert torch.allclose(result.total, expected, atol=1e-5)


def test_train_loop_runs_and_updates_parameters() -> None:
    """Wiring smoke test: the loop runs, losses stay finite, and Muon+AdamW both fire.

    Not a claim the fake model can learn cumulative parity (FakeAttention has no
    cross-token mixing, see tests/fakes.py) — only that the joint-loss /
    backward / optimizer-step wiring is correct end to end.
    """
    torch.manual_seed(0)
    vocab_size, hidden_size = 2, 8
    base = FakeBaseLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=1)
    model_config = ModelConfig(
        base_model_name="fake", concept_dim=4, freeze_base=False, commit_layer=-1
    )
    model = SeerPathAModel(base, model_config)
    optimizer = build_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)

    dataset = SyntheticStateTrackingDataset(
        domain=Domain("synthetic_parity"), num_examples=16, seq_len=8, vocab_size=2, seed=0
    )
    train_config = TrainConfig(max_steps=5, batch_size=4, seeds=[0])

    before = {n: p.clone() for n, p in model.named_parameters()}
    results = train_loop(model, optimizer, dataset, train_config, device="cpu")

    assert len(results) == 5
    assert all(torch.isfinite(r.loss.total) for r in results)

    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert len(changed) > 0


def test_train_loop_rejects_sparse_supervision() -> None:
    vocab_size, hidden_size = 2, 8
    base = FakeBaseLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=1)
    model_config = ModelConfig(base_model_name="fake", concept_dim=4)
    model = SeerPathAModel(base, model_config)
    optimizer = build_optimizer(model)
    dataset = SyntheticStateTrackingDataset(
        domain=Domain("synthetic_parity"), num_examples=4, seq_len=8, vocab_size=2, seed=0
    )
    train_config = TrainConfig(max_steps=1, batch_size=4, dense_supervision=False)

    try:
        train_loop(model, optimizer, dataset, train_config, device="cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for dense_supervision=False")
