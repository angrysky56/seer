import torch

from fakes import FakeBaseLM
from seer.config import ModelConfig, TrainConfig
from seer.data import Domain, SyntheticStateTrackingDataset
from seer.model import SeerPathAModel
from seer.optim import build_optimizer
from seer.train import TrainCursor, joint_loss, seed_everything, train_loop


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

    result = joint_loss(
        logits,
        targets,
        mask,
        certainty,
        task_loss_weight=2.0,
        energy_loss_weight=0.5,
    )
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


def _training_fixture(seed: int = 17):
    seed_everything(seed)
    base = FakeBaseLM(vocab_size=2, hidden_size=8, num_layers=1)
    model = SeerPathAModel(
        base,
        ModelConfig(base_model_name="fake", concept_dim=4, freeze_base=False),
    )
    optimizer = build_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)
    dataset = SyntheticStateTrackingDataset(
        domain=Domain("synthetic_parity"), num_examples=7, seq_len=8, seed=3
    )
    return model, optimizer, dataset


def test_train_loop_checkpoint_resume_matches_clean_scientific_records() -> None:
    config = TrainConfig(max_steps=7, batch_size=2, seeds=[17])
    clean_model, clean_optimizer, dataset = _training_fixture()
    clean = train_loop(clean_model, clean_optimizer, dataset, config, seed=17)

    resumed_model, resumed_optimizer, dataset = _training_fixture()
    checkpoints = []
    first = train_loop(
        resumed_model,
        resumed_optimizer,
        dataset,
        config,
        seed=17,
        stop_after=4,
        checkpoint_interval=2,
        checkpoint_callback=checkpoints.append,
    )
    checkpoint = checkpoints[-1]
    assert checkpoint.cursor.global_step == 4
    assert checkpoint.cursor.next_batch == 0
    assert checkpoint.cursor.next_epoch == 1
    assert checkpoint.data_order_state["epoch_permutation"] == [2, 4, 3, 5, 6, 1, 0]

    continued = train_loop(
        resumed_model,
        resumed_optimizer,
        dataset,
        config,
        seed=17,
        cursor=checkpoint.cursor,
        data_order_state=checkpoint.data_order_state,
    )
    assert [record.to_dict() for record in clean] == [
        record.to_dict() for record in first + continued
    ]
    assert [record.step for record in first + continued] == list(range(7))


def test_train_loop_honors_explicit_mid_epoch_cursor_and_callback_cadence() -> None:
    config = TrainConfig(max_steps=5, batch_size=2, seeds=[5])
    model, optimizer, dataset = _training_fixture(seed=5)
    checkpoints = []
    records = train_loop(
        model,
        optimizer,
        dataset,
        config,
        seed=5,
        cursor=TrainCursor(global_step=1, next_epoch=0, next_batch=1),
        checkpoint_interval=2,
        checkpoint_callback=checkpoints.append,
    )
    assert [record.step for record in records] == [1, 2, 3, 4]
    assert [checkpoint.cursor.global_step for checkpoint in checkpoints] == [2, 4]
    assert all("python" in checkpoint.rng_state for checkpoint in checkpoints)
