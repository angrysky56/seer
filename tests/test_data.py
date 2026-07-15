import pytest
import torch

from seer.data import (
    Domain,
    StateTrackingExample,
    SyntheticStateTrackingDataset,
    collate_state_tracking_examples,
)


def test_synthetic_dataset_length_and_shapes() -> None:
    domain = Domain("synthetic_parity")
    dataset = SyntheticStateTrackingDataset(domain=domain, num_examples=10, seq_len=16, seed=0)
    assert len(dataset) == 10
    example = dataset[0]
    assert example.input_ids.shape == (16,)
    assert example.step_targets.shape == (16,)
    assert example.step_mask.shape == (16,)
    assert example.domain == domain


def test_synthetic_dataset_targets_are_dense_running_parity() -> None:
    domain = Domain("synthetic_parity")
    dataset = SyntheticStateTrackingDataset(domain=domain, num_examples=1, seq_len=8, seed=0)
    example = dataset[0]
    expected = torch.remainder(torch.cumsum(example.input_ids, dim=0), 2)
    assert torch.equal(example.step_targets, expected)
    # dense supervision: every step has a real target, none masked out
    assert bool(example.step_mask.all())


def test_synthetic_dataset_is_deterministic_given_seed() -> None:
    domain = Domain("synthetic_parity")
    a = SyntheticStateTrackingDataset(domain=domain, num_examples=5, seq_len=8, seed=42)
    b = SyntheticStateTrackingDataset(domain=domain, num_examples=5, seq_len=8, seed=42)
    for i in range(5):
        assert torch.equal(a[i].input_ids, b[i].input_ids)


def test_collate_stacks_batch() -> None:
    domain = Domain("synthetic_parity")
    dataset = SyntheticStateTrackingDataset(domain=domain, num_examples=4, seq_len=8, seed=0)
    batch = collate_state_tracking_examples([dataset[i] for i in range(4)])
    assert batch.input_ids.shape == (4, 8)
    assert batch.step_targets.shape == (4, 8)
    assert batch.step_mask.shape == (4, 8)
    assert batch.domain == domain


def test_collate_rejects_mixed_domains() -> None:
    seq_len = 8
    a = StateTrackingExample(
        input_ids=torch.zeros(seq_len, dtype=torch.long),
        step_targets=torch.zeros(seq_len, dtype=torch.long),
        step_mask=torch.ones(seq_len, dtype=torch.bool),
        domain=Domain("domain_a"),
    )
    b = StateTrackingExample(
        input_ids=torch.zeros(seq_len, dtype=torch.long),
        step_targets=torch.zeros(seq_len, dtype=torch.long),
        step_mask=torch.ones(seq_len, dtype=torch.bool),
        domain=Domain("domain_b"),
    )
    with pytest.raises(ValueError, match="mixes domains"):
        collate_state_tracking_examples([a, b])
