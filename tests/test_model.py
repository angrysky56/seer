import torch

from seer.config import ModelConfig
from seer.model import SeerPathAModel
from fakes import FakeBaseLM


def _build(freeze_base: bool = True) -> tuple[SeerPathAModel, int, int]:
    torch.manual_seed(0)
    vocab_size, hidden_size, num_layers = 5, 16, 2
    base = FakeBaseLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers)
    config = ModelConfig(
        base_model_name="fake",
        concept_dim=8,
        freeze_base=freeze_base,
        commit_layer=-1,
    )
    model = SeerPathAModel(base, config)
    return model, vocab_size, hidden_size


def test_forward_output_shapes() -> None:
    model, vocab_size, _ = _build()
    batch, seq_len = 3, 6
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))

    out = model(input_ids)

    assert out["logits"].shape == (batch, seq_len, vocab_size)
    assert out["concept"].shape == (batch, seq_len, 8)
    assert out["energy"].shape == (batch, seq_len)
    assert out["self_certainty"].shape == (batch, seq_len)


def test_concept_vectors_are_l2_normalized() -> None:
    model, vocab_size, _ = _build()
    input_ids = torch.randint(0, vocab_size, (2, 5))
    out = model(input_ids)
    norms = out["concept"].norm(p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_self_certainty_in_unit_interval() -> None:
    model, vocab_size, _ = _build()
    input_ids = torch.randint(0, vocab_size, (2, 5))
    out = model(input_ids)
    p = out["self_certainty"]
    assert torch.all(p > 0) and torch.all(p < 1)


def test_freeze_base_stops_gradients_to_base_model() -> None:
    model, _, _ = _build(freeze_base=True)
    assert all(not p.requires_grad for p in model.base_model.parameters())
    assert all(p.requires_grad for p in model.concept_proj.parameters())
    assert all(p.requires_grad for p in model.energy_head.parameters())


def test_unfrozen_base_keeps_gradients() -> None:
    model, _, _ = _build(freeze_base=False)
    assert all(p.requires_grad for p in model.base_model.parameters())


def test_commit_layer_zero_uses_embedding_output() -> None:
    """commit_layer=0 should read the embedding output (hidden_states[0]), pre-transformer."""
    torch.manual_seed(0)
    vocab_size, hidden_size = 5, 16
    base = FakeBaseLM(vocab_size=vocab_size, hidden_size=hidden_size, num_layers=2)
    config = ModelConfig(base_model_name="fake", concept_dim=8, commit_layer=0)
    model = SeerPathAModel(base, config)

    input_ids = torch.randint(0, vocab_size, (2, 4))
    out = model(input_ids)

    with torch.no_grad():
        expected_concept = torch.nn.functional.normalize(
            model.concept_proj(base.embed_tokens(input_ids)), p=2, dim=-1
        )
    assert torch.allclose(out["concept"], expected_concept, atol=1e-5)
