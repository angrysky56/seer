import torch
from fakes import FakeBaseLM
from torch import nn

from seer.optim import build_optimizer, classify_named_parameters


def _param(*shape: int, requires_grad: bool = True) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape), requires_grad=requires_grad)


def test_hidden_matmul_routes_to_muon() -> None:
    p = _param(4, 4)
    muon, adamw = classify_named_parameters([("model.layers.0.self_attn.q_proj.weight", p)])
    assert muon == [p]
    assert adamw == []


def test_embedding_routes_to_adamw_despite_ndim_2() -> None:
    p = _param(100, 16)
    muon, adamw = classify_named_parameters([("model.embed_tokens.weight", p)])
    assert muon == []
    assert adamw == [p]


def test_lm_head_routes_to_adamw() -> None:
    p = _param(16, 100)
    muon, adamw = classify_named_parameters([("lm_head.weight", p)])
    assert adamw == [p]


def test_layernorm_and_bias_route_to_adamw() -> None:
    norm_weight = _param(16)
    bias = _param(16)
    muon, adamw = classify_named_parameters(
        [
            ("model.layers.0.input_layernorm.weight", norm_weight),
            ("model.layers.0.self_attn.q_proj.bias", bias),
        ]
    )
    assert muon == []
    assert set(adamw) == {norm_weight, bias}


def test_energy_head_forced_to_adamw_even_if_name_collides_with_hidden_hint() -> None:
    """Role beats shape/name-collision: TRAINING.md's 'easy miss'.

    A contrived name that contains BOTH a hidden-matmul hint substring and a
    forced-AdamW hint substring must still land in AdamW.
    """
    p = _param(8, 8)
    muon, adamw = classify_named_parameters(
        [("self_attn.q_proj.energy_head.weight", p)]
    )
    assert muon == []
    assert adamw == [p]


def test_requires_grad_false_params_are_skipped() -> None:
    p = _param(4, 4, requires_grad=False)
    muon, adamw = classify_named_parameters([("model.layers.0.self_attn.q_proj.weight", p)])
    assert muon == []
    assert adamw == []


def test_extra_muon_predicate_extends_routing() -> None:
    p = _param(4, 4)
    muon, adamw = classify_named_parameters(
        [("custom_block.proj.weight", p)],
        extra_muon_predicate=lambda name, _p: "custom_block" in name,
    )
    assert muon == [p]
    assert adamw == []


def test_extra_muon_predicate_does_not_override_forced_adamw() -> None:
    p = _param(4, 4)
    muon, adamw = classify_named_parameters(
        [("custom_block.energy_head.weight", p)],
        extra_muon_predicate=lambda name, _p: "custom_block" in name,
    )
    assert muon == []
    assert adamw == [p]


def test_build_optimizer_on_fake_base_lm_produces_nonempty_groups_and_steps() -> None:
    torch.manual_seed(0)
    model = FakeBaseLM(vocab_size=5, hidden_size=8, num_layers=2)
    optimizer = build_optimizer(model, muon_lr=0.02, adamw_lr=1e-3)

    muon_group, adamw_group = optimizer.param_groups
    assert muon_group["use_muon"] is True
    assert len(muon_group["params"]) > 0  # q/k/v/o_proj + gate/up/down_proj across 2 layers
    assert adamw_group["use_muon"] is False
    assert len(adamw_group["params"]) > 0  # embed_tokens, lm_head, norms

    before = {n: p.clone() for n, p in model.named_parameters()}

    input_ids = torch.randint(0, 5, (2, 6))
    logits = model(input_ids).logits
    loss = logits.sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    assert len(changed) > 0
    assert all(torch.isfinite(p).all() for _, p in model.named_parameters())
