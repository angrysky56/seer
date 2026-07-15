"""Muon + AdamW, split by parameter role (TRAINING.md section 1).

Measured on the predecessor entity-tracking task: Muon reached 0.989
answer-accuracy by step 1000 vs AdamW's 0.798 at matched default LRs. Use Muon
for hidden weight matrices; AdamW for everything else.

The Muon optimizer implementation (``zeropower_via_newtonschulz5``,
``muon_update``, ``SingleDeviceMuonWithAuxAdam``) is adapted from Keller
Jordan's reference implementation (MIT license):
https://github.com/KellerJordan/Muon — trimmed to the single-device variant
(no ``torch.distributed``), since TRAINING.md's hardware baseline is a single
RTX 3060.

The part that is genuinely SEER-specific, and the part TRAINING.md calls "the
easy miss", is ``classify_named_parameters``: routing must be done BY ROLE, not
by ``ndim``. A learnable energy-head or projection vector can be 2-D and still
not be doing ``y = x @ W`` as a transformer hidden matmul — it must go to
AdamW regardless of shape.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn

# --- Muon core (Newton-Schulz orthogonalization), vendored single-device ----


def zeropower_via_newtonschulz5(g: Tensor, steps: int) -> Tensor:
    """Newton-Schulz iteration approximating the orthogonalization (zeroth power) of g.

    See Keller Jordan's writeup: https://kellerjordan.github.io/posts/muon/
    Produces something like U S' V^T (S' ~ Uniform(0.5, 1.5) on the diagonal)
    rather than exactly U V^T — empirically this does not hurt performance
    relative to the true zeroth power.
    """
    if g.ndim < 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects ndim >= 2, got {g.ndim}")
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.bfloat16()
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        a_mat = x @ x.mT
        b_mat = b * a_mat + c * a_mat @ a_mat
        x = a * x + b_mat @ x
    if transposed:
        x = x.mT
    return x


def muon_update(
    grad: Tensor,
    momentum: Tensor,
    beta: float = 0.95,
    ns_steps: int = 5,
    nesterov: bool = True,
) -> Tensor:
    """Compute one Muon update given the raw gradient and the momentum buffer (in place)."""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # conv filters, kept for completeness though unused by SEER
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1.0, update.size(-2) / update.size(-1)) ** 0.5
    return update


def adam_update(
    grad: Tensor,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: int,
    betas: tuple[float, float],
    eps: float,
) -> Tensor:
    """Standard bias-corrected Adam update (moments updated in place)."""
    exp_avg.lerp_(grad, 1 - betas[0])
    exp_avg_sq.lerp_(grad.square(), 1 - betas[1])
    bias_corrected_avg = exp_avg / (1 - betas[0] ** step)
    bias_corrected_avg_sq = exp_avg_sq / (1 - betas[1] ** step)
    return bias_corrected_avg / (bias_corrected_avg_sq.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Single-process Muon (hidden matmuls) + AdamW (everything else) optimizer.

    Construct via :func:`build_optimizer` rather than directly, so parameter
    routing goes through :func:`classify_named_parameters` and stays consistent
    with TRAINING.md section 1.
    """

    def __init__(self, param_groups: list[dict]) -> None:
        for group in param_groups:
            if "use_muon" not in group:
                raise ValueError("each param group must set 'use_muon'")
            if group["use_muon"]:
                group.setdefault("lr", 0.02)
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
            else:
                group.setdefault("lr", 3e-4)
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-10)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(
                        p.grad,
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        state["step"],
                        group["betas"],
                        group["eps"],
                    )
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
        return loss


# --- Parameter routing (the SEER-specific, easy-to-get-wrong part) ----------

#: Name substrings identifying a transformer hidden matmul weight in the common
#: Llama-family layout that Qwen2 and Gemma both follow: decoder layers under
#: some ``...layers.<i>...`` path, with attention projections and MLP
#: projections as ``nn.Linear`` weights. This is a heuristic, not a guarantee —
#: verify against the actual base model's ``named_parameters()`` output when
#: swapping in a new architecture (print the classification and sanity-check
#: it before the first real run).
_HIDDEN_MATMUL_NAME_HINTS: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "attn.c_attn",
    "attn.c_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "mlp.c_fc",
    "mlp.c_proj",
    "mlp.fc1",
    "mlp.fc2",
)

#: Name substrings that are ALWAYS routed to AdamW even if they matched a hidden
#: matmul hint above or have ndim >= 2 — role beats shape (TRAINING.md section 1:
#: "the easy miss"). Covers embeddings, the LM/decoder head, norms, biases, and
#: anything belonging to SEER's own energy/self-certainty/projection heads.
_FORCE_ADAMW_NAME_HINTS: tuple[str, ...] = (
    "embed",
    "wte",
    "wpe",
    "lm_head",
    "norm",
    "ln_",
    "layernorm",
    "bias",
    "energy_head",
    "self_certainty",
    "concept_proj",
    "predict",
)


def classify_named_parameters(
    named_params: Iterable[tuple[str, nn.Parameter]],
    extra_muon_predicate: Callable[[str, nn.Parameter], bool] | None = None,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split named parameters into (muon_params, adamw_params) by role.

    Default heuristic: a parameter goes to Muon only if (a) its name matches a
    known hidden-matmul substring, (b) it is not forced to AdamW by name (norms,
    biases, embeddings, heads, SEER's own energy/projection modules), and
    (c) it has ``ndim >= 2``. Everything else goes to AdamW.

    Args:
        named_params: Output of ``model.named_parameters()`` (or a filtered
            subset of it).
        extra_muon_predicate: Optional override/extension for architectures not
            covered by the default name hints — called as
            ``predicate(name, param)``; if it returns True the parameter is
            routed to Muon (still subject to the AdamW force-list and the
            ``ndim >= 2`` requirement).

    Returns:
        ``(muon_params, adamw_params)``.
    """
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []

    for name, param in named_params:
        if not param.requires_grad:
            continue
        lower = name.lower()
        forced_adamw = any(hint in lower for hint in _FORCE_ADAMW_NAME_HINTS)
        is_hidden_matmul = any(hint in lower for hint in _HIDDEN_MATMUL_NAME_HINTS)
        if extra_muon_predicate is not None and extra_muon_predicate(name, param):
            is_hidden_matmul = True

        if not forced_adamw and is_hidden_matmul and param.ndim >= 2:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    return muon_params, adamw_params


def build_optimizer(
    model: nn.Module,
    muon_lr: float = 0.02,
    muon_momentum: float = 0.95,
    muon_weight_decay: float = 0.0,
    adamw_lr: float = 3e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-10,
    adamw_weight_decay: float = 0.0,
    extra_muon_predicate: Callable[[str, nn.Parameter], bool] | None = None,
) -> SingleDeviceMuonWithAuxAdam:
    """Build a :class:`SingleDeviceMuonWithAuxAdam` with SEER's default role routing.

    Mirrors ``OptimConfig`` field names in ``config.py`` — construct from an
    ``OptimConfig`` instance with ``build_optimizer(model, **vars(cfg))`` once a
    config object exists at the call site.
    """
    muon_params, adamw_params = classify_named_parameters(
        model.named_parameters(), extra_muon_predicate=extra_muon_predicate
    )
    param_groups = [
        dict(params=muon_params, use_muon=True, lr=muon_lr, momentum=muon_momentum,
             weight_decay=muon_weight_decay),
        dict(params=adamw_params, use_muon=False, lr=adamw_lr, betas=adamw_betas,
             eps=adamw_eps, weight_decay=adamw_weight_decay),
    ]
    return SingleDeviceMuonWithAuxAdam(param_groups)
