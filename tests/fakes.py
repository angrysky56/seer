"""Fake base-model components used only by tests.

These stand in for a real pretrained HF causal LM (Qwen2/Gemma) so
``SeerPathAModel`` and the training loop can be exercised on CPU, deterministically,
with no network access or weight download. Submodule names deliberately mirror
the common Llama-family layout (``self_attn.{q,k,v,o}_proj``,
``mlp.{gate,up,down}_proj``, ``norm``) so ``optim.classify_named_parameters``'s
default name-hint heuristic is exercised meaningfully, not vacuously.

Not a claim that this architecture can learn any particular task — in
particular ``FakeAttention`` mixes only within a position (no cross-token
attention), so it cannot solve cross-position tasks like cumulative parity.
That is intentional: it keeps the fakes tiny and fast, and the tests that use
them only assert wiring correctness (shapes, finite losses, parameters
actually update), never task performance.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class FakeMLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.down_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.relu(self.gate_proj(x) + self.up_proj(x)))


class FakeDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(hidden_size)
        self.mlp = FakeMLP(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return self.norm(x)


class FakeConfig:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class FakeCausalLMOutput:
    def __init__(self, logits: Tensor, hidden_states: tuple[Tensor, ...] | None) -> None:
        self.logits = logits
        self.hidden_states = hidden_states


class FakeBaseLM(nn.Module):
    """Minimal stand-in for a HF ``AutoModelForCausalLM``."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.config = FakeConfig(hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [FakeDecoderLayer(hidden_size) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> FakeCausalLMOutput:
        del attention_mask  # unused by the fake; real base models use it for padding
        h = self.embed_tokens(input_ids)
        hidden_states: list[Tensor] | None = [h] if output_hidden_states else None
        for layer in self.layers:
            h = layer(h)
            if hidden_states is not None:
                hidden_states.append(h)
        logits = self.lm_head(h)
        return FakeCausalLMOutput(
            logits=logits,
            hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        )
