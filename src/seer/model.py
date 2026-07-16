"""Path A model wrapper (ARCHITECTURE.md section 5): pretrained LM + concept + energy.

Path A augments a capable pretrained causal LM rather than training a
latent-predictive model from scratch: keep the base model as the token model,
add a projection to a concept space and an energy head reading the residual
state, and treat the base model's own competence as the pretrained prior.
ARCHITECTURE.md recommends proving H-energy-transfer on Path A first (cheap,
local, reuses the L1 ladder) before committing to Path B.

Design choice worth flagging: the concept projection here is applied
PER-POSITION (not masked-mean-pooled over the sequence). ARCHITECTURE.md
section 1's "masked-mean pooled + L2-normalized targets" rule describes Path
B's lang-jepa-style next-concept target construction; Path A instead reads the
residual state "at commit positions" (section 5), and TRAINING.md section 3's
dense-per-step-supervision rule is exactly why per-position beats pooled here —
pooling to one vector per sequence would throw away the per-step targets that
section 3 says are required to learn state-tracking at all.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from seer.cache import ResolvedSnapshot, require_transformers_version, resolve_cached_snapshot
from seer.config import ModelConfig
from seer.energy import EnergyHead, SelfCertainty


class CausalLMOutputLike(Protocol):
    """Minimal shape SeerPathAModel needs from a base model's forward output.

    Matches transformers' ``CausalLMOutputWithPast`` closely enough (``.logits``,
    ``.hidden_states``) without importing transformers types, so a small fake
    model can stand in for tests without a network call / weight download.
    """

    logits: Tensor
    hidden_states: tuple[Tensor, ...] | None


class HasHiddenSizeConfig(Protocol):
    hidden_size: int


class BaseLMLike(Protocol):
    """Minimal protocol a base model must satisfy to be wrapped by SeerPathAModel."""

    config: HasHiddenSizeConfig

    def __call__(
        self, input_ids: Tensor, attention_mask: Tensor | None, output_hidden_states: bool
    ) -> CausalLMOutputLike: ...


class SeerPathAModel(nn.Module):
    """Wraps a pretrained causal LM with a concept projection and energy head.

    Attributes named ``concept_proj``, ``energy_head``, ``self_certainty`` are
    matched by name in ``optim.classify_named_parameters``'s AdamW force-list —
    keep these names in sync if either module is renamed.
    """

    def __init__(self, base_model: BaseLMLike, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.concept_proj = nn.Linear(hidden_size, config.concept_dim)
        self.energy_head = EnergyHead(config.concept_dim)
        self.self_certainty = SelfCertainty()

        if config.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        config: ModelConfig,
        *,
        snapshot: ResolvedSnapshot | None = None,
        resolver: Callable[..., str] | None = None,
        loader: Callable[..., Any] | None = None,
        transformers_version: str | None = None,
    ) -> SeerPathAModel:
        """Verify and load one exact cached snapshot without a network fallback."""
        require_transformers_version(transformers_version)
        if not config.local_files_only:
            raise ValueError("SeerPathAModel requires local_files_only=True")
        if config.revision is None:
            raise ValueError("model.revision is required for exact cached loading")
        if snapshot is None:
            snapshot = resolve_cached_snapshot(
                config.base_model_name,
                config.revision,
                cache_dir=config.cache_dir,
                resolver=resolver,
            )
        if (snapshot.repository_id, snapshot.revision) != (
            config.base_model_name,
            config.revision,
        ):
            raise ValueError("verified snapshot identity does not match the model configuration")
        if loader is None:
            from transformers import AutoModelForCausalLM

            loader = AutoModelForCausalLM.from_pretrained
        base_model = loader(snapshot.snapshot_path, local_files_only=True)
        return cls(base_model, config)

    def _commit_hidden_states(self, hidden_states: tuple[Tensor, ...]) -> Tensor:
        """Select the residual stream at the configured commit layer.

        ``hidden_states`` from a HF-style output has length ``num_layers + 1``
        (index 0 is the embedding output). ``commit_layer=-1`` means the final
        decoder layer's output, matching Python negative-indexing convention.
        """
        return hidden_states[self.config.commit_layer]

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Run the base model and compute the concept/energy channel at every position.

        Args:
            input_ids: Token ids, shape ``(batch, seq_len)``.
            attention_mask: Optional attention mask, shape ``(batch, seq_len)``.

        Returns:
            Dict with:
                - ``logits``: task (LM) logits, shape ``(batch, seq_len, vocab)``.
                - ``concept``: L2-normalized concept vectors, shape
                  ``(batch, seq_len, concept_dim)``.
                - ``energy``: energy per position, shape ``(batch, seq_len)``.
                - ``self_certainty``: calibrated p(correct) per position, shape
                  ``(batch, seq_len)``.
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        if outputs.hidden_states is None:
            raise RuntimeError("base_model did not return hidden_states; check the call site")
        commit_hidden = self._commit_hidden_states(outputs.hidden_states)
        concept = self.concept_proj(commit_hidden)
        concept = torch.nn.functional.normalize(concept, p=2, dim=-1)
        energy = self.energy_head(concept)
        certainty = self.self_certainty(energy)
        return {
            "logits": outputs.logits,
            "concept": concept,
            "energy": energy,
            "self_certainty": certainty,
        }
