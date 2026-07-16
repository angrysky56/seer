"""Bounded, label-blind generation for immutable Phase 2 evidence."""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from seer.cache import QWEN3_REPOSITORY, QWEN3_REVISION, resolve_cached_snapshot
from seer.evidence import (
    FailureRecord,
    GenerationRecord,
    canonical_json_bytes,
    generation_id,
    record_id,
)
from seer.partitions import ProtectedExample

PROMPT_VERSION = 1


class GenerationError(RuntimeError):
    """A generation contract or immutable identity was violated."""


class TokenizerProtocol(Protocol):
    chat_template: str
    eos_token_id: int | list[int] | None
    pad_token_id: int | None

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...
    def decode(self, token_ids: list[int], **kwargs: Any) -> str: ...


class ModelProtocol(Protocol):
    def generate(self, input_ids: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class Prompt:
    template_id: str
    messages: tuple[dict[str, str], ...]


class PromptRegistry:
    """Build the sole system/user chat contract from protected fields."""

    def build(self, example: ProtectedExample) -> Prompt:
        system = example.prompt_payload.get("system")
        if not isinstance(system, str) or not system.strip():
            raise GenerationError("protected prompt is missing its system instruction")
        return Prompt(
            f"{example.domain}-v{PROMPT_VERSION}",
            ({"role": "system", "content": system},
             {"role": "user", "content": example.prompt_text}),
        )


@dataclass(frozen=True, slots=True)
class GenerationRegime:
    name: Literal["non_thinking", "thinking"]
    thinking_enabled: bool
    do_sample: bool
    max_new_tokens: int
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None

    @classmethod
    def primary(cls, domain: str) -> GenerationRegime:
        return cls("non_thinking", False, False,
                   {"gsm8k": 256, "proofwriter": 96, "babi": 48}[domain])

    @classmethod
    def thinking(cls) -> GenerationRegime:
        return cls("thinking", True, True, 1024, 0.6, 0.95, 20, 0.0)

    def __post_init__(self) -> None:
        if self.name == "non_thinking" and (self.thinking_enabled or self.do_sample):
            raise GenerationError("primary regime must be greedy non-thinking")
        if self.name == "thinking" and (
            not self.thinking_enabled or not self.do_sample or self.max_new_tokens != 1024
        ):
            raise GenerationError("thinking regime must be sampled and capped at 1024 tokens")

    def parameters(self) -> dict[str, Any]:
        return asdict(self)


def stable_seed(example_id: str, base_seed: int, regime: str) -> int:
    material = canonical_json_bytes({"example_id": example_id, "base_seed": base_seed,
                                     "regime": regime})
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def thinking_subset(example_id: str, *, limit_fraction: int = 256) -> bool:
    """Stable selection primitive; caller applies the per-domain maximum of 256."""
    return int(example_id[:8], 16) % limit_fraction == 0


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _split_thinking(text: str, enabled: bool) -> tuple[str | None, str | None]:
    if not enabled:
        return None, text
    match = re.fullmatch(r"\s*<think>(.*?)</think>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match is None:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


class GenerationRunner:
    def __init__(self, tokenizer: TokenizerProtocol, model: ModelProtocol, *,
                 model_id: str = QWEN3_REPOSITORY, model_revision: str = QWEN3_REVISION,
                 tokenizer_revision: str = QWEN3_REVISION, max_context_tokens: int = 32768,
                 dtype: str = "bfloat16", device: str = "cuda") -> None:
        self.tokenizer, self.model = tokenizer, model
        self.model_id, self.model_revision = model_id, model_revision
        self.tokenizer_revision = tokenizer_revision
        self.max_context_tokens, self.dtype, self.device = max_context_tokens, dtype, device

    def run(self, example: ProtectedExample, regime: GenerationRegime, *, base_seed: int = 0
            ) -> tuple[GenerationRecord | None, FailureRecord | None]:
        prompt = PromptRegistry().build(example)
        ids = self.tokenizer.apply_chat_template(
            list(prompt.messages), tokenize=True, add_generation_prompt=True,
            enable_thinking=regime.thinking_enabled,
        )
        nested_ids = (hasattr(ids, "ndim") and ids.ndim > 1) or (
            len(ids) and isinstance(ids[0], (list, tuple)))
        prompt_ids = list(ids[0] if nested_ids else ids)
        seed = stable_seed(example.example_id, base_seed, regime.name)
        params = regime.parameters()
        gid = generation_id(example_id=example.example_id, model_id=self.model_id,
                            model_revision=self.model_revision,
                            tokenizer_revision=self.tokenizer_revision, regime=regime.name,
                            seed=seed, prompt_token_ids_hash=_hash(prompt_ids),
                            generation_parameters=params)
        if len(prompt_ids) + regime.max_new_tokens > self.max_context_tokens:
            failure = FailureRecord(
                _failure_id("prompt", "prompt_over_budget", example.example_id, gid), "prompt",
                "prompt_over_budget", "prompt plus generation budget exceeds context", False,
                example.example_id, gid, {"prompt_tokens": len(prompt_ids),
                                         "max_new_tokens": regime.max_new_tokens},
            )
            return None, failure
        kwargs = {"max_new_tokens": regime.max_new_tokens, "do_sample": regime.do_sample,
                  "num_return_sequences": 1}
        if regime.do_sample:
            kwargs.update(temperature=regime.temperature, top_p=regime.top_p,
                          top_k=regime.top_k, min_p=regime.min_p,
                          generator=random.Random(seed))
        try:
            output = self.model.generate(ids, **kwargs)
            nested_output = (hasattr(output, "ndim") and output.ndim > 1) or (
                len(output) and isinstance(output[0], (list, tuple)))
            sequence = list(output[0] if nested_output else output)
            generated = sequence[len(prompt_ids):]
            raw = self.tokenizer.decode(generated, skip_special_tokens=False)
        except Exception as error:
            failure = FailureRecord(
                _failure_id("generation", "generation_error", example.example_id, gid),
                "generation", "generation_error", str(error), False, example.example_id, gid,
                {"regime": regime.name},
            )
            return None, failure
        eos = self.tokenizer.eos_token_id
        eos_ids = tuple(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
        exhausted = len(generated) >= regime.max_new_tokens and not (
            generated and generated[-1] in eos_ids
        )
        thinking, answer = _split_thinking(raw, regime.thinking_enabled)
        failure_id = None
        finish = (
            "length" if exhausted else
            "eos" if generated and generated[-1] in eos_ids else "stop"
        )
        return GenerationRecord(
            gid, example.example_id, self.model_id, self.model_revision,
            self.tokenizer_revision, _hash(self.tokenizer.chat_template), _hash(prompt_ids),
            len(prompt_ids), regime.name, regime.thinking_enabled, regime.do_sample, seed,
            regime.temperature, regime.top_p, regime.top_k, regime.min_p, None, None,
            regime.max_new_tokens, eos_ids, self.tokenizer.pad_token_id, eos_ids, self.dtype,
            self.device, raw, tuple(generated), None, len(generated), finish, exhausted,
            thinking, answer, failure_id,
        ), None


def _failure_id(stage: str, code: str, example_id: str, generation_id_value: str) -> str:
    return record_id(stage=stage, code=code, example_id=example_id,
                     generation_id=generation_id_value, context={})


def load_cached_qwen(*, cache_dir: str | None = None) -> tuple[Any, Any]:
    """Load the exact verified local snapshot without a network fallback."""
    snapshot = resolve_cached_snapshot(QWEN3_REPOSITORY, QWEN3_REVISION, cache_dir=cache_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(snapshot.snapshot_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(snapshot.snapshot_path, local_files_only=True)
    return tokenizer, model
