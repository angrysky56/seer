"""Bounded, label-blind generation for immutable Phase 2 evidence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import torch

from seer.cache import QWEN3_REPOSITORY, QWEN3_REVISION, resolve_cached_snapshot
from seer.evidence import (
    FailureRecord,
    GenerationRecord,
    TaskExample,
    canonical_json_bytes,
    decode_jsonl,
    encode_jsonl,
    generation_id,
    record_id,
)
from seer.normalization import score_generation
from seer.partitions import GoldScorer, ProtectedExample, protected_generation_view
from seer.runtime import atomic_write_bytes, atomic_write_json, sha256_file
from seer.sufficiency import build_sufficiency_report

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
            enable_thinking=regime.thinking_enabled, return_tensors="pt",
        )
        if isinstance(ids, Mapping):
            if "input_ids" not in ids:
                raise GenerationError("chat template did not return input_ids")
            ids = ids["input_ids"]
        nested_ids = (hasattr(ids, "ndim") and ids.ndim > 1) or (
            len(ids) and isinstance(ids[0], (list, tuple)))
        prompt_ids = [int(item) for item in (ids[0] if nested_ids else ids)]
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
                                         "max_new_tokens": regime.max_new_tokens,
                                         "regime": regime.name},
            )
            return None, failure
        kwargs = {"max_new_tokens": regime.max_new_tokens, "do_sample": regime.do_sample,
                  "num_return_sequences": 1}
        if regime.do_sample:
            generator_device = self.device if self.device != "cuda" or torch.cuda.is_available() \
                else "cpu"
            kwargs.update(temperature=regime.temperature, top_p=regime.top_p,
                          top_k=regime.top_k, min_p=regime.min_p,
                          generator=torch.Generator(device=generator_device).manual_seed(seed))
        try:
            model_ids = ids.to(self.device) if hasattr(ids, "to") else ids
            output = self.model.generate(model_ids, **kwargs)
            nested_output = (hasattr(output, "ndim") and output.ndim > 1) or (
                len(output) and isinstance(output[0], (list, tuple)))
            sequence = [int(item) for item in (output[0] if nested_output else output)]
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
    if not torch.cuda.is_available():
        raise GenerationError("cached-Qwen GPU operation requires CUDA")
    model = AutoModelForCausalLM.from_pretrained(
        snapshot.snapshot_path, local_files_only=True, dtype=torch.bfloat16).to("cuda")
    model.eval()
    return tokenizer, model


def _prepared_examples(root: Path) -> tuple[TaskExample, ...]:
    if not (root / "COMPLETE").is_file():
        raise GenerationError("prepared corpus is incomplete")
    manifest = json.loads((root / "manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        path = root / artifact["path"]
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            raise GenerationError(f"prepared artifact mutation: {artifact['path']}")
    records = []
    for path in sorted((root / "examples").glob("*.jsonl")):
        decoded = decode_jsonl(path.read_bytes())
        records.extend(item for item in decoded if isinstance(item, TaskExample))
    return tuple(records)


def _identity(prepared_root: Path, runner: GenerationRunner,
              regimes: tuple[GenerationRegime, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "prepared_complete_sha256": sha256_file(prepared_root / "COMPLETE"),
        "dataset_lock_sha256": sha256_file(prepared_root / "dataset-lock.json"),
        "partition_manifest_sha256": sha256_file(prepared_root / "partition-manifest.json"),
        "leakage_audit_sha256": sha256_file(prepared_root / "leakage-audit.json"),
        "model_id": runner.model_id, "model_revision": runner.model_revision,
        "tokenizer_revision": runner.tokenizer_revision,
        "chat_template_sha256": _hash(runner.tokenizer.chat_template),
        "regimes": [item.parameters() for item in regimes],
    }


def cache_outputs(prepared_root: str | Path, output_root: str | Path,
                  runner: GenerationRunner, *, regimes: tuple[GenerationRegime, ...] | None = None,
                  resume: bool = False, replace_existing: bool = False,
                  interrupt_after: int | None = None,
                  smoke_per_domain_regime: int | None = None) -> Path:
    """Generate label-blind shards, seal them, then open gold solely for scoring."""
    prepared, output = Path(prepared_root), Path(output_root)
    examples = _prepared_examples(prepared)
    if smoke_per_domain_regime not in (None, 1):
        raise GenerationError("smoke selector is bounded to exactly one example")
    regimes = regimes or (GenerationRegime.primary("gsm8k"),)
    identity = _identity(prepared, runner, regimes)
    if output.exists() and replace_existing:
        import shutil
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(exist_ok=True)
    identity_path = artifacts / "generation-identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise GenerationError("generation identity drift")
        if not resume and not (output / "COMPLETE").exists():
            raise GenerationError("incomplete generation exists; use resume")
    else:
        atomic_write_json(identity_path, identity)
    if (output / "COMPLETE").exists():
        index = json.loads((artifacts / "generation-index.json").read_text())
        for relative, expected in index["sealed_hashes"].items():
            if sha256_file(output / relative) != expected:
                raise GenerationError(f"sealed artifact mutation: {relative}")
        return output
    generations_path, failures_path = artifacts / "generations.jsonl", artifacts / "failures.jsonl"
    existing_gens = (
        list(decode_jsonl(generations_path.read_bytes())) if generations_path.exists() else [])
    existing_failures = (
        list(decode_jsonl(failures_path.read_bytes())) if failures_path.exists() else [])
    done = {(item.example_id, item.regime) for item in existing_gens
            if isinstance(item, GenerationRecord)} | {
        (item.example_id, str(item.context.get("regime", "non_thinking")))
        for item in existing_failures if isinstance(item, FailureRecord)}
    generated = [item for item in existing_gens if isinstance(item, GenerationRecord)]
    failures = [item for item in existing_failures if isinstance(item, FailureRecord)]
    count = 0
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    protected = tuple(protected_generation_view(examples))
    if smoke_per_domain_regime:
        selected = []
        for domain in ("gsm8k", "proofwriter", "babi"):
            candidate = next(item for item in protected
                             if item.domain == domain and item.partition == "confirmatory_test")
            selected.extend(((candidate, GenerationRegime.primary(domain)),
                             (candidate, GenerationRegime.thinking())))
    else:
        selected = [(item, GenerationRegime.primary(item.domain)) for item in protected]
    for item, regime in selected:
        if (item.example_id, regime.name) in done:
            continue
        record, failure = runner.run(item, regime)
        if record is not None:
            generated.append(record)
        if failure is not None:
            failures.append(failure)
        atomic_write_bytes(generations_path, encode_jsonl(tuple(generated)))
        atomic_write_bytes(failures_path, encode_jsonl(tuple(failures)))
        count += 1
        if interrupt_after is not None and count >= interrupt_after:
            raise GenerationError("injected generation interruption")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    generated_tokens = sum(item.generated_token_count for item in generated)
    sealed = {path.relative_to(output).as_posix(): sha256_file(path)
              for path in (generations_path, failures_path, identity_path)}
    atomic_write_json(artifacts / "generation-index.json",
                      {"schema_version": 1, "generation_ids": sorted(
                          item.generation_id for item in generated), "sealed_hashes": sealed,
                       "operational_metrics": {
                           "elapsed_seconds": elapsed,
                           "generated_tokens": generated_tokens,
                           "generated_tokens_per_second": generated_tokens / elapsed,
                           "peak_cuda_memory_bytes": (
                               torch.cuda.max_memory_allocated()
                               if torch.cuda.is_available() else None),
                       }})
    # Only after the seal exists may the narrow capability reveal gold to the scorer.
    scorer = GoldScorer(examples)
    by_id = {item.example_id: item for item in examples}
    scores = [score_generation(item, scorer.gold_for(item.example_id),
                               domain=by_id[item.example_id].domain) for item in generated]
    scores_path = artifacts / "scores.jsonl"
    atomic_write_bytes(scores_path, encode_jsonl(tuple(scores)))
    report = build_sufficiency_report(generated, scores, failures,
                                      {item.example_id: (item.domain, item.partition)
                                       for item in examples})
    atomic_write_json(artifacts / "sufficiency-report.json", report.to_dict())
    # Revalidate the immutable generation boundary before exposing completion.
    for relative, expected in sealed.items():
        if sha256_file(output / relative) != expected:
            raise GenerationError(f"sealed artifact mutation: {relative}")
    atomic_write_bytes(output / "COMPLETE", canonical_json_bytes({
        "generation_index": sha256_file(artifacts / "generation-index.json"),
        "scores": sha256_file(scores_path),
        "sufficiency": sha256_file(artifacts / "sufficiency-report.json")}) + b"\n")
    return output
