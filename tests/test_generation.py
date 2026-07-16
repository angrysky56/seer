from __future__ import annotations

from dataclasses import replace

import pytest

from seer.evidence import GenerationRecord, ScoredResult, TaskExample, decode_jsonl, encode_jsonl
from seer.generation import (
    GenerationError,
    GenerationRegime,
    GenerationRunner,
    PromptRegistry,
    cache_outputs,
)
from seer.partitions import ProtectedExample
from seer.runtime import atomic_write_bytes, atomic_write_json, sha256_file
from seer.sufficiency import build_sufficiency_report


class FakeTokenizer:
    chat_template = "fake-qwen-chat-v1"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [10, 11, 12]

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "<think>work</think>\nFINAL: true" if 8 in token_ids else "FINAL: true"


class FakeGenerator:
    def __init__(self, suffix=(7, 2)) -> None:
        self.suffix = suffix
        self.calls = []

    def generate(self, input_ids, **kwargs):
        self.calls.append((input_ids, kwargs))
        return [list(input_ids) + list(self.suffix)]


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return {"input_ids": [10, 11, 12]}


def protected(domain="proofwriter"):
    return ProtectedExample("a" * 64, domain, "confirmatory_test", "Question?",
                            {"system": "Solve exactly."}, {})


@pytest.mark.parametrize("domain", ["gsm8k", "proofwriter", "babi"])
def test_prompt_registry_and_primary_regime_contract(domain):
    item = protected(domain)
    prompt = PromptRegistry().build(item)
    assert prompt.template_id == f"{domain}-v1"
    assert [message["role"] for message in prompt.messages] == ["system", "user"]
    tokenizer, model = FakeTokenizer(), FakeGenerator()
    record, failure = GenerationRunner(tokenizer, model).run(item,
                                                              GenerationRegime.primary(domain))
    assert failure is None and record is not None
    assert tokenizer.calls[0][1] == {"tokenize": True, "add_generation_prompt": True,
                                     "enable_thinking": False, "return_tensors": "pt"}
    assert record.prompt_token_count == 3 and record.generated_token_count == 2
    assert record.finish_reason == "eos" and not record.truncated
    assert len(record.chat_template_hash) == len(record.prompt_token_ids_hash) == 64
    assert not model.calls[0][1]["do_sample"]


def test_thinking_regime_separates_thinking_and_answer_and_is_seed_stable():
    tokenizer, model = FakeTokenizer(), FakeGenerator((8, 2))
    runner = GenerationRunner(tokenizer, model)
    first, _ = runner.run(protected(), GenerationRegime.thinking(), base_seed=4)
    second, _ = runner.run(protected(), GenerationRegime.thinking(), base_seed=4)
    assert first is not None and second is not None
    assert first.seed == second.seed and first.generation_id == second.generation_id
    assert first.thinking_text == "work" and first.answer_text == "FINAL: true"
    assert tokenizer.calls[0][1]["enable_thinking"] is True
    assert model.calls[0][1]["max_new_tokens"] == 1024
    assert "generator" not in model.calls[0][1]


def test_prompt_budget_exhaustion_is_linked_and_does_not_generate():
    model = FakeGenerator()
    record, failure = GenerationRunner(FakeTokenizer(), model, max_context_tokens=20).run(
        protected("gsm8k"), GenerationRegime.primary("gsm8k")
    )
    assert record is None and failure is not None
    assert failure.code == "prompt_over_budget" and failure.generation_id
    assert model.calls == []


def test_chat_template_accepts_transformers_batch_encoding_shape():
    record, failure = GenerationRunner(MappingTokenizer(), FakeGenerator()).run(
        protected(), GenerationRegime.primary("proofwriter"))
    assert failure is None and record is not None and record.prompt_token_count == 3


def test_regime_rejects_unbounded_or_mixed_contracts():
    with pytest.raises(GenerationError):
        replace(GenerationRegime.thinking(), max_new_tokens=2048)


@pytest.mark.parametrize("correct,incorrect,status", [
    (99, 100, "underpowered"), (100, 99, "underpowered"), (100, 100, "eligible")])
def test_natural_sufficiency_boundaries(correct, incorrect, status):
    runner = GenerationRunner(FakeTokenizer(), FakeGenerator())
    base, _ = runner.run(protected(), GenerationRegime.primary("proofwriter"))
    assert base is not None
    generations, scores, metadata = [], [], {}
    for index, is_correct in enumerate([True] * correct + [False] * incorrect):
        example_id = f"{index:064x}"
        generation_id = f"{index + 1000:064x}"
        generations.append(replace(base, example_id=example_id, generation_id=generation_id))
        scores.append(ScoredResult(generation_id, "FINAL: true", "true", "true",
                                   is_correct, "scored", "proofwriter", 1,
                                   ("true",), None))
        metadata[example_id] = ("proofwriter", "confirmatory_test")
    report = build_sufficiency_report(generations, scores, (), metadata)
    assert report.status == status


def test_sufficiency_excludes_invalid_and_ambiguous_rows():
    runner = GenerationRunner(FakeTokenizer(), FakeGenerator())
    base, _ = runner.run(protected(), GenerationRegime.primary("proofwriter"))
    assert base is not None
    generations = [replace(base, example_id=f"{index:064x}", generation_id=f"{index:064x}")
                   for index in range(3)]
    scores = [
        ScoredResult(f"{0:064x}", "", None, "true", None, "invalid", "proofwriter", 1, (),
                     "source_gold_invalid"),
        ScoredResult(f"{1:064x}", "", None, "true", None, "ambiguous", "proofwriter", 1, (),
                     "multiple_conflicting_answers"),
    ]
    metadata = {f"{index:064x}": ("proofwriter", "confirmatory_test") for index in range(3)}
    group = build_sufficiency_report(generations, scores, (), metadata).groups[0]
    assert (group.correct, group.incorrect, group.invalid_source, group.ambiguous) == (0, 0, 1, 1)


def prepared_root(tmp_path, count=3, domains=None):
    root = tmp_path / "prepared"
    examples = []
    for index in range(count):
        domain = domains[index] if domains else "proofwriter"
        gold = {"proofwriter": "true", "gsm8k": "1", "babi": "hall"}[domain]
        eid = f"{index + 1:064x}"
        examples.append(TaskExample(
            eid, domain, f"source/{domain}", "7" * 40, "default", "test",
            str(index), str(index), "confirmatory_test", f"{domain}-v1",
            {"system": "Solve exactly."}, "Question?", gold, gold, "categorical", {},
            "b" * 64, "c" * 64, "unknown/not-declared"))
    path = root / "examples" / "proofwriter-confirmatory_test.jsonl"
    atomic_write_bytes(path, encode_jsonl(tuple(examples)))
    for name in ("dataset-lock.json", "partition-manifest.json", "leakage-audit.json"):
        atomic_write_json(root / name, {"schema_version": 1})
    artifacts = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            artifacts.append({"path": item.relative_to(root).as_posix(),
                              "bytes": item.stat().st_size, "sha256": sha256_file(item),
                              "media_type": "application/octet-stream", "schema_type": None})
    atomic_write_json(root / "manifest.json", {"artifacts": artifacts})
    atomic_write_bytes(root / "COMPLETE", b"complete\n")
    return root


def test_smoke_selector_emits_exactly_one_example_per_domain_regime(tmp_path):
    prepared = prepared_root(tmp_path, domains=["gsm8k", "proofwriter", "babi"])
    output = tmp_path / "smoke"
    regimes = tuple(GenerationRegime.primary(domain)
                    for domain in ("gsm8k", "proofwriter", "babi")) + (
                        GenerationRegime.thinking(),)
    cache_outputs(prepared, output, GenerationRunner(FakeTokenizer(), FakeGenerator()),
                  regimes=regimes, smoke_per_domain_regime=1)
    records = [item for item in decode_jsonl(
        (output / "artifacts/generations.jsonl").read_bytes())
        if isinstance(item, GenerationRecord)]
    assert len(records) == 6
    assert {(item.example_id, item.regime) for item in records} == {
        (f"{index:064x}", regime) for index in (1, 2, 3)
        for regime in ("non_thinking", "thinking")}
    assert sorted(item.max_new_tokens for item in records if item.regime == "non_thinking") == [
        48, 96, 256]


def test_resume_is_duplicate_free_and_matches_clean_generation(tmp_path):
    prepared = prepared_root(tmp_path)
    with pytest.raises(GenerationError, match="interruption"):
        cache_outputs(prepared, tmp_path / "resumed",
                      GenerationRunner(FakeTokenizer(), FakeGenerator()), interrupt_after=1)
    cache_outputs(prepared, tmp_path / "resumed",
                  GenerationRunner(FakeTokenizer(), FakeGenerator()), resume=True)
    cache_outputs(prepared, tmp_path / "clean",
                  GenerationRunner(FakeTokenizer(), FakeGenerator()))
    resumed = (tmp_path / "resumed/artifacts/generations.jsonl").read_bytes()
    clean = (tmp_path / "clean/artifacts/generations.jsonl").read_bytes()
    assert resumed == clean
    assert len(resumed.splitlines()) == 3


def test_completed_generation_rejects_sealed_tamper(tmp_path):
    prepared = prepared_root(tmp_path)
    output = tmp_path / "output"
    cache_outputs(prepared, output, GenerationRunner(FakeTokenizer(), FakeGenerator()))
    with (output / "artifacts/generations.jsonl").open("ab") as stream:
        stream.write(b"{}\n")
    with pytest.raises(GenerationError, match="sealed artifact mutation"):
        cache_outputs(prepared, output, GenerationRunner(FakeTokenizer(), FakeGenerator()))


def test_prepared_lock_tamper_fails_before_model_generation(tmp_path):
    prepared = prepared_root(tmp_path)
    model = FakeGenerator()
    atomic_write_json(prepared / "dataset-lock.json", {"tampered": True})
    with pytest.raises(GenerationError, match="prepared artifact mutation"):
        cache_outputs(prepared, tmp_path / "output", GenerationRunner(FakeTokenizer(), model))
    assert model.calls == []
