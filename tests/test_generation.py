from __future__ import annotations

from dataclasses import replace

import pytest

from seer.generation import GenerationError, GenerationRegime, GenerationRunner, PromptRegistry
from seer.partitions import ProtectedExample


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
                                     "enable_thinking": False}
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


def test_prompt_budget_exhaustion_is_linked_and_does_not_generate():
    model = FakeGenerator()
    record, failure = GenerationRunner(FakeTokenizer(), model, max_context_tokens=20).run(
        protected("gsm8k"), GenerationRegime.primary("gsm8k")
    )
    assert record is None and failure is not None
    assert failure.code == "prompt_over_budget" and failure.generation_id
    assert model.calls == []


def test_regime_rejects_unbounded_or_mixed_contracts():
    with pytest.raises(GenerationError):
        replace(GenerationRegime.thinking(), max_new_tokens=2048)
