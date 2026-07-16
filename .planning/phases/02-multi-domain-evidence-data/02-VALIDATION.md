---
phase: 2
slug: multi-domain-evidence-data
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-15
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for multi-domain evidence preparation and generation.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/test_evidence.py tests/test_normalization.py tests/test_adapters.py tests/test_partitions.py tests/test_generation.py -q` |
| **Full suite command** | `uv run pytest` |
| **Lint command** | `uv run ruff check .` |
| **Estimated runtime** | Under 120 seconds for automated offline checks |

## Sampling Rate

- **After every task commit:** Run the focused test file(s) named by the task.
- **After every plan wave:** Run `uv run pytest` and `uv run ruff check .`.
- **Before `/gsd-verify-work`:** Full offline suite, Ruff, adapter fixture matrix,
  partition/leakage failures, and fake-generation resume matrix must be green.
- **Max feedback latency:** 120 seconds for automated checks.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 02-01-01 | 01 | 1 | DATA-01 | — | Strict records reject unknown or malformed evidence | unit | `uv run pytest tests/test_evidence.py -q` | ❌ W0 | ⬜ pending |
| 02-01-02 | 01 | 1 | DATA-03 | — | Ambiguous/invalid outputs remain explicit instead of guessed | unit | `uv run pytest tests/test_normalization.py -q` | ❌ W0 | ⬜ pending |
| 02-02-01 | 02 | 2 | DATA-02 | — | Dataset resolution is pinned and opt-in; tests never download | unit/integration | `uv run pytest tests/test_adapters.py -q` | ❌ W0 | ⬜ pending |
| 02-02-02 | 02 | 2 | DATA-02 | — | Prepared shards preserve official identity and validate hashes | integration | `uv run pytest tests/test_adapters.py -q -k 'prepare or lock or split'` | ❌ W0 | ⬜ pending |
| 02-03-01 | 03 | 2 | DATA-04 | — | Protected groups cannot cross partitions | unit | `uv run pytest tests/test_partitions.py -q -k 'partition or overlap or duplicate'` | ❌ W0 | ⬜ pending |
| 02-03-02 | 03 | 2 | DATA-06 | — | Corruptions cannot enter primary natural-result iterators | unit | `uv run pytest tests/test_partitions.py -q -k 'corruption or natural'` | ❌ W0 | ⬜ pending |
| 02-04-01 | 04 | 3 | DATA-07 | — | Regime, seed, template, and token budgets are immutable and traceable | unit/integration | `uv run pytest tests/test_generation.py -q -k 'prompt or regime or budget'` | ❌ W0 | ⬜ pending |
| 02-04-02 | 04 | 3 | DATA-05, DATA-07 | — | Resume is duplicate-free and insufficiency fails closed | integration | `uv run pytest tests/test_generation.py -q -k 'resume or sufficiency or natural'` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

- [ ] `tests/test_evidence.py` — record schema, strict codec, canonical ID, and
  synthetic compatibility tests for DATA-01.
- [ ] `tests/test_normalization.py` — GSM8K rational, ProofWriter categorical,
  bAbI entity, ambiguity, malformed, and thinking-output fixtures for DATA-03.
- [ ] `tests/test_adapters.py` — source-shaped row fixtures, exact revision/config,
  split preservation, lock, and download-boundary tests for DATA-02.
- [ ] `tests/test_partitions.py` — group hashing, row-order invariance, duplicate,
  conflicting-gold, leakage, protected access, and corruption tests for
  DATA-04/DATA-06.
- [ ] `tests/test_generation.py` — fake tokenizer/model prompt snapshots,
  non-thinking/thinking regimes, budgets, resume, generated-token counts, and
  99/100 class-sufficiency boundary tests for DATA-05/DATA-07.

The repository already has pytest, Ruff, strict config/runtime primitives, an
offline model resolver, and fake/synthetic patterns. Wave 0 adds focused tests and
the locked `datasets` dependency but no new test framework.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Resolve and inspect each pinned dataset source | DATA-02 | Requires explicit network consent and upstream Hub state | Run `seer prepare-data ... --allow-download` against an empty temporary cache; verify full commits, source hashes, features, official splits/counts, fingerprints, and licenses in `dataset-lock.json`. Stop on any mismatch. |
| Review protected corpus audit before generation | DATA-04, DATA-06 | Real-source overlap and provenance depend on prepared artifacts | Confirm zero cross-partition content/group overlap, quarantined conflicting gold, separate corruption shard/count, and matching audit hashes. |
| Generate one cached-Qwen example per domain and regime | DATA-03, DATA-07 | Loads the real 0.6B checkpoint and uses local GPU resources | In offline mode, generate one golden example for bAbI, ProofWriter, and GSM8K under non-thinking and thinking regimes; inspect rendered prompts, answer/thinking separation, token counts, finish reasons, and peak VRAM. |
| Confirm natural class sufficiency | DATA-05 | Depends on real model outputs over confirmatory corpora | Build the per-domain/regime report and verify eligibility is `eligible` only with at least 100 correct and 100 incorrect natural scored records; accept `underpowered` as a valid Phase 2 outcome. |

## Validation Sign-Off

- [x] Every anticipated task has automated verification or manual-only rationale.
- [x] Sampling continuity has no three consecutive tasks without automated verification.
- [x] Wave 0 identifies every missing test reference.
- [x] Commands contain no watch-mode flags.
- [x] Automated feedback latency target is under 120 seconds.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** approved 2026-07-15
