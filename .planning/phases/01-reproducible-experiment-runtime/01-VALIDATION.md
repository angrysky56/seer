---
phase: 1
slug: reproducible-experiment-runtime
status: draft
nyquist_compliant: true
wave_0_complete: true
created: 2026-07-15
---

# Phase 1 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 8.3+ |
| **Config file** | `pyproject.toml` (`[tool.pytest.ini_options]`) |
| **Quick run command** | `uv run pytest tests/test_config.py tests/test_runtime.py tests/test_cli.py -q` |
| **Full suite command** | `uv run pytest` |
| **Lint command** | `uv run ruff check .` |
| **Estimated runtime** | Under 60 seconds for quick tests; under 120 seconds for the offline full suite |

## Sampling Rate

- **After every task commit:** Run the focused test file(s) named in the task.
- **After every plan wave:** Run `uv run pytest` and `uv run ruff check .`.
- **Before `/gsd-verify-work`:** The offline full suite, lint, and synthetic CLI
  acceptance matrix must be green.
- **Max feedback latency:** 120 seconds for automated checks.

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | EXP-01 | — | Reject malformed/unknown config instead of guessing | unit | `uv run pytest tests/test_config.py -q` | ❌ W0 | ⬜ pending |
| 01-01-02 | 01 | 1 | EXP-02, QUAL-04 | — | CLI errors are explicit and smoke stays synthetic/offline | unit/integration | `uv run pytest tests/test_cli.py -q` | ❌ W0 | ⬜ pending |
| 01-02-01 | 02 | 1 | EXP-03 | — | Hash mismatch prevents artifact promotion | unit | `uv run pytest tests/test_runtime.py -q -k 'manifest or hash or tamper'` | ❌ W0 | ⬜ pending |
| 01-02-02 | 02 | 1 | EXP-04 | — | Completed evidence is immutable; incompatible resume fails | unit | `uv run pytest tests/test_runtime.py -q -k 'resume or complete or lock'` | ❌ W0 | ⬜ pending |
| 01-03-01 | 03 | 2 | EXP-06 | — | Offline cache miss never falls back to network | unit | `uv run pytest tests/test_model.py tests/test_cache.py -q` | ❌ W0 | ⬜ pending |
| 01-04-01 | 04 | 2 | EXP-05, QUAL-01 | — | Full unit suite has no external model/data dependency | integration | `uv run pytest` | ✅ existing | ⬜ pending |
| 01-04-02 | 04 | 2 | QUAL-04 | — | Clean and resumed smoke runs have equal scientific hashes | integration | `uv run pytest tests/test_smoke.py -q` | ❌ W0 | ⬜ pending |
| 01-04-03 | 04 | 2 | DOC-01, QUAL-02 | — | Documented commands are valid and lint is green | lint/source | `uv run ruff check .` | ✅ existing | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

## Wave 0 Requirements

- [ ] `tests/test_config.py` — strict serialization, validation, and canonical
  digest tests for EXP-01.
- [ ] `tests/test_cli.py` — parser, dispatch, exit-code, and synthetic smoke tests
  for EXP-02 and QUAL-04.
- [ ] `tests/test_runtime.py` — atomic writes, state transitions, locks, artifact
  hashing, immutability, and resume compatibility for EXP-03/EXP-04.
- [ ] `tests/test_cache.py` — injected exact-revision resolver and offline failure
  tests for EXP-06.
- [ ] `tests/test_smoke.py` — uninterrupted versus interrupted/resumed scientific
  equivalence and completed-run no-op assertions for QUAL-04.

The repository already includes pytest, Ruff, fake model infrastructure, and a
synthetic dataset; Wave 0 adds test files but no new test framework.

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Exact cached Qwen3 snapshot loads locally | EXP-06 | CI must not depend on the developer's Hugging Face cache or GPU | Run the documented cached-model preflight with offline mode; confirm snapshot `c1899de289a04d12100db370d81485cdf75e47ca`, then perform one minimal local forward pass. |
| Opt-in download guidance is actionable | DOC-01 | Tests must not make a real network request | Temporarily point the preflight at an empty cache, verify the command exits before model construction and prints the exact repository, revision, searched cache, and explicit download command. |

## Validation Sign-Off

- [x] All anticipated tasks have automated verification or explicit manual-only rationale.
- [x] Sampling continuity has no three consecutive tasks without automated verification.
- [x] Wave 0 identifies every missing test reference.
- [x] Commands contain no watch-mode flags.
- [x] Feedback latency target is under 120 seconds.
- [x] `nyquist_compliant: true` is set in frontmatter.

**Approval:** approved 2026-07-15
