---
phase: 01-reproducible-experiment-runtime
verified: 2026-07-16T00:00:00Z
status: passed
score: 4/4 success criteria verified
requirements_coverage: 10/10 satisfied
behavior_unverified: 0
---

# Phase 1: Reproducible Experiment Runtime Verification Report

**Phase Goal:** Any experiment arm can be configured, executed, resumed, audited, and
reproduced without coupling the unit suite to network or model downloads.

**Status:** passed

## Goal Achievement

### Observable Truths

| # | Roadmap success criterion | Status | Independent evidence |
|---:|---|---|---|
| 1 | A synthetic run completes twice with identical validated records. | ✓ VERIFIED | A clean CLI run produced the documented production-shaped tree. Re-running it was a successful immutable no-op and preserved SHA-256 for config, results, JSONL records, and checkpoint. |
| 2 | Interrupting and resuming neither duplicates nor overwrites completed work. | ✓ VERIFIED | Forced interruption after step 2 followed by `resume=True` produced scientific hash `77c9088...c4a8`, exactly equal to the uninterrupted run. Runtime tests also exercise incompatible resume, completed-run refusal, replacement, and locking. |
| 3 | Offline mode never attempts network access and cache failures are actionable. | ✓ VERIFIED | Actual `HF_HUB_OFFLINE=1` preflight resolved only the pinned local snapshot `c1899de...e47ca`; injected cache-miss tests assert `local_files_only=True` and the diagnostic includes repository, revision, searched cache, and explicit opt-in download command. |
| 4 | Unit tests and Ruff pass without external weights or datasets. | ✓ VERIFIED | `78 passed` in 3.33 s; Ruff passed; the synthetic path never imports a model/dataset downloader. |

**Score:** 4/4 success criteria verified

### Required Artifacts

| Artifact | Status | Details |
|---|---|---|
| `src/seer/config.py` | ✓ SUBSTANTIVE + WIRED | Strict nested dataclass decoding, field-qualified validation, canonical serialization, and digesting are consumed by CLI/runtime/smoke. |
| `src/seer/cli.py` | ✓ SUBSTANTIVE + WIRED | All declared phase command names parse; `smoke` dispatches end to end and later-phase commands fail explicitly rather than pretending to run. |
| `src/seer/runtime.py` | ✓ SUBSTANTIVE + WIRED | Atomic writes, provenance, artifact inventory/validation, completion marker, lifecycle, lock, replacement, and checkpoint primitives are used by smoke execution. |
| `src/seer/cache.py`, `src/seer/model.py` | ✓ SUBSTANTIVE + WIRED | Exact-revision resolver is local-only; verified snapshot identity is checked before the loader receives a local path with `local_files_only=True`. |
| `src/seer/smoke.py` | ✓ SUBSTANTIVE + WIRED | Synthetic fake-model training uses the production config, runtime, checkpoint, manifest, and result-envelope seams. |
| `docs/OPERATIONS.md`, `examples/synthetic.json` | ✓ SUBSTANTIVE | Setup, smoke, resume, replacement, integrity, local preflight, manual forward, and explicit download commands are documented with a runnable config. |
| `tests/test_config.py`, `test_cli.py`, `test_runtime.py`, `test_cache.py`, `test_smoke.py`, `test_train.py` | ✓ SUBSTANTIVE | Deterministic success and failure paths cover every Phase 1 behavior. |

### Key Link Verification

| From | To | Status | Evidence |
|---|---|---|---|
| CLI config path/overrides | strict effective config | ✓ WIRED | `load_config` and dataclass replacement precede dispatch. |
| CLI `smoke` | synthetic runner | ✓ WIRED | Lazy import calls `run_smoke`; failures return an explicit nonzero exit. |
| Smoke runner | run store/checkpoint/manifest | ✓ WIRED | `RunStore.prepare`, `RunLock`, checkpoint restore/write, artifact inventory, and `finalize_run` form one transaction. |
| Completed-run no-op | artifact verifier | ✓ WIRED | Existing manifest, config digest, every artifact hash, and `COMPLETE` are validated before success. Manual tampering returned exit 2. |
| Model config | cache resolver | ✓ WIRED | Repository, exact revision, and cache directory propagate to local-only resolution. |
| Verified snapshot | Transformers loader | ✓ WIRED | Snapshot identity is rechecked and only the resolved local path is passed to the loader. |

## Requirements Coverage

| Requirement | Status | Evidence |
|---|---|---|
| EXP-01 | ✓ SATISFIED | Typed config covers model, dataset/splits, seeds, energy, optimizer, calibration, runtime, and output; strict round-trip and rejection tests pass. |
| EXP-02 | ✓ SATISFIED | The complete milestone command surface exists. Phase 1's executable vertical slice is `smoke`; commands owned by later phases deliberately return an explicit not-implemented error. |
| EXP-03 | ✓ SATISFIED | Completed smoke manifest records effective config digest, git state, dependency/environment/hardware facts, model/dataset IDs and revisions, seeds, timestamps, and relative artifact hashes. |
| EXP-04 | ✓ SATISFIED | Checkpoints include model/optimizer/RNG/cursor/data-order state; resumed science equals clean science; completed evidence is immutable unless explicit replacement is requested. |
| EXP-05 | ✓ SATISFIED | Full tests complete offline with fakes/synthetic data and no weight or dataset access. |
| EXP-06 | ✓ SATISFIED | Dependency floor is `transformers>=4.51` (lock resolves 5.14.0); exact cached Qwen3 preflight passed locally; cache failures are fail-closed and actionable. |
| DOC-01 | ✓ SATISFIED | Operations guide documents setup, explicit download, smoke/confirmatory boundary, resume/replace behavior, cache checks, and artifact interpretation. |
| QUAL-01 | ✓ SATISFIED | `UV_CACHE_DIR=/tmp/seer-uv-cache uv run pytest -q`: 78 passed. |
| QUAL-02 | ✓ SATISFIED | `UV_CACHE_DIR=/tmp/seer-uv-cache uv run ruff check .`: all checks passed. |
| QUAL-04 | ✓ SATISFIED | CPU synthetic experiment emitted the real config/manifest/checkpoint/result schemas; clean and resumed scientific hashes matched. |

**Coverage:** 10/10 requirements satisfied

## Command Evidence

| Check | Result |
|---|---|
| Full offline suite | 78 passed in 3.33 s (five environment-only NVML warnings) |
| Ruff | All checks passed |
| Lock consistency | `uv lock --check` resolved successfully |
| CLI help | All seven milestone subcommands listed |
| Transformers floor | Installed version 5.14.0 |
| CPU CLI smoke | Exit 0; config, state, manifest, completion marker, checkpoint, JSON result, and JSONL records emitted |
| Completed re-run | Exit 0; all four inventoried payload hashes unchanged |
| Tamper check | Modified `results.json` rejected with size mismatch; exit 2 |
| Resume equivalence | Forced step-2 interruption resumed to the same scientific SHA-256 as clean execution |
| Real cached preflight | Offline resolution returned `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca` and verified metadata |

## Anti-Patterns and Scope Notes

No blocking stubs, TODOs, silent network fallbacks, or mutable evidence paths were found in
the Phase 1 executable slice. Non-smoke CLI handlers intentionally stop with a clear
not-implemented diagnostic because their data, signal, evaluation, and reporting behavior is
owned by Phases 2-5; the Phase 1 roadmap promises their command contract, not premature fake
implementations.

## Manual-Only Items

- A real Qwen3 forward pass was not run. It would load the full cached 0.6B checkpoint and is
  explicitly manual-only in the approved validation plan. The safety-critical preflight and
  exact loader argument propagation were verified without loading weights. This does not block
  Phase 1.
- Network download guidance was not executed. Cache-miss behavior and the rendered opt-in command
  are covered offline; intentionally downloading weights would violate this verification's scope.

## Gaps Summary

**No gaps found.** The runtime smoke gate is achieved and Phase 2 can build on the normalized
configuration, transactional artifact, resumable checkpoint, and offline-model boundaries.

## Verification Metadata

**Approach:** Goal-backward verification from the ROADMAP success criteria, followed by literal
requirement coverage and adversarial execution against actual source.

**Human checks required:** 0 blocking checks.

---
*Verified: 2026-07-16*
*Verifier: Codex phase verifier*
