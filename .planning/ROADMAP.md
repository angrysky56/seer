# Roadmap — SEER Path A Evidence Milestone

## Overview

This roadmap turns the existing Path A skeleton into a reproducible experimental
program. Phases are ordered by evidentiary dependency: trustworthy data and
instrumentation precede signal comparisons; exploratory evaluation precedes a
frozen confirmatory run; correction is conditional on transfer success.

| Phase | Name | Goal | Requirements | Gate |
|---:|---|---|---:|---|
| 1 | Reproducible Experiment Runtime | Establish offline-capable, resumable, auditable execution | 11 | Runtime smoke gate |
| 2 | Multi-Domain Evidence Data | Build leakage-safe adapters and natural-error corpora | 7 | Data sufficiency gate |
| 3 | Concept Signals | Implement and diagnose energy, disagreement, and geometry signals | 6 | Signal validity gate |
| 4 | Matched Baselines and Calibration | Put all comparison signals under one fair protocol | 6 | Comparison readiness gate |
| 5 | Evaluation Integrity | Implement statistics, controls, reports, and fail-closed validation | 10 | Gate A |
| 6 | Confirmatory Transfer | Run the frozen experiment and decide the energy-transfer hypothesis | 3 | Gate B |
| 7 | Conditional Correction and Decision | Test correction if warranted and close the milestone honestly | 4 | Gate C or fallback |

## Phase 1 — Reproducible Experiment Runtime

**Status:** Complete (verified 2026-07-16)

**Goal:** Any experiment arm can be configured, executed, resumed, audited, and
reproduced without coupling the unit suite to network or model downloads.

**Requirements:** EXP-01, EXP-02, EXP-03, EXP-04, EXP-05, EXP-06, DOC-01,
QUAL-01, QUAL-02, QUAL-04

### Deliverables

- Versioned configuration schema and CLI command surface.
- Offline/local-files-only Qwen3 loader pinned to cached snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Transformers dependency floor updated for Qwen3 support.
- Atomic run directories, manifests, hashes, completion markers, and resume rules.
- CPU synthetic end-to-end command emitting production-shaped artifacts.
- Setup and experiment-operation documentation.

### Success Criteria

1. A synthetic run completes twice with identical validated records.
2. Interrupting and resuming a run neither duplicates nor overwrites completed work.
3. Offline mode never attempts network access and produces an actionable cache error.
4. Unit tests and Ruff pass without loading external weights or datasets.

### Plans

**Wave 1**

- `01-01-PLAN.md` — strict versioned configuration and complete CLI contract.

**Wave 2** *(blocked on Wave 1 completion)*

- `01-02-PLAN.md` — transactional run store, provenance, integrity, locking, and resume.
- `01-03-PLAN.md` — exact-revision offline Qwen3 cache and loader boundary.

**Wave 3** *(blocked on Waves 1–2 completion)*

- `01-04-PLAN.md` — deterministic synthetic vertical slice, operations docs, and quality gate.

Cross-cutting constraints:

- Tests and the synthetic smoke path remain offline and independent of cached
  weights, external datasets, and GPU availability.
- Runtime state fails closed: completed evidence is immutable, incompatible
  resumes are rejected, and `COMPLETE` follows artifact validation.
- Config, snapshot, checkpoint, seed, and artifact identities remain explicit and
  auditable across every plan.

## Phase 2 — Multi-Domain Evidence Data

**Status:** Ready for planning

**Goal:** Produce normalized, pinned, leakage-audited natural prediction records
for bAbI, ProofWriter, and GSM8K under controlled Qwen3 generation regimes.

**Requirements:** DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-07

### Deliverables

- Shared task example, generation, score, and failure record schemas.
- Tested domain adapters and answer normalizers.
- Deterministic partition planner for signal training, model selection,
  calibration, and untouched confirmatory tests.
- Duplicate/overlap audit and class-sufficiency report.
- Natural-error corpus with separate constructed-corruption provenance.
- Non-thinking primary generation and bounded thinking-mode secondary generation.

### Success Criteria

1. Golden normalization fixtures cover common valid, invalid, and ambiguous outputs.
2. No normalized example crosses protected partitions.
3. Each primary confirmatory domain has at least 100 correct and 100 incorrect
   natural predictions, or is explicitly marked underpowered before signal fitting.
4. Every generation is traceable to prompt, regime, parameters, and token budget.

## Phase 3 — Concept Signals

**Goal:** Implement energy-family signals on a shared concept representation and
prove that their outputs are numerically healthy before comparing performance.

**Requirements:** SIG-01, SIG-02, SIG-03, SIG-04, SIG-05, SIG-06

### Deliverables

- Commit-position residual extraction compatible with Qwen3 and fake models.
- Learned scalar energy with shuffled and hard/near-error negative strategies.
- Seeded predictor-disagreement estimator.
- Regularized representation distance/density baseline.
- Collapse diagnostic stream and report.
- Joint task-anchor path and frozen-versus-trainable trunk regression measurement.

### Success Criteria

1. Signal tensors have declared shapes, finite values, deterministic seeded output,
   and correct gradient boundaries.
2. Negative-strategy provenance is preserved through every learned-energy score.
3. Collapse diagnostics detect injected constant and rank-deficient representations.
4. Trainable-trunk runs cannot be promoted without task-regression results.

## Phase 4 — Matched Baselines and Calibration

**Goal:** Make energy compete against the strongest eligible alternative under
identical examples, partitions, commit positions, and reporting rules.

**Requirements:** BASE-01, BASE-02, BASE-03, BASE-04, CAL-01, CAL-02

### Deliverables

- Token probability/log-probability, self-consistency, learned linear probe, and
  optional verbalized-confidence implementations.
- Fixed sampling budgets and missing-response policies.
- Domain-aware monotone calibration fit only on calibration partitions.
- Unified signal record consumed by the evaluator.

### Success Criteria

1. Every primary example has all eligible baseline scores or an explicit reason
   the score is missing.
2. The learned probe and calibrator cannot access held-out confirmatory labels.
3. Calibration changes probability calibration but not raw ranking metrics.
4. Baseline compute and sampling budgets are recorded and reproducible.

## Phase 5 — Evaluation Integrity

**Goal:** Build a fail-closed evidence pipeline that detects leakage, alignment
errors, invalid controls, missing arms, and unsupported gate decisions.

**Requirements:** EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06,
GATE-01, QUAL-03, QUAL-05

### Deliverables

- Per-domain/per-seed metrics and coverage-risk analysis.
- Paired bootstrap confidence intervals and baseline deltas.
- Label-shuffle and feature/label-alignment controls.
- Negative-strategy-by-domain analysis matrix.
- Canonical result schema plus generated Markdown report.
- Evidence validator and Gate A calculator.
- Deterministic tests for metrics, controls, splits, resume, and gate failures.

### Success Criteria

1. Synthetic known-good evidence passes and injected leakage/misalignment fails.
2. Shuffle-control AUROC outside `[0.45, 0.55]` invalidates the affected result.
3. JSON and Markdown outputs agree because both derive from the same records.
4. Gate A cannot pass with missing arms, incomplete manifests, or underpowered
   results presented as conclusive.

## Phase 6 — Confirmatory Transfer

**Goal:** Freeze the protocol, run the three-seed bAbI-to-ProofWriter/GSM8K
experiment, and make the Path A transfer decision without post-hoc threshold drift.

**Requirements:** GATE-02, GATE-03, DOC-02

### Deliverables

- Signed/frozen confirmatory protocol and thresholds.
- Completed confirmatory matrix for all eligible signals and controls.
- Gate B calculation with per-domain evidence and paired uncertainty.
- Initial decision record stating pass, fail, or inconclusive and why.

### Success Criteria

1. Gate A passes before confirmatory results are interpreted.
2. All confirmatory runs use the frozen model, data revisions, splits, and metrics.
3. Gate B is calculated mechanically from predeclared thresholds.
4. A failure or inconclusive result routes to the Phase 7 fallback branch; it does
   not trigger threshold relaxation or unbounded tuning.

## Phase 7 — Conditional Correction and Decision

**Goal:** If Gate B passes, determine whether energy can safely guide actual
answer correction. Otherwise, close Path A with the strongest supported fallback.

**Requirements:** CORR-01, CORR-02, CORR-03, CORR-04, DOC-03, DOC-04

### Branch A — Gate B Passed

- Implement bounded concept-space energy descent with clipping and rollback.
- Decode/rescore refined concepts and compare against equal-compute controls.
- Calculate Gate C from held-out natural errors.
- If Gate C passes, produce a bounded Path B design and scale-check proposal.
- If Gate C fails, document energy as a triage signal without correction claims.

### Branch B — Gate B Failed or Inconclusive

- Do not implement or claim energy-descent correction.
- Document domain-matched correctness heads plus external admission gating as the
  supported fallback.
- Record what evidence would be required to reopen Path B.

### Success Criteria

1. Correction claims require changed decoded answers, not merely lower energy.
2. Gate C compares against matched-compute controls and passes mechanically.
3. The final decision record identifies the supported architecture branch,
   limitations, negative results, and deferred work.
4. The milestone closes successfully under either branch with no ambiguous next step.

## Requirement Traceability

| Requirement | Owning Phase |
|---|---:|
| EXP-01 | 1 |
| EXP-02 | 1 |
| EXP-03 | 1 |
| EXP-04 | 1 |
| EXP-05 | 1 |
| EXP-06 | 1 |
| DATA-01 | 2 |
| DATA-02 | 2 |
| DATA-03 | 2 |
| DATA-04 | 2 |
| DATA-05 | 2 |
| DATA-06 | 2 |
| DATA-07 | 2 |
| SIG-01 | 3 |
| SIG-02 | 3 |
| SIG-03 | 3 |
| SIG-04 | 3 |
| SIG-05 | 3 |
| SIG-06 | 3 |
| BASE-01 | 4 |
| BASE-02 | 4 |
| BASE-03 | 4 |
| BASE-04 | 4 |
| CAL-01 | 4 |
| CAL-02 | 4 |
| EVAL-01 | 5 |
| EVAL-02 | 5 |
| EVAL-03 | 5 |
| EVAL-04 | 5 |
| EVAL-05 | 5 |
| EVAL-06 | 5 |
| GATE-01 | 5 |
| GATE-02 | 6 |
| GATE-03 | 6 |
| CORR-01 | 7 |
| CORR-02 | 7 |
| CORR-03 | 7 |
| CORR-04 | 7 |
| DOC-01 | 1 |
| DOC-02 | 6 |
| DOC-03 | 7 |
| DOC-04 | 7 |
| QUAL-01 | 1 |
| QUAL-02 | 1 |
| QUAL-03 | 5 |
| QUAL-04 | 1 |
| QUAL-05 | 5 |

**Coverage:** 47/47 requirements mapped; each has one owning phase.

## After Initialization

Run `/gsd-plan-phase 1` to produce the executable plan for the reproducible
experiment runtime.
