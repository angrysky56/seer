# SEER — Self-Evaluating Energy Reasoner

## Vision

Build a language-model architecture that can detect when its own internal
prediction is implausible, spend additional inference compute correcting it in
concept space, and abstain or invoke a verifier when correction is insufficient.
SEER combines clean state transport, a native energy/self-certainty channel, and
an admission gate rather than relying on verbalized confidence.

## Current State

This is a brownfield research codebase. The Path A skeleton already includes:

- a pretrained causal-LM wrapper with `concept_proj`, `energy_head`, and
  calibrated `self_certainty` outputs;
- role-aware Muon/AdamW optimizer grouping;
- dense-supervision data interfaces and a synthetic smoke-test dataset;
- joint task/certainty training utilities;
- AUROC, adaptive-ECE, shuffle-control, baseline, and multi-seed evaluation
  primitives;
- deterministic unit tests that use fake models and avoid checkpoint downloads.

It does not yet include real experimental datasets, reproducible experiment
orchestration, trained checkpoints, decision-grade Path A results, energy-descent
correction, verifier integration, or a Path B latent-predictive model.

## Current Milestone

Complete a decision-grade Path A experimental program. Determine whether a
representation-space energy signal predicts correctness across held-out domains
better than token-space probes and standard confidence baselines. If and only if
that result survives controls and replication, test bounded energy-descent
correction and prepare the evidence needed to decide whether Path B is justified.

This milestone is successful when another researcher can reproduce the full
train/evaluate/report pipeline and reach the same go/no-go decision from durable
artifacts, not when the repository merely contains runnable scaffolding.

## Research Questions

1. Does energy discriminate correct from incorrect predictions out of domain,
   where the predecessor token probe fell to approximately chance?
2. Which energy construction transfers best: learned scalar energy, predictor
   disagreement, distance/density in concept space, or a controlled combination?
3. Are gains robust to corruption strategy, domain identity, model size, seed,
   and calibration choices, or does the head exploit synthetic-negative
   shortcuts?
4. On naturally wrong examples with high energy, does bounded latent refinement
   improve task correctness without damaging already-correct examples?
5. If transfer fails, what is the strongest honest fallback: domain-matched
   heads plus symbolic admission gating, or a revised representation objective?

## Principles and Constraints

- Preserve the distinction between validated results and hypotheses.
- Path A precedes Path B; do not spend scale-up compute before local evidence.
- Use joint task and certainty objectives whenever trunk gradients are enabled.
- Report discrimination and calibration separately: AUROC plus adaptive ECE.
- Include matched token-confidence baselines, shuffle-label controls, multiple
  seeds, confidence intervals, and per-domain results.
- Treat negative/corruption generation as an experimental variable and evaluate
  on naturally occurring model errors, not only constructed negatives.
- Keep initial work practical on an RTX 3060 with 12 GB VRAM and 64 GB RAM;
  reserve larger hardware for a predeclared replication or scale check.
- Never download model weights in tests. Unit and integration tests use fakes,
  synthetic fixtures, or explicitly cached checkpoints.
- Use Python 3.12, `uv`, PyTorch, Transformers, pytest, and Ruff.
- Failed hypotheses are valid results. Preserve the explicit off-ramp rather than
  optimizing metrics until the architecture appears successful.

## Milestone Boundaries

### In Scope

- reproducible configuration, CLI/orchestration, checkpointing, and result
  manifests;
- real multi-domain task adapters and contamination-safe splits;
- Path A energy variants and controlled negative-generation strategies;
- token and uncertainty baselines under a common evaluation protocol;
- multi-seed transfer evaluation, calibration, statistical summaries, and a
  machine-readable plus human-readable report;
- bounded energy-descent correction if the transfer gate passes;
- documentation of the Path A decision and the next architectural branch.

### Out of Scope Unless Path A Passes

- full Path B implementation or from-scratch pretraining;
- A100-scale training, broad hyperparameter sweeps, or model release;
- production efh-core integration;
- claims of soundness, universal calibration, or domain-general metacognition.

## Decision Gates

### Gate A — Experimental Integrity

The pipeline is reproducible, leakage checks pass, shuffled labels return chance
performance, and all predeclared baselines and domains are represented.

### Gate B — Energy Transfer

Advance only if the energy signal shows a replicated, practically meaningful
out-of-domain discrimination improvement over the token-probe and confidence
baselines without unacceptable task regression. Exact thresholds will be fixed
in the requirements before experiments run.

### Gate C — Correction

Advance toward Path B only if energy-descent correction improves correctness on
held-out natural errors at a matched compute budget and does not materially harm
already-correct predictions.

## Key Decisions

| Decision | Rationale | Status |
|---|---|---|
| Begin with Path A | Reuses pretrained competence and directly tests the core transfer hypothesis cheaply | Accepted |
| Treat Path A as a decision program, not a demo | The repository already has scaffolding; the missing value is credible evidence | Accepted |
| Compare multiple energy constructions | A single learned head can exploit corruption or domain shortcuts | Accepted |
| Use cached Qwen3-0.6B as the Path A base | Avoids a weight download, fits local hardware, and exposes both thinking and non-thinking regimes | Accepted |
| Gate Path B on transfer and correction evidence | Prevents a large training bet on an unvalidated mechanism | Accepted |
| Optimize for local reproducibility first | Matches available RTX 3060 hardware and keeps failed experiments affordable | Accepted |

## Existing References

- `README.md` — project thesis and status
- `docs/ARCHITECTURE.md` — architectural design and two build paths
- `docs/TRAINING.md` — empirical training rules and evaluation discipline
- `docs/ROADMAP.md` — validated findings, hypotheses, ladder, and off-ramps
