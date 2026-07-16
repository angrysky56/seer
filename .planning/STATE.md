# Project State

## Project

- **Name:** SEER — Self-Evaluating Energy Reasoner
- **Milestone:** Path A Evidence Program
- **Initialized:** 2026-07-15
- **Workflow mode:** Interactive, standard depth, research and verification enabled

## Current Position

- **Phase:** 2 of 7 — Multi-Domain Evidence Data
- **Status:** Ready for planning
- **Progress:** 10/47 milestone requirements complete
- **Completed phase:** Phase 1 — 4/4 plans, verification passed
- **Next command:** `/gsd-plan-phase 2`

## Milestone Goal

Determine whether representation-space energy predicts correctness across held-out
reasoning domains better than token/probe confidence signals, then conditionally
test whether it can safely guide concept-space correction.

## Established Context

- Existing Path A model, data, optimizer, training, and evaluation skeleton is in
  `src/seer` with offline fake-model tests in `tests`.
- Primary cached model: `Qwen/Qwen3-0.6B`, snapshot
  `c1899de289a04d12100db370d81485cdf75e47ca`.
- Primary generation regime: Qwen3 non-thinking mode; thinking mode is secondary
  and reported separately.
- Primary transfer: train on bAbI; confirm without signal updates on ProofWriter
  and GSM8K.
- Gate B requires replicated OOD AUROC and improvement over the strongest matched
  baseline; Path B remains conditional.

## Decisions

| Date | Decision | Rationale |
|---|---|---|
| 2026-07-15 | Initialize around the existing SEER direction | User confirmed the repository vision is current |
| 2026-07-15 | Make Path A decision-grade | Existing scaffolding makes credible empirical evidence the next valuable outcome |
| 2026-07-15 | Use cached Qwen3-0.6B | Avoids downloading weights and fits the local RTX 3060 |
| 2026-07-15 | Separate thinking and non-thinking regimes | Prevents trajectory length and sampling behavior from confounding transfer |
| 2026-07-15 | Gate Path B on transfer and correction | Avoids an expensive architecture bet without local evidence |

## Open Risks

- Qwen3-0.6B may yield too few correct or incorrect examples on a domain for
  stable AUROC; Phase 2 must measure class sufficiency before fitting signals.
- Constructed negatives may create shortcuts; natural errors remain the primary
  test and corruption strategy is crossed with domain.
- A scalar energy head may add no information beyond a learned hidden-state probe.
- Concept refinement may lower energy without changing decoded answers; Gate C
  requires actual correctness changes.
- The current Transformers lower bound predates Qwen3 support and must be raised.

## Session Continuity

Phase 1 is complete and independently verified. The reproducible runtime, exact
offline Qwen3 boundary, deterministic smoke experiment, transactional artifacts,
and operations guide are ready for Phase 2's real-domain data work. Confirmatory
model runs remain assigned to later phases.

## Last Activity

- **Date:** 2026-07-15
- **Action:** Completed Phase 1 — Reproducible Experiment Runtime
- **Result:** 4/4 plans complete; 4/4 success criteria and 10/10 requirements verified
