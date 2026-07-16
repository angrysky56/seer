# Research Summary — Path A Experimental Design

## Purpose

This research pass checked whether adjacent energy-based and joint-embedding
literature changes SEER's immediate direction. It does not validate SEER's core
claim; it identifies controls and variants that the Path A experiment must test.

## Findings

### 1. Energy is a credible baseline for OOD scoring, not proof of correctness transfer

Energy scores can separate in-distribution and out-of-distribution inputs better
than maximum softmax probability, and energy-aware training can shape that
separation. SEER asks a harder and different question: whether energy of an
internal prediction transfers as a correctness signal across task domains.
Therefore input-OOD benchmarks are supporting motivation, not acceptance
evidence.

Source: Liu et al., *Energy-based Out-of-distribution Detection* (2020),
https://arxiv.org/abs/2010.03759

### 2. Representation geometry should be evaluated explicitly

Work combining energy with semantic representation structure suggests that a
scalar score alone may discard useful geometry. Path A should compare the learned
scalar head with predictor disagreement and a representation-distance/density
baseline. The comparison should use identical trunk states and data splits.

Source: Joshi et al., *Semantic Driven Energy based Out-of-Distribution
Detection* (2022), https://arxiv.org/abs/2208.10787

### 3. Energy surfaces can trade off task behavior, calibration, and robustness

Joint energy models require careful optimization and can exhibit gaps relative
to ordinary discriminative models. This reinforces SEER's existing joint task
anchor and adds a required task-regression check to every energy experiment.
Sharpness or smoothness interventions belong in later ablations, not the minimal
first run.

Source: Yang et al., *Towards Bridging the Performance Gaps of Joint
Energy-based Models* (2022), https://arxiv.org/abs/2209.07959

### 4. Collapse prevention and effective-rank diagnostics remain load-bearing

Non-contrastive joint-embedding systems need architectural asymmetry or explicit
decorrelation constraints. Although Path A reads a pretrained representation,
any jointly trained concept projection can still lose dimensional utility.
Effective rank, per-dimension variance, and pairwise similarity should be logged;
VICReg-style regularization is a contingency triggered by diagnostics rather
than enabled by default.

Source: Liu et al., *Bridging the Gap from Asymmetry Tricks to Decorrelation
Principles in Non-contrastive Self-supervised Learning* (NeurIPS 2022),
https://openreview.net/forum?id=Jz98aDK5gMW

## Design Consequences

- Predeclare transfer success thresholds before looking at final test results.
- Use natural model errors as the primary correctness target; constructed
  corruptions are training aids and controlled ablations.
- Compare scalar learned energy, predictor disagreement, representation geometry,
  and token confidence under one evaluator.
- Cross corruption type and evaluation domain to expose shortcut learning.
- Log task quality and representation-collapse diagnostics beside AUROC/ECE.
- Keep Path B and expensive scale-up conditional on replicated Path A evidence.

## Local Model Inventory

The primary Path A model is `Qwen/Qwen3-0.6B`, already present in the local
Hugging Face cache at snapshot
`c1899de289a04d12100db370d81485cdf75e47ca`. Its cached configuration declares
`Qwen3ForCausalLM`, 28 layers, hidden size 1024, bf16 weights, and a 40,960
maximum-position setting. The model card identifies 0.6B total parameters and an
Apache-2.0 license. It also requires Transformers 4.51 or newer, so the current
project lower bound of 4.46 must be raised during implementation.

Qwen3 supports thinking and non-thinking generation. The primary transfer test
will use non-thinking mode for bounded compute and comparability; thinking mode
will be a separately reported secondary regime because it changes both internal
trajectory length and answer-generation behavior.

## Open Decisions for Requirements

- exact domains and datasets for the L0/L1/L2 ladder;
- base checkpoint(s) that fit local hardware and licensing constraints;
- minimum practically meaningful OOD AUROC delta and confidence interval rule;
- task-regression tolerance and correction safety threshold;
- compute budget for tuning versus held-out confirmation.
