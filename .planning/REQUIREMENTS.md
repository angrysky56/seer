# Requirements — SEER Path A Evidence Milestone

## Milestone Objective

Produce a reproducible, statistically defensible answer to the question: does a
representation-space energy signal predict model correctness across held-out
reasoning domains better than token/probe confidence signals? If it does, test
whether that signal can safely guide bounded concept-space correction.

## Fixed Experimental Defaults

- **Primary base checkpoint:** `Qwen/Qwen3-0.6B` (Apache-2.0), using the locally
  cached snapshot `c1899de289a04d12100db370d81485cdf75e47ca`. It has 28 layers,
  hidden size 1024, bf16 weights, and fits the local 12 GB GPU. The experiment
  loader must support offline/local-files-only resolution and record the resolved
  snapshot. The code must not hard-code Qwen internals; a fake model remains
  available for tests.
- **Generation regime:** non-thinking mode is the primary confirmatory condition
  because it gives a bounded, comparable token budget. Thinking mode is a
  predeclared secondary condition with its own fixed sampling and token budget;
  results from the two regimes are never pooled.
- **Primary domains:** GSM8K arithmetic, ProofWriter deductive logic, and bAbI
  state/entity tracking. These provide objectively scored answers and distinct
  task structures. Dataset revisions and licenses must be recorded in manifests.
- **Transfer protocol:** train/tune the correctness signal on one declared source
  domain and evaluate it without weight updates on each held-out domain. Rotate
  the source domain where compute permits; the predeclared primary direction is
  bAbI -> ProofWriter and GSM8K.
- **Seeds:** at least three training seeds for confirmatory results.
- **Primary metric:** AUROC for correctness discrimination. Adaptive ECE is a
  separately reported calibration metric, never a substitute for AUROC.

These are defaults, not hidden assumptions. A dataset/checkpoint can be replaced
only for a documented incompatibility, with the reason and replacement recorded
before inspecting confirmatory test results.

## Functional Requirements

### Experiment Foundation

- [x] **EXP-01** Provide a typed, serializable experiment configuration with all
  model, dataset, split, seed, energy, optimizer, calibration, and output settings.
- [x] **EXP-02** Provide a CLI that can prepare data, cache base-model outputs,
  train one arm, evaluate one arm, run the declared matrix, and build a report.
- [x] **EXP-03** Write a manifest for every run containing config, git revision,
  dependency versions, hardware, input dataset revisions, checkpoint identifier,
  seeds, timestamps, and artifact hashes.
- [x] **EXP-04** Support deterministic resumption and do not overwrite a completed
  run unless the user explicitly requests replacement.
- [x] **EXP-05** Keep tests offline and lightweight; external weights/data are
  accessed only by explicit experiment commands, never by the unit suite.
- [x] **EXP-06** Require a Transformers version with Qwen3 support (>= 4.51),
  verify the cached snapshot before a run, and fail with an actionable message
  rather than silently accessing the network when offline mode is requested.

### Data and Scoring

- [ ] **DATA-01** Define one normalized task-example/result interface shared by
  GSM8K, ProofWriter, bAbI, and synthetic fixtures.
- [ ] **DATA-02** Implement pinned adapters for GSM8K, ProofWriter, and bAbI that
  preserve official train/validation/test boundaries where available.
- [ ] **DATA-03** Normalize answers with domain-specific, tested rules and retain
  raw prompts, generations, normalized predictions, labels, and failure reasons.
- [ ] **DATA-04** Prevent prompt/example leakage across signal-training,
  calibration, model-selection, and confirmatory test partitions; detect duplicate
  normalized examples and report overlap.
- [ ] **DATA-05** Produce both naturally correct and naturally incorrect examples.
  If a domain/model pairing yields fewer than 100 examples of either class in a
  confirmatory split, mark its AUROC inconclusive rather than silently resampling.
- [ ] **DATA-06** Treat constructed corruptions as labeled training/ablation data,
  record corruption provenance, and never mix them into the primary natural-error
  test metric.
- [ ] **DATA-07** Record thinking-mode state, sampling parameters, and generated
  token count for every example; enforce the predeclared per-regime token budget.

### Model Instrumentation and Signals

- [ ] **SIG-01** Capture the declared commit-layer residual at a deterministic
  answer/commit position and project it to an L2-normalized concept vector.
- [ ] **SIG-02** Implement a learned scalar energy signal trained with at least
  two configurable negative strategies, including within-batch/shuffled concepts
  and semantically harder near-error negatives.
- [ ] **SIG-03** Implement predictor disagreement as an independently evaluable
  signal with deterministic sampling under a recorded seed.
- [ ] **SIG-04** Implement a representation-geometry baseline (for example,
  class-conditional distance or regularized density) fit only on training data.
- [ ] **SIG-05** Log effective rank, per-dimension variance, pairwise cosine
  similarity, and concept norms to detect complete or dimensional collapse.
- [ ] **SIG-06** When the base trunk is trainable, use the joint task anchor and
  measure task regression against the frozen/pretrained reference.

### Baselines and Calibration

- [ ] **BASE-01** Evaluate answer maximum probability/mean answer-token log
  probability under the same examples and splits.
- [ ] **BASE-02** Evaluate self-consistency spread with a fixed sampling budget.
- [ ] **BASE-03** Evaluate a regularized linear correctness probe on the declared
  hidden state as the direct predecessor-style learned baseline.
- [ ] **BASE-04** Support verbalized confidence as a secondary baseline where the
  prompt format makes it meaningful; missing/invalid responses are reported.
- [ ] **CAL-01** Fit monotone calibration using calibration data only, separately
  per declared domain, and report adaptive ECE plus reliability data.
- [ ] **CAL-02** Never use target-domain confirmatory labels to train, select, or
  calibrate the cross-domain signal reported as zero-shot transfer.

### Evaluation and Statistics

- [ ] **EVAL-01** Report per-domain/per-seed AUROC, adaptive ECE, accuracy,
  coverage-risk curves, class balance, and sample counts for every signal.
- [ ] **EVAL-02** Report mean, standard deviation, paired bootstrap 95% confidence
  intervals, and paired deltas versus the strongest eligible baseline.
- [ ] **EVAL-03** Run label-shuffle and feature/label alignment controls; shuffled
  AUROC must fall in `[0.45, 0.55]` or the affected result is invalid.
- [ ] **EVAL-04** Cross negative/corruption training strategy with held-out domain
  to expose shortcut-sensitive energy heads.
- [ ] **EVAL-05** Generate machine-readable JSON/JSONL results and a human-readable
  Markdown report from the same validated result records.
- [ ] **EVAL-06** Clearly label exploratory, tuning, confirmatory, failed, and
  inconclusive runs; only confirmatory runs determine the milestone decision.

### Transfer Decision Gate

- [ ] **GATE-01** Gate A passes only when manifests are complete, required arms
  exist, split/leakage checks pass, and shuffle controls are valid.
- [ ] **GATE-02** Gate B passes only if one predeclared energy variant, evaluated
  on natural errors, achieves all of the following:
  - mean OOD AUROC >= 0.65 on both primary held-out domains;
  - mean AUROC improvement >= 0.05 over the strongest eligible baseline across
    those domains;
  - paired-bootstrap 95% confidence interval for the aggregate improvement has a
    lower bound above 0;
  - no held-out domain has mean AUROC below 0.60;
  - task accuracy regression is <= 2 absolute percentage points for a trainable
    trunk, with no regression requirement for a frozen trunk.
- [ ] **GATE-03** Thresholds are frozen before confirmatory test labels are
  inspected. Failure yields the documented domain-matched fallback; thresholds
  must not be relaxed post hoc.

### Energy-Descent Correction (Conditional on Gate B)

- [ ] **CORR-01** Implement bounded, configurable gradient descent on concept
  vectors without mutating model weights, with norm/step clipping and rollback.
- [ ] **CORR-02** Decode or rescore refined concepts through a documented model
  interface; do not claim correction from reduced energy alone.
- [ ] **CORR-03** Compare energy-guided correction against equal-compute retry,
  self-consistency, and random-selection controls on held-out natural errors.
- [ ] **CORR-04** Gate C passes only if correction has a positive paired-bootstrap
  95% confidence lower bound for net accuracy change, corrects at least 10% of
  eligible wrong answers, and flips no more than 2% of eligible correct answers
  to wrong at the chosen operating point.

### Documentation and Decision Record

- [x] **DOC-01** Document setup, explicit download commands, local smoke runs,
  confirmatory runs, resume behavior, and artifact interpretation.
- [ ] **DOC-02** Generate a final decision record stating which gates passed,
  limitations, negative results, and the supported next branch.
- [ ] **DOC-03** If Gate B fails, specify the validated fallback of domain-matched
  heads plus admission gating and stop Path B work in this milestone.
- [ ] **DOC-04** If Gates B and C pass, produce a bounded Path B design brief and
  scale-check proposal; full Path B training remains a later milestone.

## Quality Requirements

- [x] **QUAL-01** `uv run pytest` passes without network access.
- [x] **QUAL-02** `uv run ruff check .` passes.
- [ ] **QUAL-03** Core normalization, splitting, metrics, controls, resume logic,
  and gate calculations have deterministic unit tests, including failure cases.
- [x] **QUAL-04** A synthetic end-to-end experiment completes on CPU in CI and
  emits the same artifact schema as a real run.
- [ ] **QUAL-05** Invalid or incomplete experimental evidence fails closed with a
  clear diagnostic rather than producing a pass decision.

## Traceability

Roadmap phases will map every requirement ID to exactly one owning phase. A
requirement may be exercised by later phases, but ownership remains singular so
completion and omissions are auditable.

## Deferred

- Full Path B latent-predictive architecture and from-scratch pretraining.
- Production efh-core integration and authoritative admission policies.
- Large-model or A100 scale-up before Gates B and C pass locally.
- Universal/domain-free calibration claims.
- Public checkpoint or dataset redistribution.
