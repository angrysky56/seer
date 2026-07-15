# SEER — Training & Model-Modification Dev Docs

Concrete, battle-tested recipes. Every rule here was learned empirically in the
predecessor experiments (`sps-blindspot`, `probe-confidence`) or is standard JEPA
practice from `lang-jepa`. Where a rule cost a wasted run, that is noted — those
are the expensive ones to re-learn.

Hardware baseline: single RTX 3060 (12 GB), 64 GB RAM. All recipes below are
sized to run locally; the A100/T4 notes are for scale-up only.

---

## 0. Environment

- `uv` for Python envs; `torch` + `transformers` + `numpy` + `scikit-learn`.
- Node's npm here defaults to `omit=dev` — pass `--include=dev` if you add a TS
  component. (Only relevant if wiring to efh-core.)
- Long training runs launched from a shell tool get **killed when that shell's
  session ends** (both backgrounded and foreground). Drive multi-arm sweeps from a
  single Python script that loops internally and writes a durable results file, or
  keep each run short (≤ ~2 min) so it finishes within one session.

## 1. Optimizer: Muon + AdamW split (grounded, ~4× faster than AdamW here)

Measured: on the entity-tracking task, Muon reached 0.989 answer-accuracy by
step 1000 vs AdamW's 0.798 at matched (default) LRs. Use Muon.

- **Muon on hidden weight matrices only** (attention Q/K/V/O, MLP projections).
- **AdamW on everything else**, routed BY ROLE not shape: token embeddings, the
  LM/decoder head, LayerNorm gains, biases, and — the easy miss — any learnable
  **placeholder/`<predict>`/energy-head vector** that is 2-D-ish but is not doing
  `y = x @ W`. KellerJordan/Muon ships this split; reuse it.
- Muon's LR is in different units (spectral-norm-per-update) — sweep it on its own
  grid, do not share AdamW's.
- Combine with **muP** so a small-proxy LR sweep transfers to scale-up without
  retuning. Ground the base LR with a real cheap grid (2 orders of magnitude, add
  one point past the apparent best), never a single guess.
- When comparing SEER-vs-baseline AND Muon-vs-AdamW, freeze the optimizer first,
  then change only the architecture — never both at once.

## 2. Precision

- **Train in bf16** (mixed precision). fp16 is fine for the forward but bf16 is
  safer for gradients.
- The forward-mode **JVP / spectral measurements need fp32** — fp16 underflows the
  double-backward to exactly 0, and bf16's 7-bit mantissa swamps a ~0.01 signal.
  Do measurement in fp32; it runs at inference and costs little.
- Attention: use **eager** attention for any JVP/double-backward code — the fused
  SDPA kernels lack the second-order backward.

## 3. Supervision density (the task-design rule that cost a run)

A single terminal target does not train state-tracking: full-LM loss spends
~29/30 of its gradient on unpredictable content and the target mapping gets almost
none — accuracy plateaus. Use **dense, per-step, state-dependent supervision**
(emit the tracked quantity after every step; loss at those positions). This is
also more JEPA-faithful (predict-a-thing-that-needs-long-range-state at every
position). For a from-scratch reasoning task, prefer dense over sparse targets.

## 4. The self-certainty head (the rule that cost the worst run)

Training a confidence/energy head **only** on its own objective — with gradient
flowing into the model — **catastrophically forgets the task** (measured: task
accuracy 0.931 → 0.059). Always use a **JOINT objective**: task loss (anchored on
clean data) + self-certainty loss (on the target distribution), summed. With the
anchor, baking the head in preserves competence (0.931 → 0.924) and slightly
improves self-legibility (AUROC 0.955 → 0.968).

- **Head-only (frozen state)** already gives ~0.955 AUROC — the frozen state
  carries the signal; baking in mainly buys *native emission*, not much accuracy.
- Self-certainty targets are the model's own correctness (`argmax == truth`),
  **stop-gradient on the target**.
- Train the self-certainty head on the regime where it MATTERS: the confident-error
  (blind-spot) distribution, not just clean data. Output confidence is already
  fine on ordinary errors; the head's value is where the model is *confidently
  wrong* (measured: energy/probe AUROC 0.96 there, output confidence 0.55–0.65).

## 5. Domain matching (the transfer rule)

The token-space self-probe is **domain-specific**: trained on arithmetic it read
correctness at 0.79 in-domain but transferred to other formal theories at ~0.50
(chance). Consequences for SEER:

- A single self-certainty head is **not** a universal metacognition organ. Train
  it on the deployment distribution, or run multiple domain heads with routing.
- The whole point of the **energy channel** (Path B / the EBM head) is to beat this
  — representation plausibility should transfer where token-surface features did
  not. **This is unproven; it is the first thing to test** (see the L1 ladder in
  ROADMAP). Do not assume energy transfers; measure it.

## 6. JEPA training (Path B), from `lang-jepa`

- **EMA target encoder** (momentum 0.996→1.0). Same-encoder-both-paths collapses.
- **Masked-mean pool + L2-normalize** targets; predictions and targets same dim;
  **no target-side projection head** (collapse accelerant).
- **Smooth-L1** on normalized features (I-JEPA default), kinder than MSE.
- Watch collapse diagnostics: target effective-rank must stay >> 1, std > ~0.05,
  cosine-sim rising but NOT pinned at 1.0 in the first hundreds of steps. If it
  collapses at scale, add **VICReg** variance/covariance regularization.
- Energy head: train contrastively (in-distribution concepts low energy; corrupted
  / shuffled / off-manifold concepts high energy) or by score matching. Keep it
  OFF the target-encoder path.

## 7. Evaluation you must not skip (the rules that killed a false result)

- **Calibration != discrimination.** Report **AUROC** (discrimination — the
  hard-to-fix, transfer-critical part) and ECE with **adaptive/equal-mass bins**
  (equal-width manufactures artifacts). Temperature-scale ECE per domain.
- **Shuffle-label control** at every eval: train the head on permuted labels;
  AUROC must fall to ~0.5, else the pipeline leaks and results are void.
- **Multi-seed.** A single-seed effect in a small, high-variance model is not to be
  believed — a headline result here reversed sign across 3 seeds. Report mean ± sd.
- **Faithful instruments.** Logit-Lens-style mid-layer reads carry affine drift and
  invent signals; a forced-attention (Δ=0) read on this model is cheap (run the
  hidden through the remaining blocks as a length-1 sequence — self-attention is
  identity, RoPE cancels). Use the faithful read before any strong claim.
- **Baselines the probe must beat**, every rung: verbalized confidence, mean token
  logprob, self-consistency spread. Distinguish *verbalized* confidence (spoken
  self-report — role-play, near-chance) from *answer* confidence (softmax prob —
  fine in-domain for a calibrated model).

## 8. Compute allocation

- Cheap sweeps (LR, optimizer ablation, seeds) → local 3060 + T4 pool in parallel.
  T4 ≈ 3060 per hour; treat it as more lanes, not an accelerator.
- Reserve limited A100 hours for the ONE thing they answer: a scale-up run to test
  whether a small-scale result survives past toy scale (muP makes the transfer
  valid). Don't burn A100 on sweeps.
- Downloads on a flaky link: `hf_transfer` + a subprocess-timeout retry loop
  (plain downloader HANGS on a dead socket; hf_transfer raises, so a timeout kill +
  resume ratchets forward). Don't mix downloaders on one file — incompatible
  `.incomplete` formats reset progress to 0.

## 9. Wiring the self-certainty channel to efh-core

The energy/probe output becomes `confidence_score` (+ `confidence_source:
energy:<id>` in the audit trail) at efh-core's commit gate. Gate rule:
`commit ⇔ proof_confidence ≥ 0.7 ∧ confidence_score ≥ 0.7 ∧ closure = KERNEL1`.
The energy channel is trusted **only in domains the transfer ladder validated**,
with a fitted per-domain temperature; elsewhere the gate falls back to the
symbolic verifier's "unknown = not a pass". The energy channel augments; it never
replaces the verifier.
