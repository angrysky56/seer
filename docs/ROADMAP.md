# SEER — Roadmap: validated vs hypothesis

Discipline: state plainly what is measured vs assumed. The predecessor work killed
one exciting-but-false result (the "SPS Detectability Paradox"); keep that habit.

## Validated (from sps-blindspot / probe-confidence)

- Muon ≫ AdamW at matched config on the local task (0.989 vs 0.798 @ 1k steps).
- Dense per-step supervision is required to learn state-tracking (sparse plateaus).
- SPS is a state-fidelity amplifier, orthogonal to the blind spot (H1 ~6× state
  propagation; H2 no behavioral self-correction). Geometric "detectability" story
  did NOT survive faithful lens + multi-seed — treat as dead.
- The internal-state self-signal is real (AUROC 0.96 on confident errors),
  architecture-independent (base = SPS), works as a native head (0.955–0.968), and
  drives selective prediction (99.7% accuracy at 80% coverage).
- A confidence head must be JOINTLY trained with a task anchor (confidence-only
  forgets: 0.931 → 0.059).
- The token self-probe is DOMAIN-SPECIFIC (L1 transfer 0.79 → 0.50).
- Probe-confidence L0: activation probe 0.79 vs verbalized 0.54 (Goodfire result
  reproduced locally with verifier-free labels).

## Hypotheses (unproven — the reason SEER exists)

- **H-energy-transfer (primary).** The energy / representation-plausibility signal
  transfers across domains where the token probe (0.50) did not. THE first thing
  to test. If false, SEER's core advantage over "modify Qwen + domain probes" is
  gone and Path A with per-domain heads is the honest fallback.
- **H-blindspot-relocation.** Reasoning in concept space (not autoregressing
  tokens) reduces or relocates the blind spot. Test: inject a corrupted concept,
  measure downstream propagation vs a token model.
- **H-energy-descent.** A few steps of energy descent on a high-energy predicted
  concept improve correctness before decoding. Test on cases the model gets wrong.

## Experiment ladder (in order; each gates the next)

1. **Energy transfer (Path A, local).** Add an energy/plausibility head to a
   modified small model; run the probe-confidence L0→L1→L2 ladder on the ENERGY
   signal. Accept only if energy AUROC survives transfer where the token probe
   collapsed. Metrics: AUROC (primary), adaptive-ECE, shuffle control, multi-seed.
2. **Energy-descent correction.** On the same model, apply bounded energy descent
   to high-energy predictions; measure correction rate on held-out errors.
3. **Blind-spot relocation.** Concept-corruption propagation, SEER vs token
   baseline.
4. **Gate integration.** Wire the validated energy channel to efh-core's commit
   gate as `confidence_source: energy:<id>`, domain-gated + per-domain temperature.
5. **Path B (only if 1 succeeds).** From-scratch LANG-JEPA-style latent-predictive
   model with a native energy objective; scale-up on A100 under muP.

## Explicit off-ramps

- If step 1 fails (energy doesn't transfer), SEER-as-a-distinct-architecture is not
  justified; fall back to "modified Qwen/Gemma + domain-matched probes + efh-core
  gate", which IS validated. Document and stop — do not chase a beautiful ghost.
