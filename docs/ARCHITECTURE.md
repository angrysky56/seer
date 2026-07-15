# SEER — Architecture

SEER is defined by three functionally separate channels: a **state/transport**
channel, a **prediction** channel, and an **energy / self-certainty** channel
that scores the prediction's plausibility. Correction is energy descent in
representation space; admission to authoritative state is gated.

## 1. Why representation-space prediction (the JEPA move)

Autoregressive token models emit a token, feed it back as state, and cannot
detect an error already in that state (the blind spot; SPARC). Two things follow
from predicting **concepts** (representations) instead of tokens:

- The prediction has a **native uncertainty measure**: the energy /
  (implausibility) of the predicted representation. No token softmax (role-play),
  no external probe.
- Errors need not autoregress through the token channel during reasoning; the
  model reasons concept→concept and decodes to tokens only at the boundary. This
  *relocates* the blind spot rather than removing it — a hypothesis to test, not
  a guarantee.

JEPA discipline is load-bearing and inherited from `lang-jepa`: EMA target
encoder (stop-gradient asymmetry), masked-mean pooled + L2-normalized targets, no
target-side projection head (collapse accelerant), smooth-L1 on normalized
features, and VICReg variance/covariance regularization if collapse appears at
scale.

## 2. The energy / self-certainty channel (the novel part)

An **energy head** E(z) scores how plausible a concept z is under the model's
learned manifold. Two ways to realize it, in increasing ambition:

- **Predictor disagreement** (cheap, available at inference). Sample K predictor
  passes (dropout / small noise / an ensemble of predictor heads); the variance
  of predicted concepts is an uncertainty estimate. High variance = the model is
  unsure. No target needed.
- **Learned energy** (EBM). Train E(z) so in-distribution concepts have low
  energy and corrupted / off-manifold concepts have high energy (contrastive or
  score-matching). At inference, E(predicted_z) is the self-certainty. This is the
  channel with a shot at *domain-general* calibration, because it scores
  representation plausibility, not token surface — the explicit fix for the token
  probe's L1 failure.

Self-certainty p(correct) = a calibrated monotone map of −E(predicted_z) (fit per
the transfer ladder; expect a per-domain temperature, possibly per-domain
retraining — see ROADMAP). This is the in-model, fast metacognition.

## 3. Self-correction = energy descent

Given a predicted concept z0 with high energy (implausible / likely wrong), refine
it: z_{t+1} = z_t − η ∇E(z_t), a few steps of gradient descent on the energy in
concept space, then decode. This is genuine correction (revise toward
plausibility), not mere abstention. It is the operational form of "slow thinking":
spend more inference compute *only when energy is high*. Ties to the
active-perception paper's internal time axis and speed–accuracy tradeoff.

Abstention remains the floor: if energy stays high after refinement, decline /
route to the verifier rather than emit.

## 4. Admission gate (efh-core)

The energy channel is fast and in-model but not sound. For *formalizable* claims
the belief must clear the external gate before becoming authoritative state:
Z3 / Isabelle verification (efh-core), with "unknown = not a pass". The energy
channel decides *when to pay* for verification (high energy → verify), and the
gate decides *whether to admit*. Cheap triage + sound backstop.

## 5. Two substrates

### Path A — augment a capable autoregressive model
- Keep a pretrained Qwen/Gemma as the token model.
- Add a projection to a concept space + an energy head reading the residual
  state at commit positions.
- Self-certainty = energy of the current state's concept; correction = energy
  descent on the concept then re-decode (bounded steps).
- Pros: leverages pretrained competence; the validated self-probe recipe
  transfers directly (train the head jointly with a task anchor — see TRAINING).
- Cons: still an autoregressive token model underneath; energy is read off, not
  the native prediction objective.

### Path B — from-scratch latent-predictive model
- LANG-JEPA-style: encoder + predictor in concept space, EMA target, decoder for
  text. Add the energy head as a first-class objective.
- Self-certainty and correction are native (energy of the predictor's own output).
- Pros: cleanest native signal; the design's intended form.
- Cons: no pretrained competence; restarts the whole validation program; a large
  training bet.

Recommended sequence: **prove the energy-beats-token-probe transfer hypothesis on
Path A first** (cheap, local, reuses the L1 ladder). Only commit to Path B if the
energy signal demonstrably transfers where the token probe did not.

## 6. What the prior experiments settled (so we don't re-litigate)

- SPS is orthogonal to the blind spot (state-fidelity/efficiency only) — Transport
  earns pillars 2–3, doesn't replace them.
- The internal-state self-signal is real (AUROC 0.96 on confident errors) and
  architecture-independent (base = SPS), works as a native head (0.955–0.968),
  but is DOMAIN-SPECIFIC (token probe transfer collapsed to ~0.50). Domain-general
  transfer is exactly what the energy channel must earn.
- A confidence objective must be *jointly* trained with a task anchor;
  confidence-only fine-tuning catastrophically forgets the task.
