# SEER — Self-Evaluating Energy Reasoner

A design + development project for a language model that **knows when it doesn't
know** — and can act on it. SEER predicts in representation ("concept") space,
scores its own predictions by an **energy / plausibility** signal, and
self-corrects by moving toward lower energy — rather than emitting a confident
token and being blind to its own error.

This is the synthesis of a run of experiments (see `docs/ROADMAP.md` for what is
*validated* vs *hypothesis*) and several 2024–2026 papers. It is architecture-
paper-first: the deliverable is the design and the training recipes, not yet a
trained model.

## The one-paragraph idea

Autoregressive LLMs are fluent, fast, confident — and structurally unable to
detect their own errors from inside the trajectory (the *self-correction blind
spot*). A trained model's *internal state* nonetheless carries a strong
correctness signal that its *output* has lost: in our experiments an activation
probe predicted a model's own confident errors at AUROC 0.96 where its stated
confidence was near-chance. SEER makes that self-signal **native and structural**
instead of a bolted-on probe, by borrowing JEPA's move of predicting in
representation space: the distance between a predicted concept and a *plausible*
concept is an intrinsic uncertainty measure (energy), and correction is energy
descent. External symbolic verification (efh-core's commit gate) remains the
backstop for formalizable claims; SEER's energy channel is the fast, in-model
metacognition that decides when to invoke it.

## Three pillars (and why none is optional)

1. **Transport** — clean state propagation. State-Prediction Separation keeps the
   prediction workspace from cluttering the persistent state (an efficiency win),
   but it is *orthogonal to self-correction*: it amplifies whatever enters the
   state, error included. So it earns the need for pillars 2–3, it does not
   replace them.
2. **Inspection** — the energy / self-certainty channel. Read the model's own
   prediction plausibility. Native (JEPA energy) rather than a token-space probe,
   with the explicit goal of *domain-general* transfer the token probe lacked.
3. **Admission** — the verification gate. A belief becomes authoritative state
   only if it passes: cheap in-model energy triage first, external symbolic
   verification (efh-core, Z3/Isabelle) for formalizable claims, and refusal /
   abstention otherwise.

## Two build paths (documented; not yet chosen)

- **Path A — augment a capable model.** Add a JEPA/energy-style
  representation-plausibility head to a modified Qwen/Gemma. Leverages pretrained
  weights; validated self-probe recipe (`docs/TRAINING.md`). Pragmatic.
- **Path B — from-scratch latent-predictive model.** A LANG-JEPA-style
  concept-space predictor with a native energy head and energy-descent
  correction. Cleaner native signal; restarts validation. Research.

The first experiment either way is the transfer test: **does the energy signal
beat the token probe out-of-domain?** (the L1 rung that broke the token probe).

## Status

Design + dev docs. See:
- `docs/ARCHITECTURE.md` — the model, both paths, the energy self-certainty channel.
- `docs/TRAINING.md` — how to train / modify: the hard-won recipes.
- `docs/ROADMAP.md` — validated vs hypothesis, the experiment ladder.

## Papers this builds on

**State / prediction separation & efficiency**
- Monea, Godey, Brantley, Artzi. *The State-Prediction Separation Hypothesis.* arXiv:2607.01218 (2026).
- Yang, Sun, Xia. *Depth Exploration for LLM Decoding.* arXiv:2606.29223 (2026).

**The blind spot & self-correction**
- Petrova, Vejsiu. *Spectral Origins of the Self-Correction Blind Spot in Autoregressive Generation.* arXiv:2607.09803 (2026).

**Internal-state / self-knowledge probing**
- Sarfati et al. (Goodfire / Eternis). *What LLM Forecasters Know but Don't Say.* arXiv:2607.08046 (2026).
- Anthropic. *A global workspace in language models* (J-space). transformer-circuits.pub/2026/workspace (2026).

**Representation geometry**
- Ma, Wolfinger. *Laguerre Geometry for Interpreting Large Language Models.* arXiv:2607.10578 (2026).
- Chae. *Infrared Organization and Critical Cognitive Field Formation in Transformer Dynamics.* arXiv:2607.10923 (2026).

**Joint-embedding predictive architectures (JEPA / EBM)**
- LeCun. *A Path Towards Autonomous Machine Intelligence.* (2022).
- Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA).* CVPR (2023).
- Bardes et al. *V-JEPA: Video Joint-Embedding Predictive Architecture.* (2024).
- Bardes, Ponce, LeCun. *VICReg: Variance-Invariance-Covariance Regularization.* ICLR (2022).
- `lang-jepa` (internal): next-sentence concept prediction, EMA target, decoder.

**Latent reasoning theory**
- *A First-Principles Theory of Slow Thinking and Active Perception.* (alphaXiv, 2026).

**Optimization**
- Jordan et al. *Muon.* (2024); Liu et al. *Muon is Scalable for LLM Training.* arXiv:2502.16982 (2025).
- Yang & Hu et al. *Tensor Programs V (muP).* (2022).

**Closure / world-model grounding**
- Rosas et al. *Software in the natural world.* arXiv:2402.09090 (2024).

## License

Research. Frameworks and results attributed to their authors.
