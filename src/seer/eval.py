"""Eval harness (TRAINING.md section 7): the checks that killed a false result before.

Every metric/control below exists because skipping it once already produced a
wrong headline number in the predecessor work:

- AUROC is primary; report it alongside adaptive/equal-mass-bin ECE (equal-width
  bins manufacture artifacts).
- A shuffle-label control MUST be run alongside any real result — if AUROC on
  permuted labels doesn't fall to ~0.5, the pipeline leaks and the real result
  is void.
- Multi-seed: a single-seed effect on a small, high-variance model is not to be
  believed — report mean +/- sd, never one number.
- Baselines the probe/energy signal must beat: verbalized confidence, mean
  token logprob, self-consistency spread.

Nothing here requires a trained SEER model — every function takes plain arrays
of scores/labels so it can be exercised on synthetic data in tests, and reused
unchanged once real model outputs exist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


def auroc(labels: ArrayLike, scores: ArrayLike) -> float:
    """AUROC of ``scores`` predicting binary ``labels`` (primary metric, TRAINING.md section 7).

    Args:
        labels: Binary ground truth (e.g. ``argmax == truth``), shape ``(n,)``.
        scores: Higher = more confident/plausible, shape ``(n,)``.

    Returns:
        AUROC in [0, 1]. 0.5 = chance.
    """
    labels_arr = np.asarray(labels)
    if len(np.unique(labels_arr)) < 2:
        raise ValueError("AUROC is undefined with only one class present in labels")
    return float(roc_auc_score(labels_arr, np.asarray(scores)))


def adaptive_ece(labels: ArrayLike, probs: ArrayLike, n_bins: int = 10) -> float:
    """Expected Calibration Error with equal-MASS (adaptive) bins, not equal-width.

    Equal-width bins manufacture artifacts when scores cluster (TRAINING.md
    section 7); equal-mass bins avoid empty/overcrowded bins by construction.

    Args:
        labels: Binary ground truth, shape ``(n,)``.
        probs: Predicted p(correct) in [0, 1], shape ``(n,)``.
        n_bins: Number of equal-mass bins.

    Returns:
        Weighted mean absolute gap between bin-mean confidence and bin-mean
        accuracy.
    """
    labels_arr = np.asarray(labels, dtype=np.float64)
    probs_arr = np.asarray(probs, dtype=np.float64)
    n = len(probs_arr)
    if n == 0:
        raise ValueError("adaptive_ece requires at least one example")
    n_bins = min(n_bins, n)

    order = np.argsort(probs_arr)
    sorted_probs = probs_arr[order]
    sorted_labels = labels_arr[order]

    bin_edges = np.array_split(np.arange(n), n_bins)
    ece = 0.0
    for idx in bin_edges:
        if len(idx) == 0:
            continue
        bin_conf = sorted_probs[idx].mean()
        bin_acc = sorted_labels[idx].mean()
        ece += (len(idx) / n) * abs(bin_conf - bin_acc)
    return float(ece)


def shuffle_label_control(
    labels: ArrayLike, scores: ArrayLike, seed: int = 0
) -> float:
    """AUROC of ``scores`` against a random permutation of ``labels``.

    Must return ~0.5. If it doesn't, the eval pipeline leaks information from
    labels into scores somewhere upstream and any accompanying real result is
    void (TRAINING.md section 7).
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.asarray(labels))
    return auroc(shuffled, scores)


def multi_seed_eval(
    fn: Callable[[int], float], seeds: Sequence[int]
) -> tuple[float, float]:
    """Run ``fn(seed)`` across ``seeds`` and report (mean, sample_std).

    A single-seed number is not a result (TRAINING.md section 7) — every
    reported metric in an experiment writeup should come from this, not a bare
    call to e.g. ``auroc``.

    Args:
        fn: Any evaluation callable that is a function of a seed, e.g.
            ``lambda s: auroc(*eval_with_seed(s))``.
        seeds: At least 2 seeds; TRAINING.md examples use 3.

    Returns:
        ``(mean, sample_std)``. ``sample_std`` uses ddof=1; NaN if only one seed.
    """
    if len(seeds) < 1:
        raise ValueError("multi_seed_eval requires at least one seed")
    values = np.array([fn(s) for s in seeds], dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    return mean, std


# --- Baselines the energy/probe signal must beat (TRAINING.md section 7) ----


def verbalized_confidence_baseline(labels: ArrayLike, verbalized_confidence: ArrayLike) -> float:
    """AUROC of the model's spoken (role-play) self-report confidence.

    Expected near-chance per the predecessor results (Goodfire reproduction:
    0.54) — distinguish this from *answer* confidence (softmax prob), which is
    a separate, stronger baseline below.
    """
    return auroc(labels, verbalized_confidence)


def mean_token_logprob_baseline(labels: ArrayLike, token_logprobs: ArrayLike) -> float:
    """AUROC of per-example mean token log-probability (the answer-confidence baseline).

    Args:
        labels: Binary correctness, shape ``(n,)``.
        token_logprobs: Mean log-probability of the generated answer tokens per
            example, shape ``(n,)`` (already reduced over the token dimension by
            the caller).
    """
    return auroc(labels, token_logprobs)


def self_consistency_spread_baseline(
    labels: ArrayLike, agreement_fractions: ArrayLike
) -> float:
    """AUROC of self-consistency agreement (fraction of sampled answers matching the mode).

    Args:
        labels: Binary correctness, shape ``(n,)``.
        agreement_fractions: For each example, the fraction of K sampled
            generations that agreed with the modal answer, shape ``(n,)``.
            Higher agreement = higher confidence, so this is used directly as
            the AUROC score (no sign flip needed).
    """
    return auroc(labels, agreement_fractions)


def full_eval_report(
    labels: NDArray[np.float64],
    scores: NDArray[np.float64],
    probs: NDArray[np.float64],
    seeds: Sequence[int],
    n_bins: int = 10,
) -> dict[str, float]:
    """Convenience bundle: AUROC (multi-seed via bootstrap resample per seed) + ECE + shuffle control.

    Note this resamples with replacement per seed to get a spread even from a
    single fixed ``(labels, scores)`` set — when real per-seed *training* runs
    exist, prefer calling ``multi_seed_eval`` directly with a seed-conditioned
    training+eval closure instead of bootstrap resampling of one run's outputs.

    Returns:
        Dict with ``auroc_mean``, ``auroc_std``, ``ece``, ``shuffle_auroc``.
    """
    n = len(labels)

    def _bootstrap_auroc(seed: int) -> float:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=n)
        return auroc(labels[idx], scores[idx])

    auroc_mean, auroc_std = multi_seed_eval(_bootstrap_auroc, seeds)
    return {
        "auroc_mean": auroc_mean,
        "auroc_std": auroc_std,
        "ece": adaptive_ece(labels, probs, n_bins=n_bins),
        "shuffle_auroc": shuffle_label_control(labels, scores),
    }
