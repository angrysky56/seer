import numpy as np
import pytest

from seer.eval import (
    adaptive_ece,
    auroc,
    full_eval_report,
    mean_token_logprob_baseline,
    multi_seed_eval,
    self_consistency_spread_baseline,
    shuffle_label_control,
    verbalized_confidence_baseline,
)


def test_auroc_perfect_separation() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    assert auroc(labels, scores) == pytest.approx(1.0)


def test_auroc_worst_case_separation() -> None:
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.9, 0.8, 0.95, 0.1, 0.2, 0.3]
    assert auroc(labels, scores) == pytest.approx(0.0)


def test_auroc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        auroc([1, 1, 1], [0.1, 0.2, 0.3])


def test_adaptive_ece_zero_for_perfectly_calibrated() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    probs = rng.uniform(0.01, 0.99, size=n)
    labels = (rng.uniform(size=n) < probs).astype(float)
    ece = adaptive_ece(labels, probs, n_bins=10)
    assert ece < 0.05  # well-calibrated by construction; small sampling slack


def test_adaptive_ece_high_for_overconfident_wrong() -> None:
    n = 100
    probs = np.full(n, 0.99)
    labels = np.zeros(n)  # always wrong despite 0.99 confidence
    ece = adaptive_ece(labels, probs, n_bins=10)
    assert ece > 0.9


def test_shuffle_label_control_is_near_chance() -> None:
    rng = np.random.default_rng(0)
    n = 500
    labels = (rng.uniform(size=n) < 0.5).astype(float)
    # scores strongly correlated with the REAL labels
    scores = labels + rng.normal(scale=0.1, size=n)

    shuffled_aurocs = [shuffle_label_control(labels, scores, seed=s) for s in range(20)]
    mean_shuffled = float(np.mean(shuffled_aurocs))
    assert abs(mean_shuffled - 0.5) < 0.05


def test_multi_seed_eval_returns_mean_and_std() -> None:
    mean, std = multi_seed_eval(lambda seed: float(seed), seeds=[0, 1, 2])
    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(1.0)


def test_multi_seed_eval_single_seed_std_is_nan() -> None:
    mean, std = multi_seed_eval(lambda seed: 0.5, seeds=[0])
    assert mean == pytest.approx(0.5)
    assert np.isnan(std)


def test_baselines_are_thin_auroc_wrappers() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    assert verbalized_confidence_baseline(labels, scores) == auroc(labels, scores)
    assert mean_token_logprob_baseline(labels, scores) == auroc(labels, scores)
    assert self_consistency_spread_baseline(labels, scores) == auroc(labels, scores)


def test_full_eval_report_keys_and_ranges() -> None:
    rng = np.random.default_rng(0)
    n = 300
    labels = (rng.uniform(size=n) < 0.5).astype(np.float64)
    scores = labels + rng.normal(scale=0.2, size=n)
    probs = 1 / (1 + np.exp(-scores))

    report = full_eval_report(labels, scores, probs, seeds=[0, 1, 2])

    assert set(report.keys()) == {"auroc_mean", "auroc_std", "ece", "shuffle_auroc"}
    assert 0.0 <= report["auroc_mean"] <= 1.0
    assert 0.0 <= report["ece"] <= 1.0
    assert 0.0 <= report["shuffle_auroc"] <= 1.0
