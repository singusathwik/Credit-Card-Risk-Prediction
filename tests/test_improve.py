import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from credit_risk.improve import (
    checked_folds,
    constrained_threshold,
    out_of_fold,
    paired_intervals,
    passes_guardrails,
)


class UnseenRowsOnly(ClassifierMixin, BaseEstimator):
    """Fail if an OOF prediction contains a training observation."""

    def fit(self, X, y):
        self.tokens_ = set(X["token"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        assert not self.tokens_.intersection(X["token"])
        return np.tile([0.6, 0.4], (len(X), 1))


def test_threshold_respects_recall_instead_of_maximizing_accuracy_alone():
    y = np.array([0] * 8 + [1] * 2)
    p = np.array([0.05] * 6 + [0.4, 0.6, 0.3, 0.7])
    threshold, rows = constrained_threshold(y, p, 1.0)
    assert ((p >= threshold)[y == 1]).mean() == 1
    assert ((p >= threshold) == y).mean() < max(r["accuracy"] for r in rows)


def test_threshold_ties_prefer_recall():
    y = np.array([0, 1])
    p = np.array([0.6, 0.4])
    threshold, _ = constrained_threshold(y, p, 0.5, grid=[0, 0.4, 0.5, 1])
    assert threshold == 0.4


@pytest.mark.parametrize(
    "y,p,floor",
    [
        ([0, 0], [0.1, 0.2], 0.5),
        ([0, 1], [0.2, np.nan], 0.5),
        ([0, 1], [0.2, 1.1], 0.5),
        ([0, 1], [0.2, 0.3], 0),
        ([0, 1], [0.2], 0.5),
    ],
)
def test_rejects_invalid_threshold_inputs(y, p, floor):
    with pytest.raises(ValueError):
        constrained_threshold(y, p, floor)


def test_impossible_custom_grid_is_rejected():
    with pytest.raises(ValueError, match="No threshold"):
        constrained_threshold([0, 1], [0.2, 0.3], 1, grid=[0.9, 1])


def test_grouped_oof_never_predicts_fit_rows():
    X = pd.DataFrame({"token": np.arange(60)})
    groups = np.repeat(np.arange(30), 2)
    y = np.tile([0, 0, 1, 1], 15)
    folds = checked_folds(X, y, groups, 42)
    for fit, score in folds:
        assert not set(groups[fit]) & set(groups[score])
    p = out_of_fold(UnseenRowsOnly(), X, y, folds)
    np.testing.assert_allclose(p, 0.4)


def test_promotion_rejects_accuracy_gain_with_excessive_recall_loss():
    base = {"accuracy": 0.79, "recall": 0.59, "brier": 0.13}
    assert not passes_guardrails(
        base, {"accuracy": 0.83, "recall": 0.36, "brier": 0.13}, 0.56, 0.03
    )
    assert not passes_guardrails(
        base, {"accuracy": 0.80, "recall": 0.57, "brier": 0.15}, 0.56, 0.03
    )
    assert passes_guardrails(
        base, {"accuracy": 0.80, "recall": 0.58, "brier": 0.13}, 0.56, 0.03
    )


def test_paired_intervals_equal_models_are_zero():
    y = np.tile([0, 1], 30)
    pred = np.tile([0, 0], 30)
    result = paired_intervals(y, pred, pred, np.repeat(np.arange(30), 2), repeats=20)
    assert result == {"accuracy_delta": [0.0, 0.0], "recall_delta": [0.0, 0.0]}
