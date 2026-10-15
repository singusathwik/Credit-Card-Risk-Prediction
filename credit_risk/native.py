"""Native categorical candidates; all learned statistics remain inside CV fits."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from .features import CAT, FeatureBuilder


class NativeCategories(TransformerMixin, BaseEstimator):
    """Stable categorical representation for both integer training and float API data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = X.copy()
        for col in CAT:
            result[col] = result[col].map(
                lambda value: (
                    "missing" if pd.isna(value) else format(float(value), ".15g")
                )
            )
        return result


def native_candidates(seed=42):
    return {
        f"catboost_depth{depth}": Pipeline(
            [
                ("features", FeatureBuilder()),
                ("categories", NativeCategories()),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=600,
                        depth=depth,
                        learning_rate=0.04,
                        l2_leaf_reg=10,
                        cat_features=CAT,
                        loss_function="Logloss",
                        random_seed=seed,
                        thread_count=2,
                        verbose=False,
                        allow_writing_files=False,
                    ),
                ),
            ]
        )
        for depth in (4, 5, 6)
    }
