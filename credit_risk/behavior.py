"""Row-local repayment dynamics available before the next-month target.

No population aggregates, IDs, target encoding or outcome-derived features.
Months are newest first. Ratios to a preceding bill are proxies, not a claim
about contractual due amounts. All clipping bounds are fixed before fitting.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from .features import RAW, STATUS, BILLS, PAYMENTS, FeatureBuilder


class RepaymentFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enriched=True):
        self.enriched = enriched

    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(RAW, dtype=object)
        return self

    def transform(self, X):
        raw = (
            pd.DataFrame(X, columns=RAW)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        if not self.enriched:
            return raw
        result = FeatureBuilder().transform(raw)
        limit = raw.LIMIT_BAL.where(raw.LIMIT_BAL > 0)
        bills, payments, status = raw[BILLS], raw[PAYMENTS], raw[STATUS]
        delay = status.clip(lower=0)
        extra = {}
        for i, (bill, payment) in enumerate(zip(BILLS, PAYMENTS), start=1):
            extra[f"BILL_LIMIT_{i}"] = (raw[bill] / limit).clip(-5, 5)
            extra[f"PAYMENT_LIMIT_{i}"] = (raw[payment] / limit).clip(0, 5)
            if i < 6:
                prior = raw[BILLS[i]]
                extra[f"PAYMENT_PRIOR_BILL_{i}"] = (
                    raw[payment] / prior.where(prior > 0)
                ).clip(0, 5)
                extra[f"BALANCE_FLOW_LIMIT_{i}"] = (
                    (raw[bill] - prior + raw[payment]) / limit
                ).clip(-5, 5)
        extra["RECENT_BILL_LIMIT"] = bills.iloc[:, :3].mean(axis=1) / limit
        extra["OLDER_BILL_LIMIT"] = bills.iloc[:, 3:].mean(axis=1) / limit
        extra["BILL_TREND_LIMIT"] = (
            bills.iloc[:, :3].mean(axis=1) - bills.iloc[:, 3:].mean(axis=1)
        ) / limit
        extra["PAYMENT_TREND_LIMIT"] = (
            payments.iloc[:, :3].mean(axis=1) - payments.iloc[:, 3:].mean(axis=1)
        ) / limit
        extra["PAYMENT_VOLATILITY"] = payments.std(axis=1) / (
            payments.mean(axis=1).abs() + 1
        )
        extra["BILL_VOLATILITY_LIMIT"] = bills.std(axis=1) / limit
        extra["RECENT_DELAY_MEAN"] = delay.iloc[:, :3].mean(axis=1)
        extra["OLDER_DELAY_MEAN"] = delay.iloc[:, 3:].mean(axis=1)
        extra["DELAY_TREND"] = extra["RECENT_DELAY_MEAN"] - extra["OLDER_DELAY_MEAN"]
        extra["DELAY_RECENCY_WEIGHTED"] = (
            delay.mul([6, 5, 4, 3, 2, 1]).sum(axis=1, min_count=1) / 21
        )
        extra["DELAY_SEVERITY_SUM"] = delay.sum(axis=1, min_count=1)
        extra["DELAY_TWO_PLUS_COUNT"] = (
            (status >= 2).sum(axis=1).where(status.notna().any(axis=1))
        )
        # Newest-first uninterrupted delayed months; unknown stops the run.
        extra["CURRENT_DELAY_STREAK"] = (
            (status > 0).cumprod(axis=1).sum(axis=1).where(status.notna().any(axis=1))
        )
        for code in (-2, -1, 0):
            extra[f"STATUS_CODE_{code}_COUNT"] = (
                (status == code).sum(axis=1).where(status.notna().any(axis=1))
            )
        extra["ZERO_PAYMENT_COUNT"] = (
            (payments == 0).sum(axis=1).where(payments.notna().any(axis=1))
        )
        extra["ZERO_PAYMENT_POSITIVE_BILL_COUNT"] = (
            pd.DataFrame(
                (payments.to_numpy() == 0) & (bills.to_numpy() > 0), index=raw.index
            )
            .sum(axis=1)
            .where(payments.notna().any(axis=1))
        )
        return pd.concat(
            [result, pd.DataFrame(extra, index=raw.index)], axis=1
        ).replace([np.inf, -np.inf], np.nan)


def behavior_candidates(incumbent, seed=42):
    def pipe(model, enriched=True):
        return Pipeline([("features", RepaymentFeatures(enriched)), ("model", model)])

    def xgb(depth):
        return XGBClassifier(
            n_estimators=350,
            max_depth=depth,
            learning_rate=0.035,
            min_child_weight=20,
            reg_lambda=10,
            reg_alpha=0.5,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=2,
            random_state=seed,
            eval_metric="logloss",
        )

    def lgb(leaves):
        return LGBMClassifier(
            n_estimators=350,
            num_leaves=leaves,
            learning_rate=0.035,
            min_child_samples=60,
            reg_lambda=10,
            reg_alpha=0.5,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )

    models = {"incumbent_xgb": incumbent}
    models.update({f"ordinal_xgb_d{d}": pipe(xgb(d), False) for d in (3, 4)})
    models.update({f"behavior_xgb_d{d}": pipe(xgb(d)) for d in (3, 4, 5)})
    models.update(
        {f"behavior_lgb_{leaves}": pipe(lgb(leaves)) for leaves in (7, 15, 31)}
    )
    models["behavior_hist7"] = pipe(
        HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.05,
            max_leaf_nodes=7,
            min_samples_leaf=50,
            l2_regularization=15,
            early_stopping=False,
            random_state=seed,
        )
    )
    models["behavior_blend"] = Pipeline(
        [
            (
                "model",
                VotingClassifier(
                    estimators=[("xgb", pipe(xgb(4))), ("lgb", pipe(lgb(15)))],
                    voting="soft",
                ),
            )
        ]
    )
    models["incumbent_behavior_blend"] = Pipeline(
        [
            (
                "model",
                VotingClassifier(
                    estimators=[
                        ("incumbent", incumbent),
                        ("xgb", pipe(xgb(4))),
                        ("lgb", pipe(lgb(15))),
                    ],
                    voting="soft",
                ),
            )
        ]
    )
    return models
