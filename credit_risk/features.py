"""One feature contract shared by fitting, predictions and SHAP."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

STATUS = [f"PAY_{i}" for i in (0, 2, 3, 4, 5, 6)]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]
RAW = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"] + STATUS + BILLS + PAYMENTS
CAT = ["SEX", "EDUCATION", "MARRIAGE"] + STATUS
DERIVED = [
    "AVG_BILL_AMT",
    "STD_BILL_AMT",
    "AVG_PAY_AMT",
    "PAY_CONSISTENCY",
    "UTILIZATION_RATIO",
    "BILL_CHANGE",
    "DELAY_COUNT",
    "MAX_PAY_DELAY",
]
NUM = [c for c in RAW if c not in CAT] + DERIVED


def validate_payload(payload):
    if not isinstance(payload, dict):
        raise ValueError("Expected one JSON object containing all 23 raw fields.")
    if set(payload) != set(RAW):
        raise ValueError(
            f"Missing fields: {sorted(set(RAW)-set(payload))}; unknown fields: {sorted(set(payload)-set(RAW))}"
        )
    values = {}
    for name in RAW:
        v = payload[name]
        if v is None or isinstance(v, (bool, list, dict)):
            raise ValueError(f"{name}: a finite number is required.")
        try:
            v = float(v)
        except (ValueError, TypeError):
            raise ValueError(f"{name}: a finite number is required.") from None
        if not np.isfinite(v):
            raise ValueError(f"{name}: a finite number is required.")
        codes = (
            {"SEX": [1, 2], "EDUCATION": list(range(7)), "MARRIAGE": list(range(4))}
        ).get(name)
        if name in STATUS:
            codes = list(range(-2, 10))
        if codes is not None and v not in codes:
            raise ValueError(f"{name}: allowed codes are {codes}.")
        low, high = (
            (18, 100)
            if name == "AGE"
            else (
                (1, 10000000)
                if name == "LIMIT_BAL"
                else (-10000000, 10000000) if name in BILLS else (0, 10000000)
            )
        )
        if codes is None and (
            not low <= v <= high or (name == "AGE" and not v.is_integer())
        ):
            raise ValueError(
                f"{name}: expected a value between {low} and {high}"
                + (" (whole years)." if name == "AGE" else ".")
            )
        values[name] = v
    return pd.DataFrame([values], columns=RAW)


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.feature_names_in_ = np.array(RAW, dtype=object)
        return self

    def transform(self, X):
        d = pd.DataFrame(X, columns=RAW).copy().astype(float)
        d = d.replace([np.inf, -np.inf], np.nan)
        d["AVG_BILL_AMT"] = d[BILLS].mean(axis=1)
        d["STD_BILL_AMT"] = d[BILLS].std(axis=1)
        d["AVG_PAY_AMT"] = d[PAYMENTS].mean(axis=1)
        d["PAY_CONSISTENCY"] = d[PAYMENTS].std(axis=1)
        d["UTILIZATION_RATIO"] = d.AVG_BILL_AMT / d.LIMIT_BAL.where(d.LIMIT_BAL > 0)
        d["BILL_CHANGE"] = d.BILL_AMT1 - d.BILL_AMT6
        d["DELAY_COUNT"] = (
            (d[STATUS] > 0).sum(axis=1).where(d[STATUS].notna().any(axis=1))
        )
        d["MAX_PAY_DELAY"] = d[STATUS].clip(lower=0).max(axis=1)
        return d


def make_pipeline(estimator):
    numerical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return Pipeline(
        [
            ("features", FeatureBuilder()),
            (
                "preprocess",
                ColumnTransformer(
                    [("numeric", numerical, NUM), ("categorical", categorical, CAT)]
                ),
            ),
            ("model", estimator),
        ]
    )
