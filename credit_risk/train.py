import argparse
import io
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from .artifacts import save_bundle, versions
from .data import load_data, split_data, fingerprint, TARGET
from .features import RAW, make_pipeline
from .metrics import metrics, threshold_study


def candidates(seed):
    result = [("dummy", DummyClassifier(strategy="prior"))]
    for weight in [None, "balanced"]:
        result.append(
            (
                f"logistic_{weight}",
                LogisticRegression(
                    C=1, class_weight=weight, max_iter=2000, random_state=seed
                ),
            )
        )
    for leaf in [10, 30]:
        result.append(
            (
                f"random_forest_leaf{leaf}",
                RandomForestClassifier(
                    n_estimators=150,
                    min_samples_leaf=leaf,
                    max_depth=14,
                    class_weight="balanced_subsample",
                    n_jobs=2,
                    random_state=seed,
                ),
            )
        )
    for depth in [2, 3]:
        result.append(
            (
                f"gradient_boosting_depth{depth}",
                GradientBoostingClassifier(
                    n_estimators=120,
                    learning_rate=0.05,
                    max_depth=depth,
                    min_samples_leaf=30,
                    random_state=seed,
                ),
            )
        )
    try:
        from xgboost import XGBClassifier

        for depth in [2, 4]:
            result.append(
                (
                    f"xgboost_depth{depth}",
                    XGBClassifier(
                        n_estimators=150,
                        max_depth=depth,
                        learning_rate=0.05,
                        min_child_weight=10,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        n_jobs=2,
                        random_state=seed,
                        eval_metric="logloss",
                    ),
                )
            )
    except ImportError:
        print("XGBoost unavailable; skipped.", flush=True)
    return result


def run(args):
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(
            "Output directory is not empty. Choose a new run directory; existing results are immutable."
        )
    out.mkdir(parents=True, exist_ok=True)
    d = load_data(args.data)
    splits, groups = split_data(d, args.seed)
    train, val = splits["train"], splits["validation"]
    X, y = d[RAW], d[TARGET].to_numpy()
    folds = list(
        StratifiedGroupKFold(3, shuffle=True, random_state=args.seed + 2).split(
            X.iloc[train], y[train], groups[train]
        )
    )
    records, fitted = [], {}

    def fit_candidate(name, model, params):
        print(f"Fitting {name}", flush=True)
        start = time.perf_counter()
        scores = cross_val_score(
            model,
            X.iloc[train],
            y[train],
            cv=folds,
            scoring="average_precision",
            n_jobs=1,
        )
        model.fit(X.iloc[train], y[train])
        seconds = time.perf_counter() - start
        start = time.perf_counter()
        prob = model.predict_proba(X.iloc[val])[:, 1]
        inference = (time.perf_counter() - start) / len(val) * 1000
        buffer = io.BytesIO()
        joblib.dump(model, buffer, compress=3)
        records.append(
            dict(
                algorithm=name,
                hyperparameters=params,
                cv_average_precision=scores.tolist(),
                cv_mean=float(scores.mean()),
                cv_std=float(scores.std()),
                validation=metrics(y[val], prob),
                training_seconds_including_cv=seconds,
                batch_inference_ms_per_row=inference,
                model_bytes=len(buffer.getvalue()),
            )
        )
        fitted[name] = model
        (out / "progress.json").write_text(json.dumps(records, indent=2))

    for name, estimator in candidates(args.seed):
        # XGBoost's missing=np.nan is a configuration sentinel, not a JSON number.
        params = json.loads(json.dumps(estimator.get_params()), parse_constant=str)
        fit_candidate(name, make_pipeline(estimator), params)
    order = sorted(
        records, key=lambda r: r["validation"]["average_precision"], reverse=True
    )
    names = [r["algorithm"] for r in order[:2]]
    voting = VotingClassifier(
        [(n, clone(fitted[n])) for n in names], voting="soft", n_jobs=1
    )
    fit_candidate("soft_voting", voting, {"members": names})
    winner = max(records, key=lambda r: r["validation"]["average_precision"])[
        "algorithm"
    ]
    base = fitted[winner]
    calibration, calibrated = [], {"uncalibrated": base}
    for method in ["uncalibrated", "sigmoid", "isotonic"]:
        print(f"Calibration: {winner} / {method}", flush=True)
        start = time.perf_counter()
        if method != "uncalibrated":
            m = CalibratedClassifierCV(
                clone(base), method=method, cv=folds, ensemble=False, n_jobs=1
            )
            m.fit(X.iloc[train], y[train])
            calibrated[method] = m
        calibration.append(
            dict(
                method=method,
                seconds=time.perf_counter() - start,
                **metrics(y[val], calibrated[method].predict_proba(X.iloc[val])[:, 1]),
            )
        )
    calibration_method = min(calibration, key=lambda r: r["brier"])["method"]
    model = calibrated[calibration_method]
    threshold, thresholds = threshold_study(
        y[val], model.predict_proba(X.iloc[val])[:, 1]
    )
    digest = fingerprint(args.data)
    metadata = dict(
        model_version=f"uci-{digest[:8]}-seed{args.seed}-{winner}-{calibration_method}",
        algorithm=winner,
        calibration=calibration_method,
        decision_threshold=threshold,
        risk_boundaries=[args.low_boundary, args.high_boundary],
        risk_boundaries_illustrative=True,
        primary_metric="validation average precision",
        threshold_objective="validation F1",
        data_sha256=digest,
        seed=args.seed,
        versions=versions(),
        rows=len(d),
        positives=int(y.sum()),
        missing_values=int(d.isna().sum().sum()),
        duplicate_ids=int(d.ID.duplicated().sum()),
        duplicate_profiles=int(d[RAW].duplicated().sum()),
        split_sizes={k: len(v) for k, v in splits.items()},
        split_prevalence={k: float(y[v].mean()) for k, v in splits.items()},
        validation=metrics(y[val], model.predict_proba(X.iloc[val])[:, 1], threshold),
    )
    bundle = dict(
        schema_version=1,
        model=model,
        metadata=metadata,
        background=X.iloc[train]
        .sample(n=min(24, len(train)), random_state=args.seed)
        .reset_index(drop=True),
    )
    save_bundle(bundle, out / "model.joblib")
    (out / "selection.json").write_text(
        json.dumps(
            dict(
                metadata=metadata,
                models=records,
                calibration=calibration,
                thresholds=thresholds,
            ),
            indent=2,
        )
    )
    pd.DataFrame(
        [
            dict(
                algorithm=r["algorithm"],
                cv_ap=r["cv_mean"],
                validation_ap=r["validation"]["average_precision"],
                validation_brier=r["validation"]["brier"],
                seconds=r["training_seconds_including_cv"],
                bytes=r["model_bytes"],
                inference_ms=r["batch_inference_ms_per_row"],
            )
            for r in records
        ]
    ).to_csv(out / "comparison.csv", index=False)
    np.savez_compressed(out / "splits.npz", **splits, groups=groups)
    print(json.dumps(metadata, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Fit on training and select on validation; evaluate test separately."
    )
    p.add_argument("--data", default="default of credit card clients.xls")
    p.add_argument("--output", default="artifacts/run-001")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--low-boundary", type=float, default=0.2)
    p.add_argument("--high-boundary", type=float, default=0.5)
    args = p.parse_args()
    if not 0 < args.low_boundary < args.high_boundary < 1:
        p.error("Risk boundaries must satisfy 0 < low < high < 1.")
    run(args)


if __name__ == "__main__":
    main()
