"""Accuracy under recall constraints, evaluated with nested grouped CV.

Only original training rows enter search/CV. The original validation set is a
reused development audit, never advertised as a new holdout. Original test rows
are excluded and never scored. Frozen protocol is saved before any model fits.
"""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from .artifacts import load_bundle, save_bundle
from .data import TARGET, fingerprint, load_data
from .features import RAW, make_pipeline
from .metrics import metrics


def candidate_models(seed=42):
    common = dict(
        n_jobs=2,
        random_state=seed,
        eval_metric="logloss",
        subsample=0.9,
        colsample_bytree=0.9,
    )
    return {
        "incumbent_xgb": make_pipeline(
            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.05,
                min_child_weight=10,
                **common,
            )
        ),
        "xgb_shallow": make_pipeline(
            XGBClassifier(
                n_estimators=300,
                max_depth=2,
                learning_rate=0.05,
                min_child_weight=20,
                reg_lambda=5,
                **common,
            )
        ),
        "xgb_regularized_d3": make_pipeline(
            XGBClassifier(
                n_estimators=400,
                max_depth=3,
                learning_rate=0.035,
                min_child_weight=20,
                reg_lambda=5,
                reg_alpha=0.5,
                **common,
            )
        ),
        "xgb_regularized_d4": make_pipeline(
            XGBClassifier(
                n_estimators=450,
                max_depth=4,
                learning_rate=0.035,
                min_child_weight=30,
                reg_lambda=10,
                reg_alpha=1,
                gamma=0.1,
                **common,
            )
        ),
        "hist_7_leaves": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=7,
                min_samples_leaf=40,
                l2_regularization=10,
                early_stopping=False,
                random_state=seed,
            )
        ),
        "hist_15_leaves": make_pipeline(
            HistGradientBoostingClassifier(
                max_iter=250,
                learning_rate=0.06,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=20,
                early_stopping=False,
                random_state=seed,
            )
        ),
        "forest_unweighted": make_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=16,
                min_samples_leaf=10,
                max_features=0.7,
                random_state=seed,
                n_jobs=2,
            )
        ),
        "extra_trees": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=250,
                max_depth=18,
                min_samples_leaf=15,
                max_features=0.8,
                random_state=seed,
                n_jobs=2,
            )
        ),
    }


def checked_folds(X, y, groups, seed, count=3):
    folds = list(
        StratifiedGroupKFold(count, shuffle=True, random_state=seed).split(X, y, groups)
    )
    seen = np.zeros(len(y), dtype=int)
    for fit, score in folds:
        if set(groups[fit]) & set(groups[score]):
            raise ValueError("Related observations crossed a fold.")
        seen[score] += 1
    if not np.all(seen == 1):
        raise ValueError("OOF predictions must cover each observation exactly once.")
    return folds


def out_of_fold(model, X, y, folds):
    p = np.full(len(y), np.nan)
    for fit, score in folds:
        estimator = clone(model).fit(X.iloc[fit], y[fit])
        p[score] = estimator.predict_proba(X.iloc[score])[:, 1]
    if not np.isfinite(p).all():
        raise ValueError("Incomplete or invalid OOF probabilities.")
    return p


def constrained_threshold(y, p, recall_floor, grid=None):
    """Maximize accuracy subject to observed OOF recall; never fit on scored rows."""
    y, p = np.asarray(y), np.asarray(p)
    if y.ndim != 1 or p.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError("Aligned binary labels and probabilities are required.")
    if (
        not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
        or not 0 < recall_floor <= 1
    ):
        raise ValueError("Invalid probabilities or recall constraint.")
    grid = np.linspace(0, 1, 201) if grid is None else np.asarray(grid)
    rows = []
    for threshold in grid:
        pred = p >= threshold
        recall = float(pred[y == 1].mean())
        rows.append(
            dict(
                threshold=float(threshold),
                accuracy=float((pred == y).mean()),
                recall=recall,
            )
        )
    feasible = [r for r in rows if r["recall"] >= recall_floor]
    if not feasible:
        raise ValueError("No threshold satisfies the recall constraint.")
    # Prefer higher recall for equal accuracy, then a deterministic threshold.
    best = max(feasible, key=lambda r: (r["accuracy"], r["recall"], r["threshold"]))
    return best["threshold"], rows


def select_on_inner_oof(
    models, X, y, groups, seed, floor, max_drop, baseline_threshold
):
    folds = checked_folds(X, y, groups, seed)
    predictions = {}
    records = []
    for name, model in models.items():
        print(f"  Inner OOF: {name}", flush=True)
        start = time.perf_counter()
        predictions[name] = out_of_fold(model, X, y, folds)
        records.append({"algorithm": name, "seconds": time.perf_counter() - start})
    baseline = metrics(y, predictions["incumbent_xgb"], baseline_threshold)
    target = max(floor, baseline["recall"] - max_drop)
    for record in records:
        p = predictions[record["algorithm"]]
        threshold, _ = constrained_threshold(y, p, target)
        record.update(
            threshold=threshold, recall_floor=target, metrics=metrics(y, p, threshold)
        )
    winner = max(
        records,
        key=lambda r: (
            r["metrics"]["accuracy"],
            r["metrics"]["recall"],
            r["metrics"]["average_precision"],
        ),
    )
    return winner, records, baseline


def paired_intervals(y, baseline, challenger, groups, repeats=500, seed=20261015):
    """Paired profile-group bootstrap of fixed OOF decisions; exploratory intervals."""
    rng = np.random.default_rng(seed)
    keys, inverse = np.unique(groups, return_inverse=True)
    # Group sufficient statistics make bootstrap efficient and preserve pairing.
    n = np.bincount(inverse)
    positives = np.bincount(inverse, weights=(y == 1))
    correct_delta = np.bincount(
        inverse, weights=(challenger == y).astype(int) - (baseline == y).astype(int)
    )
    true_positive_delta = np.bincount(
        inverse,
        weights=((challenger == 1) & (y == 1)).astype(int)
        - ((baseline == 1) & (y == 1)).astype(int),
    )
    draws = []
    for _ in range(repeats):
        sample = rng.integers(0, len(keys), len(keys))
        draws.append(
            [
                correct_delta[sample].sum() / n[sample].sum(),
                true_positive_delta[sample].sum() / positives[sample].sum(),
            ]
        )
    return dict(
        zip(
            ["accuracy_delta", "recall_delta"],
            np.quantile(draws, [0.025, 0.975], axis=0).T.tolist(),
        )
    )


def passes_guardrails(base, new, floor, max_drop, min_gain=0.005):
    return (
        new["accuracy"] >= base["accuracy"] + min_gain
        and new["recall"] >= max(floor, base["recall"] - max_drop)
        and new["brier"] <= base["brier"] + 0.005
    )


def run(args):
    output, report = Path(args.output), Path(args.report)
    if any(p.exists() and any(p.iterdir()) for p in (output, report)):
        raise ValueError(
            "Use empty run/report directories. Do not overwrite earlier experiments."
        )
    original = Path(args.baseline)
    incumbent = load_bundle(original / "model.joblib")
    if fingerprint(args.data) != incumbent["metadata"]["data_sha256"]:
        raise ValueError("Dataset differs from the original experiment.")
    if (
        incumbent["metadata"]["algorithm"] != "xgboost_depth4"
        or incumbent["metadata"]["calibration"] != "uncalibrated"
    ):
        raise ValueError("This comparison requires the recorded run-001 incumbent.")
    split = np.load(original / "splits.npz")
    train, validation, excluded = split["train"], split["validation"], split["test"]
    for a, b in [(train, validation), (train, excluded), (validation, excluded)]:
        if set(split["groups"][a]) & set(split["groups"][b]):
            raise ValueError("Original partitions share groups.")
    models = candidate_models()
    if args.profile == "native":
        from .native import native_candidates

        models = {"incumbent_xgb": models["incumbent_xgb"], **native_candidates()}
    if args.profile == "behavior":
        from .behavior import behavior_candidates

        models = behavior_candidates(models["incumbent_xgb"])
    selection_floor = max(args.recall_floor, args.selection_recall_floor)
    protocol = dict(
        seed=20261015,
        profile=args.profile,
        selection_recall_floor=selection_floor,
        outer_folds=3,
        inner_folds=3,
        rows_used_for_search=len(train),
        validation_role="Previously used validation: development audit only; no model/threshold changes after audit.",
        test_role="Excluded, never scored or used for tuning.",
        recall_floor=args.recall_floor,
        maximum_recall_drop=args.max_recall_drop,
        min_accuracy_gain=0.005,
        max_brier_increase=0.005,
        promotion="Nested and validation point estimates must pass guardrails; nested paired accuracy CI lower bound > 0.",
        threshold_grid="0.000 to 1.000 in increments of 0.005; maximize accuracy then recall.",
        candidate_parameters={
            n: json.loads(
                json.dumps(model.get_params(), default=str), parse_constant=str
            )
            for n, model in models.items()
        },
        source_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["improve.py", "features.py"]
            + (["behavior.py"] if args.profile == "behavior" else [])
        },
        caveat="Nested CV on previously explored training data is exploratory, not a fresh external holdout.",
    )
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    (report / "protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False)
    )
    # Reading the workbook is necessary, but no excluded test features/labels are indexed for fitting/scoring.
    data = load_data(args.data)
    X = data.iloc[train][RAW].reset_index(drop=True)
    y = data.iloc[train][TARGET].to_numpy()
    groups = split["groups"][train]
    outer = checked_folds(X, y, groups, protocol["seed"])
    old_p, new_p = np.empty(len(y)), np.empty(len(y))
    thresholds, outer_ids = np.empty(len(y)), np.empty(len(y), dtype=int)
    baseline_threshold = incumbent["metadata"]["decision_threshold"]
    folds = []
    for i, (fit, score) in enumerate(outer):
        print(f"Outer fold {i+1}/3", flush=True)
        winner, candidates, inner_base = select_on_inner_oof(
            models,
            X.iloc[fit],
            y[fit],
            groups[fit],
            protocol["seed"] + i + 1,
            selection_floor,
            args.max_recall_drop,
            baseline_threshold,
        )
        old = clone(models["incumbent_xgb"]).fit(X.iloc[fit], y[fit])
        chosen = clone(models[winner["algorithm"]]).fit(X.iloc[fit], y[fit])
        old_p[score] = old.predict_proba(X.iloc[score])[:, 1]
        new_p[score] = chosen.predict_proba(X.iloc[score])[:, 1]
        thresholds[score] = winner["threshold"]
        outer_ids[score] = i
        folds.append(
            dict(
                fold=i + 1,
                selected=winner["algorithm"],
                threshold=winner["threshold"],
                fit_rows=len(fit),
                score_rows=len(score),
                inner_candidates=candidates,
                inner_baseline=inner_base,
                baseline=metrics(y[score], old_p[score], baseline_threshold),
                challenger=metrics(y[score], new_p[score], winner["threshold"]),
            )
        )
        (report / "progress.json").write_text(
            json.dumps(folds, indent=2, allow_nan=False)
        )
    nested_base, nested_new = metrics(y, old_p, baseline_threshold), metrics(
        y, new_p, thresholds
    )
    intervals = paired_intervals(
        y, old_p >= baseline_threshold, new_p >= thresholds, groups
    )
    print("Final selection: OOF on all original training rows", flush=True)
    winner, final_candidates, final_base = select_on_inner_oof(
        models,
        X,
        y,
        groups,
        protocol["seed"] + 100,
        selection_floor,
        args.max_recall_drop,
        baseline_threshold,
    )
    model = clone(models[winner["algorithm"]]).fit(X, y)
    # Freeze candidate and threshold before reading validation features/outcomes.
    (report / "frozen_selection.json").write_text(
        json.dumps(winner, indent=2, allow_nan=False)
    )
    valX, valY = data.iloc[validation][RAW], data.iloc[validation][TARGET].to_numpy()
    val_old = incumbent["model"].predict_proba(valX)[:, 1]
    val_new = model.predict_proba(valX)[:, 1]
    val_base, val_challenger = metrics(valY, val_old, baseline_threshold), metrics(
        valY, val_new, winner["threshold"]
    )
    eligible = (
        passes_guardrails(
            nested_base, nested_new, args.recall_floor, args.max_recall_drop
        )
        and intervals["accuracy_delta"][0] > 0
        and passes_guardrails(
            val_base, val_challenger, args.recall_floor, args.max_recall_drop
        )
    )
    meta = copy.deepcopy(incumbent["metadata"])
    protocol_hash = hashlib.sha256((report / "protocol.json").read_bytes()).hexdigest()[
        :10
    ]
    meta.update(
        model_version=f"accuracy-recall-{protocol_hash}-{winner['algorithm']}",
        algorithm=winner["algorithm"],
        decision_threshold=winner["threshold"],
        primary_metric="inner-OOF accuracy subject to recall constraints",
        threshold_objective="inner-OOF constrained accuracy",
        validation=val_challenger,
        evaluation_status="exploratory nested CV and reused validation; no new test score",
        promotion_eligible=eligible,
    )
    save_bundle(
        dict(
            schema_version=1,
            extra_packages=(
                ["lightgbm"]
                if args.profile == "behavior"
                else ["catboost"] if winner["algorithm"].startswith("catboost") else []
            ),
            model=model,
            metadata=meta,
            background=incumbent["background"],
        ),
        output / "model.joblib",
    )
    # Individual predictions/indices stay in ignored artifact storage, never public reports.
    np.savez_compressed(
        output / "nested_predictions.npz",
        row_indices=train,
        groups=groups,
        y=y,
        baseline=old_p,
        challenger=new_p,
        thresholds=thresholds,
        outer_fold=outer_ids,
    )
    result = dict(
        protocol=protocol,
        metadata=meta,
        folds=folds,
        nested_baseline=nested_base,
        nested_challenger=nested_new,
        paired_intervals_95=intervals,
        final_inner_candidates=final_candidates,
        final_inner_baseline=final_base,
        validation_baseline=val_base,
        validation_challenger=val_challenger,
        promotion_eligible=eligible,
        original_test_evaluated=False,
        default_model_changed=False,
    )
    (report / "comparison.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    pd.DataFrame(
        [
            dict(
                algorithm=r["algorithm"],
                threshold=r["threshold"],
                **{k: v for k, v in r["metrics"].items() if k != "confusion_matrix"},
            )
            for r in final_candidates
        ]
    ).to_csv(report / "inner_candidates.csv", index=False)
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "nested_baseline",
                    "nested_challenger",
                    "paired_intervals_95",
                    "validation_baseline",
                    "validation_challenger",
                    "promotion_eligible",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="default of credit card clients.xls")
    p.add_argument("--baseline", default="artifacts/run-001")
    p.add_argument("--output", default="artifacts/accuracy-recall-001")
    p.add_argument("--report", default="reports/accuracy-recall-001")
    p.add_argument("--recall-floor", type=float, default=0.56)
    p.add_argument("--max-recall-drop", type=float, default=0.03)
    p.add_argument(
        "--profile", choices=["standard", "native", "behavior"], default="standard"
    )
    p.add_argument("--selection-recall-floor", type=float, default=0.56)
    args = p.parse_args()
    if (
        not 0 < args.recall_floor <= 1
        or not 0 <= args.max_recall_drop < 1
        or not 0 < args.selection_recall_floor <= 1
    ):
        p.error("Invalid recall constraints.")
    with threadpool_limits(limits=2):
        run(args)


if __name__ == "__main__":
    main()
