"""Use all non-test records for development, freeze once, then audit once.

The original test result was previously published. The comparison is therefore
an audit on a reused benchmark, not fresh external confirmation. No test rows,
labels, predictions or metrics enter candidate/threshold selection here.
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
from threadpoolctl import threadpool_limits
from .artifacts import load_bundle, save_bundle
from .behavior import behavior_candidates
from .data import load_data, fingerprint, TARGET
from .features import RAW
from .improve import (
    candidate_models,
    checked_folds,
    out_of_fold,
    constrained_threshold,
    paired_intervals,
    passes_guardrails,
)
from .metrics import metrics, bootstrap


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def train(args):
    output, report = Path(args.output), Path(args.report)
    if any(p.exists() and any(p.iterdir()) for p in (output, report)):
        raise ValueError("Use new, empty run directories.")
    original = load_bundle("artifacts/run-001/model.joblib")
    if fingerprint(args.data) != original["metadata"]["data_sha256"]:
        raise ValueError("Dataset fingerprint differs.")
    split = np.load("artifacts/run-001/splits.npz")
    development = np.sort(np.concatenate([split["train"], split["validation"]]))
    test = split["test"]
    assert not set(development) & set(test)
    assert not set(split["groups"][development]) & set(split["groups"][test])
    models = behavior_candidates(candidate_models()["incumbent_xgb"])
    protocol = dict(
        development_rows=len(development),
        test_rows=len(test),
        seed=20261015,
        selection_folds=5,
        selection_recall_floor=0.58,
        threshold_grid="0.000 to 1.000 by 0.001",
        selection="Maximize grouped OOF accuracy subject to recall >= max(0.58, incumbent OOF recall minus 0.03); ties favor recall then AP.",
        final_fit="All 24,004 original non-test rows; prior validation is now development data.",
        test_policy="Exactly one frozen-candidate audit. No model or threshold changes after test outcomes. Previously reported benchmark, not fresh external validation.",
        adoption=dict(
            min_accuracy_gain=0.005,
            maximum_recall_loss=0.03,
            recall_floor=0.56,
            maximum_brier_increase=0.005,
        ),
        candidate_parameters={
            n: json.loads(json.dumps(m.get_params(), default=str), parse_constant=str)
            for n, m in models.items()
        },
        source_sha256={
            n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ["finalize.py", "behavior.py", "improve.py", "features.py"]
        },
    )
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    write_json(report / "protocol.json", protocol)
    data = load_data(args.data)
    X = data.iloc[development][RAW].reset_index(drop=True)
    y = data.iloc[development][TARGET].to_numpy()
    groups = split["groups"][development]
    folds = checked_folds(X, y, groups, protocol["seed"], count=5)
    records, probabilities = [], {}
    baseline_threshold = original["metadata"]["decision_threshold"]
    for name, model in models.items():
        print(f"Five-fold development OOF: {name}", flush=True)
        start = time.perf_counter()
        probabilities[name] = out_of_fold(model, X, y, folds)
        if name == "incumbent_xgb":
            baseline = metrics(y, probabilities[name], baseline_threshold)
            target = max(0.58, baseline["recall"] - 0.03)
        threshold, _ = constrained_threshold(
            y, probabilities[name], target, grid=np.linspace(0, 1, 1001)
        )
        records.append(
            dict(
                algorithm=name,
                threshold=threshold,
                metrics=metrics(y, probabilities[name], threshold),
                seconds=time.perf_counter() - start,
            )
        )
        write_json(report / "selection_progress.json", records)
    winner = max(
        records,
        key=lambda r: (
            r["metrics"]["accuracy"],
            r["metrics"]["recall"],
            r["metrics"]["average_precision"],
        ),
    )
    print(f"Frozen choice: {winner['algorithm']} at {winner['threshold']}", flush=True)
    model = clone(models[winner["algorithm"]]).fit(X, y)
    meta = copy.deepcopy(original["metadata"])
    # Do not attach the original validation performance to a new fitted model.
    meta.pop("validation", None)
    digest = hashlib.sha256((report / "protocol.json").read_bytes()).hexdigest()[:10]
    meta.update(
        model_version=f"full-development-{digest}-{winner['algorithm']}",
        algorithm=winner["algorithm"],
        decision_threshold=winner["threshold"],
        split_sizes=dict(train=len(development), validation=0, test=len(test)),
        primary_metric="grouped development OOF constrained accuracy",
        threshold_objective="grouped development OOF constrained accuracy",
        selection_scope="five-fold development OOF",
        evaluation_status="one frozen audit on the previously reported test benchmark",
        development_oof=winner["metrics"],
        development_recall_target=target,
    )
    save_bundle(
        dict(
            schema_version=1,
            extra_packages=["lightgbm"],
            model=model,
            metadata=meta,
            background=original["background"],
        ),
        output / "model.joblib",
    )
    frozen = dict(
        winner=winner,
        model_version=meta["model_version"],
        model_sha256=hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256(
            (report / "protocol.json").read_bytes()
        ).hexdigest(),
        original_test_scored=False,
    )
    write_json(report / "frozen_selection.json", frozen)
    write_json(
        report / "selection.json",
        dict(
            models=records,
            baseline_oof=baseline,
            selection_scope=meta["selection_scope"],
        ),
    )
    np.savez_compressed(
        output / "development.npz",
        row_indices=development,
        groups=groups,
        y=y,
        selected_oof=probabilities[winner["algorithm"]],
        baseline_oof=probabilities["incumbent_xgb"],
    )
    np.savez_compressed(
        output / "splits.npz",
        train=development,
        validation=np.array([], dtype=int),
        test=test,
        groups=split["groups"],
    )


def audit(args):
    output, report = Path(args.output), Path(args.report)
    frozen = json.loads((report / "frozen_selection.json").read_text())
    if (
        hashlib.sha256((output / "model.joblib").read_bytes()).hexdigest()
        != frozen["model_sha256"]
    ):
        raise ValueError("Model changed after freezing.")
    if (
        hashlib.sha256((report / "protocol.json").read_bytes()).hexdigest()
        != frozen["protocol_sha256"]
    ):
        raise ValueError("Protocol changed after freezing.")
    if (report / "audit_started.json").exists():
        raise ValueError(
            "This candidate has already entered its one-time audit. Do not retune or rerun."
        )
    bundle = load_bundle(output / "model.joblib")
    original = load_bundle("artifacts/run-001/model.joblib")
    if fingerprint(args.data) != bundle["metadata"]["data_sha256"]:
        raise ValueError("Dataset differs.")
    split = np.load(output / "splits.npz")
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["groups"][split["train"]]) & set(
        split["groups"][split["test"]]
    )
    with (report / "audit_started.json").open("x") as handle:
        json.dump(
            dict(
                model_sha256=frozen["model_sha256"], test_use="frozen comparison only"
            ),
            handle,
        )
    data = load_data(args.data)
    X = data.iloc[split["test"]][RAW]
    y = data.iloc[split["test"]][TARGET].to_numpy()
    old_p = original["model"].predict_proba(X)[:, 1]
    new_p = bundle["model"].predict_proba(X)[:, 1]
    old_t, new_t = (
        original["metadata"]["decision_threshold"],
        bundle["metadata"]["decision_threshold"],
    )
    base, new = metrics(y, old_p, old_t), metrics(y, new_p, new_t)
    groups = split["groups"][split["test"]]
    intervals = paired_intervals(
        y, old_p >= old_t, new_p >= new_t, groups, repeats=2000
    )
    eligible = passes_guardrails(base, new, 0.56, 0.03)
    result = dict(
        metadata=bundle["metadata"],
        baseline=base,
        test=new,
        paired_intervals_95=intervals,
        confidence_intervals_95=bootstrap(y, new_p, new_t, groups),
        promotion_eligible=eligible,
        uncertainty_method="Fixed-prediction group bootstrap; excludes training/search uncertainty and benchmark reuse bias.",
        test_reuse_notice="The baseline result was previously published. Candidate and threshold were frozen before this audit; this is not fresh external validation.",
    )
    write_json(report / "audit.json", result)
    np.savez_compressed(
        output / "audit_predictions.npz",
        y=y,
        baseline=old_p,
        challenger=new_p,
        row_indices=split["test"],
        groups=groups,
    )
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["train", "audit"])
    p.add_argument("--data", default="default of credit card clients.xls")
    p.add_argument("--output", default="artifacts/full-development-004")
    p.add_argument("--report", default="reports/full-development-004")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        (train if args.stage == "train" else audit)(args)


if __name__ == "__main__":
    main()
