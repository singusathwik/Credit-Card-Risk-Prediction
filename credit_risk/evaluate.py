"""Final frozen-model evaluation. Will not overwrite an existing test report."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from .artifacts import load_bundle
from .data import load_data, fingerprint, TARGET
from .features import RAW
from .metrics import metrics, bootstrap
from .explain import explain


def save_plot(path, title=None):
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")


def run(args):
    run_dir = Path(args.run)
    out = Path(args.report)
    if (out / "evaluation.json").exists():
        raise ValueError(
            "Test evaluation already exists. Read it; do not repeatedly optimize on test outcomes."
        )
    out.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle(run_dir / "model.joblib")
    meta = bundle["metadata"]
    if fingerprint(args.data) != meta["data_sha256"]:
        raise ValueError("Dataset fingerprint differs from training.")
    data = load_data(args.data)
    split = np.load(run_dir / "splits.npz")
    test = split["test"]
    val = split["validation"]
    X = data.iloc[test][RAW]
    y = data.iloc[test][TARGET].to_numpy()
    p = bundle["model"].predict_proba(X)[:, 1]
    t = meta["decision_threshold"]
    results = metrics(y, p, t)
    groups = []
    # Observed recorded attributes only. Descriptive audit, not legal certification.
    for attribute in ["SEX", "AGE"]:
        labels = (
            X[attribute].astype(str)
            if attribute == "SEX"
            else pd.cut(
                X.AGE, [0, 30, 50, 120], labels=["18-30", "31-50", "51+"]
            ).astype(str)
        )
        for label in sorted(labels.unique()):
            mask = (labels == label).to_numpy()
            groups.append(
                dict(
                    attribute=attribute,
                    group=label,
                    n=int(mask.sum()),
                    positives=int(y[mask].sum()),
                    mean_probability=float(p[mask].mean()),
                    observed_rate=float(y[mask].mean()),
                    **metrics(y[mask], p[mask], t),
                )
            )
    errors = []
    for label, mask in [
        ("false_positive", (y == 0) & (p >= t)),
        ("false_negative", (y == 1) & (p < t)),
    ]:
        errors.append(
            dict(
                type=label,
                n=int(mask.sum()),
                mean_probability=float(p[mask].mean()) if mask.any() else None,
                median_limit=(
                    float(X.loc[mask, "LIMIT_BAL"].median()) if mask.any() else None
                ),
                median_recent_status=(
                    float(X.loc[mask, "PAY_0"].median()) if mask.any() else None
                ),
            )
        )
    report = dict(
        metadata=meta,
        test=results,
        test_at_05=metrics(y, p, 0.5),
        confidence_intervals_95=bootstrap(y, p, t, split["groups"][test]),
        subgroups=groups,
        error_analysis=errors,
        uncertainty_method="300 profile-group bootstrap replicates; fixed model; not training uncertainty",
    )
    selection = json.loads((run_dir / "selection.json").read_text())
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    (out / "comparison.csv").write_bytes((run_dir / "comparison.csv").read_bytes())
    ConfusionMatrixDisplay.from_predictions(
        y, p >= t, display_labels=["No default", "Default"], cmap="Blues"
    )
    save_plot(out / "confusion_matrix.png", "Held-out test confusion matrix")
    RocCurveDisplay.from_predictions(y, p)
    save_plot(out / "roc.png", "Held-out test ROC")
    PrecisionRecallDisplay.from_predictions(y, p)
    plt.axhline(y.mean(), ls="--", color="gray", label="Test prevalence")
    plt.legend()
    save_plot(out / "precision_recall.png", "Held-out test precision–recall")
    actual, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    report["calibration_bins"] = dict(
        predicted=predicted.tolist(), observed=actual.tolist()
    )
    plt.figure(figsize=(6, 5))
    plt.plot(predicted, actual, "o-", label=meta["calibration"])
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed default rate")
    plt.legend()
    save_plot(out / "calibration.png", "Held-out test calibration (quantile bins)")
    plt.figure(figsize=(8, 5))
    rows = selection["models"]
    plt.barh(
        [r["algorithm"] for r in rows],
        [r["validation"]["average_precision"] for r in rows],
        color="#326a85",
    )
    plt.xlabel("Validation average precision")
    save_plot(out / "comparison.png", "Model selection on validation only")
    plt.figure(figsize=(7, 4))
    plt.plot(
        [r["threshold"] for r in selection["thresholds"]],
        [r["f1"] for r in selection["thresholds"]],
        label="F1",
    )
    plt.plot(
        [r["threshold"] for r in selection["thresholds"]],
        [r["recall"] for r in selection["thresholds"]],
        label="Recall",
    )
    plt.axvline(t, ls="--", color="gray")
    plt.xlabel("Decision threshold")
    plt.ylabel("Validation metric")
    plt.legend()
    save_plot(out / "thresholds.png", "Validation threshold selection")
    train_data = data.iloc[split["train"]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label in [0, 1]:
        part = train_data[train_data[TARGET] == label]
        axes[0].hist(
            part.AGE, bins=20, alpha=0.5, density=True, label=f"Default={label}"
        )
        axes[1].hist(
            part.LIMIT_BAL, bins=20, alpha=0.5, density=True, label=f"Default={label}"
        )
    axes[0].set_xlabel("Age (years)")
    axes[1].set_xlabel("Credit limit (NT$)")
    for a in axes:
        a.set_ylabel("Density")
        a.legend()
    save_plot(out / "distributions.png", "Training feature distributions")
    # Explain validation rows, not cherry-picked high-performing test examples.
    sample = data.iloc[val][RAW].sample(min(30, len(val)), random_state=42)
    print("Generating global SHAP for 30 validation observations...", flush=True)
    explanation = explain(bundle["model"], bundle["background"], sample)
    import shap

    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save_plot(
        out / "shap_beeswarm.png",
        "Permutation SHAP: class-1 probability (validation sample)",
    )
    shap.plots.bar(explanation, max_display=15, show=False)
    save_plot(out / "shap_importance.png", "Mean absolute SHAP (30 validation rows)")
    importance = np.abs(explanation.values).mean(axis=0)
    top = int(importance.argmax())
    plt.figure(figsize=(6, 4))
    plt.scatter(sample.iloc[:, top], explanation.values[:, top], alpha=0.7)
    plt.xlabel(RAW[top])
    plt.ylabel("SHAP contribution (probability)")
    save_plot(
        out / "shap_relationship.png", "Sampled relationship, not a causal effect"
    )
    report["shap"] = dict(
        sample_size=len(sample),
        background_size=len(bundle["background"]),
        mean_absolute_contributions={n: float(v) for n, v in zip(RAW, importance)},
        max_additivity_error=float(
            np.max(
                np.abs(
                    explanation.base_values
                    + explanation.values.sum(axis=1)
                    - bundle["model"].predict_proba(sample)[:, 1]
                )
            )
        ),
    )
    (out / "evaluation.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Frozen-model test evaluation",
        "",
        f"Model: `{meta['model_version']}`",
        "",
        f"Threshold: {t:.2f}; chosen on validation F1.",
        "",
        "| Metric | Test value | 95% cluster bootstrap interval |",
        "|---|---:|---|",
    ]
    for key, value in results.items():
        if key != "confusion_matrix":
            lo, hi = report["confidence_intervals_95"][key]
            lines.append(f"| {key} | {value:.4f} | {lo:.4f}–{hi:.4f} |")
    lines += [
        "",
        f"Confusion matrix (true rows, predicted columns): `{results['confusion_matrix']}`.",
        "",
        "Intervals condition on this fitted model and cohort. They do not measure population drift or training uncertainty.",
        "Error aggregates, subgroup metrics and calibration bins: `evaluation.json`. No applicant rows are published.",
    ]
    (out / "EVALUATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="default of credit card clients.xls")
    p.add_argument("--run", default="artifacts/run-001")
    p.add_argument("--report", default="reports/run-001")
    run(p.parse_args())


if __name__ == "__main__":
    main()
