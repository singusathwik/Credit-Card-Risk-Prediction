"""Render the bounded accuracy/recall experiment without refitting or retuning."""

import json
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/accuracy-recall-001"


def main(report=REPORT):
    REPORT = Path(report)
    result = json.loads((REPORT / "comparison.json").read_text())
    nested_old, nested_new = result["nested_baseline"], result["nested_challenger"]
    val_old, val_new = result["validation_baseline"], result["validation_challenger"]
    intervals = result["paired_intervals_95"]
    rows = []
    for label, a, b in [
        ("Nested grouped CV (18,000 training records)", nested_old, nested_new),
        ("Reused validation audit (6,004 records)", val_old, val_new),
    ]:
        rows.append(
            f"| {label} | {a['accuracy']:.2%} | {b['accuracy']:.2%} | {100*(b['accuracy']-a['accuracy']):+.2f} pp | {a['recall']:.2%} | {b['recall']:.2%} | {100*(b['recall']-a['recall']):+.2f} pp |"
        )
    decision = (
        "The challenger passed the prespecified point-estimate guardrails and paired accuracy interval check. This is exploratory evidence, not a new held-out test result."
        if result["promotion_eligible"]
        else "The challenger did not pass all prespecified guardrails. The existing application model is retained; an accuracy increase alone is insufficient to justify replacing it."
    )
    candidate_rows = "\n".join(
        f"| {r['algorithm']} | {r['threshold']:.3f} | {r['metrics']['accuracy']:.2%} | {r['metrics']['recall']:.2%} | {r['metrics']['average_precision']:.4f} | {r['metrics']['brier']:.4f} |"
        for r in result["final_inner_candidates"]
    )
    fold_rows = "\n".join(
        f"| {r['fold']} | {r['selected']} | {r['threshold']:.3f} | {r['baseline']['accuracy']:.2%} | {r['challenger']['accuracy']:.2%} | {r['baseline']['recall']:.2%} | {r['challenger']['recall']:.2%} |"
        for r in result["folds"]
    )
    checks = []
    protocol = result["protocol"]
    for label, old, new in [
        ("Nested CV", nested_old, nested_new),
        ("Reused validation", val_old, val_new),
    ]:
        conditions = {
            "Accuracy gain at least 0.5 pp": new["accuracy"] - old["accuracy"]
            >= protocol["min_accuracy_gain"],
            "Recall at least 56%": new["recall"] >= protocol["recall_floor"],
            "Recall loss at most 3 pp": old["recall"] - new["recall"]
            <= protocol["maximum_recall_drop"],
            "Brier increase at most 0.005": new["brier"] - old["brier"]
            <= protocol["max_brier_increase"],
        }
        checks.extend(
            f"| {label} | {name} | {'Pass' if passed else 'Fail'} |"
            for name, passed in conditions.items()
        )
    checks.append(
        f"| Nested CV | Accuracy-delta interval lower bound above zero | {'Pass' if intervals['accuracy_delta'][0] > 0 else 'Fail'} |"
    )
    text = f"""# Accuracy improvement with recall constraints

## Decision
{decision}

The original model/report are preserved. Original test data were **not scored again**. No model change is hidden behind the old test report. The final candidate is stored only in ignored `artifacts/accuracy-recall-001/model.joblib`.

## Comparison on the same evaluation rows
| Evaluation | Baseline accuracy | Challenger accuracy | Change | Baseline recall | Challenger recall | Change |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Nested CV estimates the entire selection procedure: each outer fold may select a different model/threshold. The final all-training selection is **{result['metadata']['algorithm']}**, threshold **{result['metadata']['decision_threshold']:.3f}**. Validation numbers refer to that single fitted candidate and the exact saved incumbent.

Exploratory paired 95% group-bootstrap intervals on nested OOF differences:
- Accuracy change: {100*intervals['accuracy_delta'][0]:+.2f} to {100*intervals['accuracy_delta'][1]:+.2f} percentage points.
- Recall change: {100*intervals['recall_delta'][0]:+.2f} to {100*intervals['recall_delta'][1]:+.2f} percentage points.

Intervals condition on the fitted OOF predictions, ignore dependence from overlapping training sets, and do not capture retraining/model-search uncertainty. Do not interpret them as definitive population-level significance.

## Guardrail audit
| Evaluation | Prespecified requirement | Result |
|---|---|---|
{chr(10).join(checks)}

## Protocol fixed before fitting
- Original training partition only: 18,000 records. Three grouped outer folds, each with three grouped inner folds. Identical profiles/IDs cannot cross any fold.
- Eight fixed candidate configurations: incumbent XGBoost, three regularized XGBoost variants, two histogram gradient boosters, unweighted random forest and Extra Trees. All retain the shared raw-input preprocessing pipeline.
- No preprocessing is fitted outside its training fold. No synthetic oversampling. Histogram boosting internal early stopping is disabled, avoiding an ungrouped internal validation split.
- For each candidate, select a threshold from 0.000–1.000 in 0.005 steps on inner OOF predictions only. Maximize accuracy while recall is at least max(56%, baseline inner recall minus 3 percentage points). Ties favor higher recall.
- Select the candidate by inner constrained accuracy. Refit it on outer training and score outer validation exactly once. The incumbent architecture is also freshly fitted on each outer training set and retains threshold 0.28.
- After nested evaluation, run the same procedure on all original training rows. Freeze candidate and threshold before a single reused-validation audit; do not adapt them afterward.
- Promotion requires at least +0.5 percentage points accuracy, recall >=56%, recall loss <=3 percentage points and Brier deterioration <=0.005 on both nested CV and reused validation. Nested paired accuracy interval lower bound must exceed zero. These checks are conservative project criteria, not proof of generalization.

Protocol: [protocol.json](protocol.json). Complete aggregate results: [comparison.json](comparison.json). Individual OOF predictions, row indices and model bundles remain in ignored artifacts.

## Outer fold outcomes
| Fold | Inner-selected model | Threshold | Baseline accuracy | Challenger accuracy | Baseline recall | Challenger recall |
|---|---|---:|---:|---:|---:|---:|
{fold_rows}

## Final inner OOF selection (selection-biased; not test results)
| Candidate | Threshold | Accuracy | Recall | Average precision | Brier |
|---|---:|---:|---:|---:|---:|
{candidate_rows}

![Accuracy and recall](accuracy_recall.png)

## Interpretation and limitations
Raising a threshold often reduces false positives and increases accuracy, but can miss more defaults. A recall constraint on fitting/OOF data does **not** guarantee recall on other observations. The outer and validation checks detect this failure rather than relaxing the criteria after observing results.

The original training and validation data have already influenced earlier project choices. This nested comparison prevents direct fold leakage, but it is exploratory development evidence, not a fresh unbiased evaluation of the whole research process. Reused validation is not an independent holdout. No claim is made that the published 79.37% test accuracy has improved. A new temporal/geographic cohort is needed to establish a new test claim.

## Reproduce
With Python 3.12, the pinned requirements, original workbook and run-001 artifact:
```bash
python -m credit_risk.improve --output artifacts/accuracy-recall-new --report reports/accuracy-recall-new
python -m pytest -q
```
Use a new directory for exact reproduction; never run repeated searches against the observed test outcome. `scripts/report_improvement.py` renders this run's aggregate outputs without fitting anything.

## Method references
- [Scikit-learn: nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [Scikit-learn: decision-threshold tuning and overfitting](https://scikit-learn.org/stable/modules/classification_threshold.html).
"""
    if protocol.get("profile") == "native":
        text = text.replace(
            "artifacts/accuracy-recall-001/model.joblib",
            "artifacts/native-002/model.joblib",
        )
        text = text.replace(
            "Eight fixed candidate configurations: incumbent XGBoost, three regularized XGBoost variants, two histogram gradient boosters, unweighted random forest and Extra Trees. All retain the shared raw-input preprocessing pipeline.",
            "Four fixed configurations: incumbent XGBoost and native CatBoost at depths 4, 5 and 6. CatBoost receives categorical codes as stable strings and numeric missing values natively. Its target-dependent categorical statistics are fitted only inside training folds. The raw input and row-local feature contract is shared.",
        )
        text = text.replace(
            "max(56%, baseline inner recall minus 3 percentage points)",
            "max(60%, baseline inner recall minus 3 percentage points)",
        )
        text = text.replace(
            "--output artifacts/accuracy-recall-new --report reports/accuracy-recall-new",
            "--profile native --selection-recall-floor 0.60 --output artifacts/native-new --report reports/native-new",
        )
        text += "\nThis follow-up was designed after inspecting the earlier accuracy-recall-001 results. The higher inner recall target was fixed before this run. It does not undo adaptive development bias.\n\n[CatBoost native categorical features](https://catboost.ai/docs/en/features/categorical-features).\n"
    if protocol.get("profile") == "behavior":
        text = text.replace(
            "artifacts/accuracy-recall-001/model.joblib",
            "artifacts/behavior-003/model.joblib",
        )
        text = text.replace(
            "Eight fixed candidate configurations: incumbent XGBoost, three regularized XGBoost variants, two histogram gradient boosters, unweighted random forest and Extra Trees. All retain the shared raw-input preprocessing pipeline.",
            "Twelve fixed configurations: incumbent, two raw ordinal XGBoost controls, three XGBoost models with behavioral features, three LightGBM configurations, a histogram booster and two fixed soft-voting ensembles. Behavioral features describe historical payment coverage, limit-normalized balance flows, volatility, delay recency and consecutive delays. No IDs, target-derived features or population aggregates are used.",
        )
        text = text.replace(
            "max(56%, baseline inner recall minus 3 percentage points)",
            "max(58%, baseline inner recall minus 3 percentage points)",
        )
        text = text.replace(
            "--output artifacts/accuracy-recall-new --report reports/accuracy-recall-new",
            "--profile behavior --selection-recall-floor 0.58 --output artifacts/behavior-new --report reports/behavior-new",
        )
        text += "\nFeature chronology follows the [UCI variable definitions](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients). Ratios to a preceding bill are historical proxies, not contractual repayment amounts. [LightGBM API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).\n"
    (REPORT / "REPORT.md").write_text(text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, base, new, color in [
        ("Nested CV", nested_old, nested_new, "#286580"),
        ("Reused validation", val_old, val_new, "#a16a43"),
    ]:
        ax.plot(
            [base["recall"] * 100, new["recall"] * 100],
            [base["accuracy"] * 100, new["accuracy"] * 100],
            color=color,
            linewidth=1,
        )
        ax.scatter(
            base["recall"] * 100,
            base["accuracy"] * 100,
            color=color,
            marker="o",
            s=80,
            label=f"{name}: incumbent",
        )
        ax.scatter(
            new["recall"] * 100,
            new["accuracy"] * 100,
            color=color,
            marker="^",
            s=90,
            label=f"{name}: challenger",
        )
    ax.axvline(56, ls="--", color="gray", label="56% recall floor")
    ax.set_xlabel("Default recall (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy–recall trade-off: exploratory comparison")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(REPORT / "accuracy_recall.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT))
    main(parser.parse_args().report)
