"""Publish frozen audit results and verify serving; never rescore test rows."""

import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from credit_risk.artifacts import load_bundle, save_bundle, versions
from credit_risk.features import validate_payload
from server import create_app


def main():
    report, artifact = Path("reports/full-development-004"), Path(
        "artifacts/full-development-004"
    )
    audit = json.loads((report / "audit.json").read_text())
    frozen = json.loads((report / "frozen_selection.json").read_text())
    protocol = json.loads((report / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        assert (
            hashlib.sha256((report / "frozen-source" / name).read_bytes()).hexdigest()
            == expected
        )
    assert (
        hashlib.sha256((artifact / "model.joblib").read_bytes()).hexdigest()
        == frozen["model_sha256"]
    )
    bundle = load_bundle(artifact / "model.joblib")
    development = np.load(artifact / "development.npz")
    split = np.load(artifact / "splits.npz")
    assert np.array_equal(development["row_indices"], split["train"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(development["groups"]) & set(split["groups"][split["test"]])
    # Correct inherited descriptive metadata without touching the frozen model,
    # its threshold, or its audit. The separate serving copy uses identical weights.
    serving = copy.deepcopy(bundle)
    serving["metadata"]["split_prevalence"] = dict(
        train=float(development["y"].mean()),
        validation=None,
        test=bundle["metadata"]["split_prevalence"]["test"],
    )
    serving["metadata"]["versions"] = versions(["lightgbm"])
    serving["metadata"]["frozen_model_sha256"] = frozen["model_sha256"]
    save_bundle(serving, artifact / "serving.joblib")
    serving = load_bundle(artifact / "serving.joblib")
    payload = json.loads(Path("examples/applicant.json").read_text())
    X = validate_payload(payload)
    probability = float(bundle["model"].predict_proba(X)[0, 1])
    assert float(serving["model"].predict_proba(X)[0, 1]) == probability
    client = create_app(bundle=serving, report_path=report).test_client()
    prediction = client.post("/api/predict", json=payload)
    explanation = client.post("/api/explain", json=payload)
    assert prediction.status_code == explanation.status_code == 200
    assert abs(prediction.json["default_probability"] - probability) < 1e-10
    assert (
        abs(explanation.json["explanation"]["reconstructed_probability"] - probability)
        < 1e-6
    )
    verification = dict(
        api_probability_parity=True,
        shap_reconstruction=True,
        test_rows_excluded_from_development=True,
        groups_disjoint=True,
        frozen_model_sha256=frozen["model_sha256"],
        frozen_sources_verified=True,
        serving_metadata_correction="Development prevalence and dependency list only; weights and threshold unchanged.",
        example_probability=probability,
        serving_sha256=hashlib.sha256(
            (artifact / "serving.joblib").read_bytes()
        ).hexdigest(),
    )
    (report / "verification.json").write_text(json.dumps(verification, indent=2))
    model = serving["model"]["model"]
    importance = sorted(
        zip(
            model.feature_name_,
            model.booster_.feature_importance(importance_type="gain"),
        ),
        key=lambda p: p[1],
        reverse=True,
    )
    (report / "feature_importance.json").write_text(
        json.dumps(
            [dict(feature=k, training_gain=float(v)) for k, v in importance], indent=2
        )
    )
    base, new = audit["baseline"], audit["test"]
    rows = "\n".join(
        f"| {name} | {base[key]:.2%} | {new[key]:.2%} | {100*(new[key]-base[key]):+.2f} pp |"
        for name, key in [
            ("Accuracy", "accuracy"),
            ("Default recall", "recall"),
            ("Precision", "precision"),
            ("F1", "f1"),
            ("ROC AUC", "roc_auc"),
            ("Average precision", "average_precision"),
        ]
    )
    candidates = json.loads((report / "selection.json").read_text())["models"]
    candidate_rows = "\n".join(
        f"| {r['algorithm']} | {r['threshold']:.3f} | {r['metrics']['accuracy']:.2%} | {r['metrics']['recall']:.2%} |"
        for r in candidates
    )
    lower, upper = audit["paired_intervals_95"]["accuracy_delta"]
    text = f"""# Behavioral LightGBM: frozen final comparison

The candidate achieved **{new['accuracy']:.2%} accuracy**, compared with **{base['accuracy']:.2%}** for the incumbent. Recall changed from **{base['recall']:.2%} to {new['recall']:.2%}**. This is a small observed improvement, not a statistically established gain. The paired fixed-prediction 95% interval for accuracy change is **{100*lower:+.2f} to {100*upper:+.2f} percentage points**.

The candidate did not meet the prespecified +0.5-point adoption margin, so the application's default remains the incumbent. The trained candidate is available locally in ignored `artifacts/full-development-004/serving.joblib`.

## Same 5,996 benchmark records
| Metric | Incumbent | Candidate | Change |
|---|---:|---:|---:|
{rows}

False alarms fell from **698 to 685**; missed defaults increased from **539 to 546**. That gives **6 more correct classifications**, with **13 fewer false alarms** and **7 more missed defaults**. Brier score improved from {base['brier']:.5f} to {new['brier']:.5f}.

## What changed
- The model is LightGBM with seven leaves per tree, 350 trees, regularization and a frozen **{serving['metadata']['decision_threshold']:.3f}** threshold.
- Features add payment-to-prior-bill ratios, limit-normalized balances/payments/flows, recent-versus-older trends, payment variability, delay severity/recency and consecutive delays. All are row-local and use only the recorded six-month history. ID and the target are never predictors. Ratios are historical proxies, not contractual due amounts.
- All **24,004 non-test records** enter development and the final fit. The former validation partition is now training data; there is no separate 6,004-record validation claim for this model.
- Twelve fixed candidates were compared using five-fold grouped OOF predictions. Every prediction used for selection came from a model that did not train on that record or its duplicate/ID group. The threshold maximized OOF accuracy subject to a 58% recall floor (or baseline recall minus three points if larger).
- Candidate, threshold, protocol and model hash were saved **before** test scoring. A one-shot audit guard rejects repeat audits. The model was not changed after the audit. A separate serving copy corrects inherited prevalence/dependency metadata; the frozen fitted bundle and audit remain immutable.

## Limits
The original test benchmark's baseline result had already been reported, and earlier development experiments influenced this work. No test observations entered fitting or model/threshold selection, but the benchmark is not fresh external evidence. Bootstrap intervals exclude training/search uncertainty and benchmark reuse bias. Further tuning must not use these test outcomes. A new cohort is needed for independent confirmation.

The change also combines feature engineering, a different estimator and more training records. This comparison does not isolate the causal contribution of each change. Training gain importance is descriptive, not causal.

## Development selection (selection-biased, not test results)
| Candidate | Threshold | OOF accuracy | OOF recall |
|---|---:|---:|---:|
{candidate_rows}

## Reproduce
```powershell
.venv-clean\\Scripts\\python.exe -m credit_risk.finalize train --output artifacts/full-new --report reports/full-new
.venv-clean\\Scripts\\python.exe -m credit_risk.finalize audit --output artifacts/full-new --report reports/full-new
```
Use a new directory. Do not rerun the audit as a tuning loop. The committed frozen sources preserve the exact audited implementation; current training code additionally corrects descriptive metadata before saving.

Evidence: [protocol](protocol.json), [frozen choice](frozen_selection.json), [audit](audit.json), [verification](verification.json), [training feature importance](feature_importance.json), [earlier nested feature experiment](../behavior-003/REPORT.md).

References: [UCI variable chronology](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [LightGBM 4.6 API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html).

![Frozen comparison](comparison.png)
"""
    (report / "REPORT.md").write_text(text, encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key, label in zip(
        axes, ["accuracy", "recall"], ["Accuracy", "Default recall"]
    ):
        values = [100 * base[key], 100 * new[key]]
        ax.bar(["Incumbent", "Candidate"], values, color=["#657986", "#286580"])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percent")
        ax.set_title(label)
        for i, value in enumerate(values):
            ax.text(i, value + 1, f"{value:.2f}%", ha="center")
    fig.suptitle("Frozen comparison on the previously reported test benchmark")
    fig.tight_layout()
    fig.savefig(report / "comparison.png", dpi=150)
    plt.close(fig)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
