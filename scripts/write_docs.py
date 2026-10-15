"""Build measured documentation tables from the immutable evaluation report."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
r = json.loads((ROOT / "reports/run-001/evaluation.json").read_text())
s = json.loads((ROOT / "reports/run-001/selection.json").read_text())
m, t = r["metadata"], r["test"]


def write(path, text):
    (ROOT / path).write_text(text.strip() + "\n", encoding="utf-8")


comparison = (
    "| Candidate | Training CV AP (mean ± SD) | Validation AP | Brier | Fit + CV seconds |\n|---|---:|---:|---:|---:|\n"
    + "\n".join(
        f"| {x['algorithm']} | {x['cv_mean']:.4f} ± {x['cv_std']:.4f} | {x['validation']['average_precision']:.4f} | {x['validation']['brier']:.4f} | {x['training_seconds_including_cv']:.1f} |"
        for x in s["models"]
    )
)
metrics = "| Metric | Held-out test | 95% group-bootstrap CI |\n|---|---:|---|\n" + "\n".join(
    f"| {k} | {v:.4f} | {r['confidence_intervals_95'][k][0]:.4f}–{r['confidence_intervals_95'][k][1]:.4f} |"
    for k, v in t.items()
    if k != "confusion_matrix"
)
calibration = (
    "| Method | Validation Brier (lower is better) |\n|---|---:|\n"
    + "\n".join(f"| {x['method']} | {x['brier']:.6f} |" for x in s["calibration"])
)
refs = """1. Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. [Dataset and dictionary](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), DOI: 10.24432/C55S3H.
2. Lundberg, S. M. and Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874).
3. Scikit-learn contributors. [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html), [group-aware cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#stratifiedgroupkfold), and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
4. SHAP contributors. [PermutationExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html).

Sources checked 15 October 2026. Online documentation tracks current releases; experiments use requirements.txt pins."""

write(
    "docs/EXPERIMENTS.md",
    f"""
# Experiments

## Protocol frozen before test evaluation
Seed 42; connected groups join identical 23-field profiles or repeated IDs. StratifiedGroupKFold creates 18,000 training, 6,004 validation and 5,996 test rows. Group constraints make prevalence approximate: training 21.81%, validation 23.23%, test 21.95%. Three group-aware training folds evaluate each pipeline; all fitted preprocessing remains inside each fold.

Bounded search: two logistic class-weight settings, two forest leaf sizes, two gradient-boosting depths, two XGBoost depths and a dummy prior. The original soft-voting idea is retained with the two validation-leading estimators. Its membership uses validation; its CV result is descriptive, not nested selection evidence. The original deep stacking configuration remains archived because its resampling/CV protocol leaked information and it was expensive; no gain under this corrected protocol has been established.

No SMOTE: category interpolation would be inappropriate; class weighting is directly compared. Structured JSON/CSV track full parameters, all fold scores, runtime, batch inference timing and compressed sizes. Timing is machine-specific and includes CV plus final fit, excluding calibration. See [selection.json](../reports/run-001/selection.json) and [comparison.csv](../reports/run-001/comparison.csv).

## Model comparison
{comparison}

Winner: `{m['algorithm']}` by validation AP. Its small advantage over voting is not a statistical-significance claim. AP emphasizes minority-class retrieval across thresholds; it is not trapezoidal PR-AUC. Accuracy alone can reward predicting no default.

## Calibration and threshold
{calibration}

Calibrators fit on training out-of-fold predictions using group-aware folds and ensemble=False; the base estimator is refit on all training data. Validation chooses minimum Brier. Original probabilities win; calibration was evaluated, **not demonstrated to improve this model**.

Thresholds 0.05–0.95 in increments of 0.01 were compared by validation F1; ties favor larger thresholds. Selected **0.28**. F1 is an educational precision/recall proxy, not a credit-loss optimum: no exposure, recovery or decision-cost data exists. Display bands [0.20, 0.50] are configurable at training and explicitly illustrative, independent of the classifier threshold.

## Final held-out results
{metrics}

Confusion matrix (truth rows, prediction columns, order 0/1): `{t['confusion_matrix']}`. At threshold 0.50 the frozen model has accuracy 82.39%, recall 36.25% and F1 0.4746. Threshold 0.28 catches 300 more defaults while flagging 481 more non-defaults. This is descriptive, not subsequent tuning.

Bootstrap resamples entire profile/ID groups 300 times with seed 42. Intervals condition on this fitted model and cohort; they omit training/selection uncertainty and future drift. Validation is reused for several choices and therefore is selection-biased. Test is opened after freezing choices. CLI refuses overwriting runs/final reports, but cannot prevent intentional repeated experiments in new directories.

## Error analysis and legacy comparison
False positives: 698, mean p=0.4502, median limit NT$80,000, median PAY_0 code 1. False negatives: 539, mean p=0.1495, median limit NT$130,000, median PAY_0 code 0. These are associations, not explanations of why a person defaulted. A hypothetical approval policy could incur default losses from false negatives and unfairly deny non-defaulting customers through false positives. This demo implements no lending policy.

Historical accuracy says 81.32%, but the artifact is missing and its test was used for selection. No controlled before/after gain can be claimed.
""",
)

write(
    "docs/TECHNICAL_REPORT.md",
    f"""
# Explainable AI-Based Credit Default Risk Prediction

## 1. Abstract
This study upgrades an existing Flask/React credit-default demo into a reproducible experiment. Correcting training/inference mismatch and test-driven selection is the central contribution. Ten configurations were compared on the available 30,000-row UCI cohort. Selected XGBoost depth 4 achieved held-out AP {t['average_precision']:.4f}, ROC-AUC {t['roc_auc']:.4f}, and Brier {t['brier']:.4f}. Probability-space SHAP explains the exact served model. This is an engineering study, not a novel algorithm or state-of-the-art claim.

## 2. Introduction
Accuracy is insufficient when default is a minority event and inference fails to reproduce training features. The original API replaced engineered features with zeros and training selected its winner using test accuracy. The upgrade prioritizes defensible evaluation, reproducibility and explicit limitations.

## 3. Problem definition
Estimate P(next-month default=1 | historical observations) from demographic, credit-limit and six-month repayment/statement/payment information. Classification at a validation-selected threshold is secondary to ranking and reliability. The system does not estimate causal intervention effects or decide eligibility.

## 4. Related methods
Logistic regression provides a linear baseline; forests and boosted trees capture nonlinear patterns; dummy prior prediction establishes a reference. Soft voting tests whether averaging adds value. Calibration and group CV follow documented sklearn methods [3]. SHAP supplies additive attributions [2,4]. No algorithmic novelty is asserted.

## 5. Dataset
The original workbook has 30,000 historical Taiwan records, 23 predictors and 6,636 defaults [1]. Local inspection finds no missing values, unique IDs and 56 duplicate profiles. Six-month histories are columns within one cohort; there are no independent outcome cohorts for chronological evaluation. Grouping protects identical profiles from crossing splits. Unknown family/account relationships remain unobserved. See [dataset card](DATASET_CARD.md).

## 6. Preprocessing
A shared sklearn pipeline computes eight conservative row-local summaries, imputes numeric medians/categorical modes, scales numeric values and one-hot encodes nominal demographic/status codes. Every fitted transformation stays inside training folds. Nonpositive ratio denominators become missing values; negative bills are retained. ID and target are excluded. Categorical SMOTE is avoided and weighting is compared empirically.

## 7. Model development
Ten configurations cover dummy, two logistic models, two forests, two gradient-boosting models, two XGBoost models and voting. Fixed seeds and bounded search limit expense. Legacy deep stacking is archived, not represented as validated. Artifacts retain the full pipeline, training background, threshold and dependency metadata.

## 8. Experimental methodology
18,000 / 6,004 / 5,996 group-aware training/validation/test records. Three training folds measure AP; validation AP selects the model. Sigmoid/isotonic fit on training out-of-fold predictions; validation Brier selects calibration. Validation F1 selects 0.28 from a predeclared grid. No refit on validation follows selection. Test is evaluated after freezing all choices. Reusing validation for several choices causes selection bias there, which the held-out check helps assess within this cohort.

## 9. Results
{comparison}

{calibration}

{metrics}

![ROC](../reports/run-001/roc.png)
![Precision recall](../reports/run-001/precision_recall.png)
![Calibration](../reports/run-001/calibration.png)

Original XGBoost probabilities minimize validation Brier; additional calibration did not help. The small AP advantage over voting is not tested for significance. Confusion matrix at 0.28: {t['confusion_matrix']}. At 0.50 accuracy is 82.39% but recall is 36.25%. The selected threshold retrieves more defaults at the cost of more false positives. Historical 81.32% accuracy is not a controlled comparator.

Intervals use 300 group-bootstrap replicates and omit training/selection variation. There are 698 false positives and 539 false negatives. Recorded-sex FPRs are 18.12% and 12.85%; the smaller 51+ age group has FPR 21.32%. Descriptive differences do not establish causal discrimination or significance. Aggregate errors and complete subgroup metrics are in evaluation.json.

## 10. Explainability analysis
Permutation SHAP calls the exact served probability function, including raw feature engineering and selected calibration. It uses 24 training-background rows, three permutations, and 30 sampled validation rows for global plots. PAY_0 has the largest sampled mean absolute contribution. Values are probabilities, not log odds. Maximum additivity error is {r['shap']['max_additivity_error']:.2e}. This consistency check does not prove attribution stability or causality.

![SHAP summary](../reports/run-001/shap_beeswarm.png)

## 11. Limitations and ethics
One historical geographic cohort cannot establish contemporary validity. Demographic features may encode disadvantage. Group uncertainty, intersectional studies and sensitive-feature ablations remain future work. F1 omits lending costs. Correlated histories, marginal masking and a small SHAP background limit attribution interpretation. Reports publish no individual records; model bundles contain a background sample and are data-bearing. The prototype must not be the sole basis for lending. Docker configuration is provided but not executed here; no cloud deployment is claimed.

## 12. Conclusion
The contribution is consistent train/serve behavior, leakage-aware comparisons, calibration/threshold evaluation, uncertainty and SHAP in an integrated dashboard. Stronger evidence requires new temporal/geographic cohorts, preregistered operating costs and attribution/fairness studies.

## 13. References
{refs}
""",
)

write(
    "docs/RESUME_BULLETS.md",
    f"""
# Verified resume bullets

- Refactored a credit-default application into shared scikit-learn pipelines and compared 10 configurations on 30,000 UCI records using group-aware splits and training-only cross-validation.
- Evaluated XGBoost with held-out ROC-AUC {t['roc_auc']:.3f} and average precision {t['average_precision']:.3f}; implemented calibration comparisons, validation threshold selection, bootstrap intervals and probability-space SHAP.
- Integrated a React research dashboard with validated Flask prediction/explanation APIs, artifact integrity checks and automated Python/browser tests; added pinned dependencies, CI and Docker configuration.

Do not claim verified cloud deployment, executed GitHub CI, a tested Docker image, a controlled gain over the original model, or a novel algorithm.
""",
)

write(
    "README.md",
    f"""
# Explainable AI-Based Credit Default Risk Prediction

An upgrade of [Sathwik's original application](https://github.com/singusathwik/Credit-Card-Risk-Prediction), retaining Flask and React while correcting evaluation and inference methodology.

## Current results and practical interpretation

The application still serves the original **XGBoost** model. The latest trained **LightGBM candidate** adds repayment-behavior features and uses all 24,004 non-test records for development. Both results below refer to the same 5,996 test records.

| Metric | Active XGBoost | Latest LightGBM candidate |
|---|---:|---:|
| Accuracy | 79.37% | **79.47%** |
| Default recall | **59.04%** | 58.51% |
| Precision | 52.68% | **52.92%** |
| ROC AUC | 0.7930 | **0.7936** |
| Average precision | 0.5708 | **0.5796** |

**How good are these results?** They support an educational or portfolio demonstration of reproducible modeling and explanations, but do not establish readiness for real lending or state-of-the-art predictive performance. Predicting “no default” for every test record would achieve **78.05% accuracy while catching no defaults**, so accuracy alone is insufficient. The candidate catches about **59 of every 100 actual defaults**, misses about **41**, and approximately **53% of its flagged clients actually default**.

The candidate's **+0.10 percentage-point accuracy gain** costs **0.53 points of recall**: 13 fewer false alarms, 7 more missed defaults, and 6 additional correct classifications. Its paired 95% accuracy-change interval is **−0.37 to +0.55 percentage points**, which includes zero. It did not meet the predefined +0.5-point adoption margin, so it has **not replaced the active model**.

The candidate and threshold were frozen before this audit; test records were excluded from fitting and selection. However, the benchmark's original result was already known, so this is **not fresh external validation**. [Full candidate report and reproducibility evidence](reports/full-development-004/REPORT.md).

## Overview and problem statement
Estimate next-month default probability from historical credit-card records and inspect the evidence. This project demonstrates reproducible comparisons, calibration assessment, threshold trade-offs and SHAP. It is an **educational research prototype**, not a lending decision system. The principal gain is methodological correctness; no controlled legacy accuracy improvement or state-of-the-art claim is made.

## Dataset and target
30,000 Taiwan customers, 23 predictors, 6,636 defaults (22.12%). Target: `default payment next month` (1=default). Histories cover April–September 2005; ID is excluded. The 56 repeated predictor profiles stay grouped across partitions. [Dataset card and acquisition](docs/DATASET_CARD.md) · [UCI source](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

## Architecture
```text
UCI workbook -> grouped train/validation/test -> fold-local sklearn pipelines
 -> validation selection + calibration + threshold -> full saved pipeline
 -> Flask predict / explain APIs -> React dashboard
 -> frozen test evaluation -> JSON / CSV / charts / model card
```
[Architecture and features](docs/ARCHITECTURE.md) · [Baseline audit](docs/BASELINE_AUDIT.md).

## Active model methodology
- 18,000 training / 6,004 validation / 5,996 test rows; seed 42, profile/ID groups.
- Three-fold group-aware training CV; fitted preprocessing stays in each fold.
- Validation average precision selects the model; training-CV calibration is selected by validation Brier; validation F1 selects threshold 0.28.
- One-hot nominal/status codes, train-fitted imputation/scaling and eight historical summaries. Class weighting is compared; no categorical SMOTE.
- Test scores do not choose the model, calibration or threshold. Structured JSON/CSV provide experiment tracking.

## Original model comparison
{comparison}

Selected **XGBoost depth 4**. Voting offered no validation improvement. Sigmoid/isotonic were compared; original probabilities had the lowest validation Brier. Calibration was evaluated, not assumed to help. [Full experiments](docs/EXPERIMENTS.md).

## Explainable AI
`POST /explain` returns SHAP contributions to the exact served default probability. Positive/negative values increase/decrease the estimate relative to 24 training-background rows. UI shows percentage points, not log odds. Global plots use 30 validation observations. Additivity is checked; explanations remain approximate and non-causal. [Model card](docs/MODEL_CARD.md).

## Active model evaluation results
{metrics}

Threshold 0.28; confusion matrix `{t['confusion_matrix']}` (truth rows, predicted columns). At 0.50 accuracy is 82.39% with recall 36.25%; the selected threshold is a validation F1 compromise. Historical 81.32% lacks its model artifact and used test-based selection, so it is not directly comparable.

[Evaluation](reports/run-001/EVALUATION.md) · [JSON results](reports/run-001/evaluation.json) · [Technical report](docs/TECHNICAL_REPORT.md).

## Further accuracy–recall research
The [behavioral-feature and full-development follow-up](reports/full-development-004/REPORT.md) produced a LightGBM candidate with **79.47% accuracy / 58.51% default recall**, versus the incumbent's **79.37% / 59.04%** on the same 5,996 records. The +0.10-point accuracy gain is small and its paired interval includes zero. This candidate uses new repayment-history features and 24,004 development records; its model/threshold were frozen before a single audit on the previously reported test benchmark. It did not meet the predefined +0.5-point adoption margin, so the default remains the incumbent. The trained candidate is available locally at `artifacts/full-development-004/serving.joblib`; model files are not published.

**Earlier two follow-ups:** neither met the replacement criteria. Native categorical boosting with a stricter recall target achieved 77.91% accuracy / 60.43% recall in nested CV versus the incumbent's 78.74% / 58.57% on the same folds.

The main dashboard now displays held-out accuracy and default recall directly. The evaluation page explains how many defaults are caught or missed, alongside uncertainty intervals and the full technical metrics.

The [native CatBoost follow-up](reports/native-002/REPORT.md) compares three categorical boosting configurations against the incumbent with a stricter training recall target. It is exploratory research, not a replacement test score. Run it with `python -m credit_risk.improve --profile native --selection-recall-floor 0.60 --output artifacts/native-new --report reports/native-new`.

A bounded nested, group-aware CV follow-up evaluates accuracy subject to explicit default-recall constraints. The original test partition is excluded from that search. See the [follow-up report](reports/accuracy-recall-001/REPORT.md) for the comparison, fixed protocol and replacement decision. These exploratory results do not replace the original held-out metrics above.

## Application screenshots
![Prediction and local SHAP](docs/screenshots/prediction-desktop.png)
![Evaluation dashboard](docs/screenshots/evaluation-desktop.png)
[Mobile view](docs/screenshots/prediction-mobile.png).

## Installation
Use **Python 3.12** and Node.js 22. Python 3.14 is unsupported by these pins because SHAP 0.50 specifies beta-only numba dependencies there.

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
npm ci
```
On Linux/macOS use `source .venv/bin/activate`. `requirements-lock.txt` additionally records the full clean Python 3.12 dependency snapshot. If absent, obtain the original workbook following the dataset card; xlrd is pinned. Do not substitute data. Export settings from `.env.example` in your shell; this application does not automatically load dotenv files.

## Training and evaluation
```bash
python train_model.py --output artifacts/run-001
python -m credit_risk.evaluate --run artifacts/run-001 --report reports/run-001
```
Existing runs and final reports are protected from overwrite. This checkout already has frozen `reports/run-001`. To reproduce, use `--output artifacts/reproduction` and then `--run artifacts/reproduction --report reports/reproduction`; set MODEL_PATH and REPORT_DIR to those paths to serve the reproduction. Do not tune against an observed test set; future optimization requires a new external holdout. Bundles include the full pipeline and background data and are ignored by Git.

## API usage
Run `python server.py`. Endpoints (also with `/api` prefix): `GET /health`, `GET /model-info`, `POST /predict`, `POST /explain`. Additional `/api/evaluation`, `/api/plots/<allowlisted-name>` and legacy `/accuracy`/`/api/accuracy` remain.

```bash
curl -H "Content-Type: application/json" --data-binary @examples/applicant.json http://127.0.0.1:5000/predict
curl -H "Content-Type: application/json" --data-binary @examples/applicant.json http://127.0.0.1:5000/explain
```
Use `curl.exe` if PowerShell aliases curl. Supply exactly 23 raw fields. Invalid/missing/non-finite values return 400, wrong media type 415, unavailable model 503, busy explanations 429. Responses include probability, label, threshold, illustrative bands and version. No applicant records are stored by application code.

## Running the frontend
```bash
npm run build
python server.py
```
Open http://127.0.0.1:5000. Waitress serves React and APIs. For development run `npm run dev` in another terminal and open http://localhost:3000; Vite proxies `/api` to Flask. Debug is disabled; source/data files are not static assets. CORS defaults to same-origin.

## Tests
```bash
python -m pytest -q
npm run build
npx playwright test
```
Browser tests require the trained model, frozen report, build and running server. They use installed Edge by default; set PLAYWRIGHT_CHANNEL=chromium and run `npx playwright install chromium` elsewhere. Tests exercise real predictions, SHAP reconstruction, evaluation plots, invalid-input/error states and a 390px viewport. CI runs synthetic Python tests/build without data or a fitted artifact. [Verification record](docs/VERIFICATION.md).

## Docker setup
```bash
docker build -t credit-risk .
docker run --rm -p 5000:5000 -v "$PWD/artifacts:/app/artifacts:ro" credit-risk
```
Train first with the pinned environment. The bundle is excluded from the image and mounted read-only. Container user is non-root. **Docker execution was not verified: Docker is unavailable here.** No cloud deployment was performed.

## Limitations
Historical single-country data, unmeasured drift, imperfect probabilities, no lending-cost optimum, descriptive subgroups without subgroup intervals and small SHAP samples limit conclusions. Manifests do not make untrusted pickle safe. No authentication or durable rate limiting is provided. See model and dataset cards.

## Responsible AI
Use synthetic inputs and never use this as the sole basis for real lending. Demographic fields may encode disadvantage. SHAP explains model associations, not causes; risk bands are illustrative, not policy recommendations. Evaluate external validity and relevant governance before any real-world use.

## Future research and portfolio use
Preregister cost-based objectives; evaluate new temporal/geographic cohorts; study sensitive-feature ablations, subgroup uncertainty and SHAP stability. For Master's applications, explain the leakage/inference corrections and threshold trade-offs, link the report and demo, and present this as applied ML engineering. [Verified resume bullets](docs/RESUME_BULLETS.md).

## Attribution and license
Original application: linked repository owner. Historical code/README are retained in `legacy/`; original notebook remains historical. Dataset: I-Cheng Yeh/UCI, CC BY 4.0. No explicit code license was present and this upgrade does not invent one; the owner should select one before redistribution. Model artifacts and local environments are excluded from Git; published reports contain aggregate results only.
""",
)

for name in ["HOW_TO_RUN.md", "QUICKSTART.md", "CHECK_STATUS.md", "TROUBLESHOOTING.md"]:
    write(
        name,
        """# Current setup and troubleshooting

See [README](README.md) for verified setup, training, APIs and frontend commands.
GET /health returns 200 only when a compatible model is loaded. If unavailable,
train in the pinned environment and check MODEL_PATH. Build with npm run build.
Existing reports are immutable; use a new run/report directory for reproduction.
A fixed accuracy value does not prove server health.""",
    )
