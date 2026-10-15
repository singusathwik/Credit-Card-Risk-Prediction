# Model card

## Identity and intended use
Version: `uci-f2649c89-seed42-xgboost_depth4-uncalibrated`. Educational research/portfolio demonstration on a historical Taiwan cohort. Not a bank scoring service or causal model. Unsupported uses include autonomous approvals/denials, sole-basis lending decisions, and contemporary/international deployment without external validation.

## Training and performance
23 raw inputs plus eight deterministic summaries; one-hot nominal/status codes; training-fitted imputation/scaling. Profile/ID groups separate train/validation/test and three CV folds. Validation AP selects the algorithm, validation Brier selects calibration, validation F1 selects threshold 0.28. XGBoost: depth 4, 150 trees, learning rate 0.05; full settings in selection.json.

Held-out ROC-AUC **0.7930**, AP **0.5708**, Brier **0.1309**, recall **0.5904**, F1 **0.5568**, accuracy **0.7937**. [All metrics and intervals](../reports/run-001/EVALUATION.md). Sigmoid/isotonic were evaluated but not selected. Probability estimates remain imperfect.

## Bias and subgroup audit
Recorded SEX, AGE, EDUCATION and MARRIAGE remain predictors for historical replication and may encode disadvantage. No sensitive characteristics are inferred. Sex code 1: FPR 18.12%, FNR 39.12%, n=2,420; code 2: FPR 12.85%, FNR 42.45%, n=3,576. The 51+ group has n=457 and FPR 21.32%. Group Brier and mean predicted/observed rates are reported. These descriptive differences neither establish discrimination nor certify fairness. Subgroup intervals, intersectional studies and feature-removal ablations remain future work. Removing demographics alone would not remove correlated proxies.

## Explainability
Permutation SHAP calls the exact saved class-1 probability function, including any selected calibration. Masking raw fields before feature engineering preserves consistency. Values are probabilities (UI percentage points), not log odds. Three permutations and 24 training-background rows bound cost; global plots summarize 30 validation records. Maximum measured reconstruction error: 1.11e-16. This does not prove attribution stability. Correlation, marginal masking and background choice affect allocations; explanations are non-causal. PAY_0 leads sampled mean absolute contributions.

## Privacy and security
Application code does not log request bodies or applicant inputs. Use synthetic records. Bundles contain background data and must remain trusted/local. SHA-256 detects corruption, not a maliciously replaced pickle; package/schema checks precede deserialization. No API accepts executable model uploads. Repository source/data are not served, debug is off, CORS defaults to same-origin and Waitress serves requests.

No authentication, durable rate limit, encrypted transport or production security certification is supplied. SHAP concurrency is limited per process only. Docker and CI configurations exist, but Docker execution, GitHub CI execution and cloud deployment are unverified. Historical single-cohort performance does not establish real-world suitability.
