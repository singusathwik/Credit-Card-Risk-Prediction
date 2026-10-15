# Experiments

## Protocol frozen before test evaluation
Seed 42; connected groups join identical 23-field profiles or repeated IDs. StratifiedGroupKFold creates 18,000 training, 6,004 validation and 5,996 test rows. Group constraints make prevalence approximate: training 21.81%, validation 23.23%, test 21.95%. Three group-aware training folds evaluate each pipeline; all fitted preprocessing remains inside each fold.

Bounded search: two logistic class-weight settings, two forest leaf sizes, two gradient-boosting depths, two XGBoost depths and a dummy prior. The original soft-voting idea is retained with the two validation-leading estimators. Its membership uses validation; its CV result is descriptive, not nested selection evidence. The original deep stacking configuration remains archived because its resampling/CV protocol leaked information and it was expensive; no gain under this corrected protocol has been established.

No SMOTE: category interpolation would be inappropriate; class weighting is directly compared. Structured JSON/CSV track full parameters, all fold scores, runtime, batch inference timing and compressed sizes. Timing is machine-specific and includes CV plus final fit, excluding calibration. See [selection.json](../reports/run-001/selection.json) and [comparison.csv](../reports/run-001/comparison.csv).

## Model comparison
| Candidate | Training CV AP (mean ± SD) | Validation AP | Brier | Fit + CV seconds |
|---|---:|---:|---:|---:|
| dummy | 0.2181 ± 0.0023 | 0.2323 | 0.1786 | 0.6 |
| logistic_None | 0.5315 ± 0.0167 | 0.5480 | 0.1401 | 1.6 |
| logistic_balanced | 0.5289 ± 0.0179 | 0.5469 | 0.1845 | 1.8 |
| random_forest_leaf10 | 0.5460 ± 0.0210 | 0.5695 | 0.1646 | 10.3 |
| random_forest_leaf30 | 0.5447 ± 0.0232 | 0.5690 | 0.1742 | 9.2 |
| gradient_boosting_depth2 | 0.5434 ± 0.0192 | 0.5660 | 0.1391 | 37.8 |
| gradient_boosting_depth3 | 0.5504 ± 0.0160 | 0.5722 | 0.1384 | 46.2 |
| xgboost_depth2 | 0.5446 ± 0.0186 | 0.5658 | 0.1388 | 2.2 |
| xgboost_depth4 | 0.5510 ± 0.0159 | 0.5747 | 0.1380 | 2.4 |
| soft_voting | 0.5522 ± 0.0163 | 0.5744 | 0.1381 | 49.6 |

Winner: `xgboost_depth4` by validation AP. Its small advantage over voting is not a statistical-significance claim. AP emphasizes minority-class retrieval across thresholds; it is not trapezoidal PR-AUC. Accuracy alone can reward predicting no default.

## Calibration and threshold
| Method | Validation Brier (lower is better) |
|---|---:|
| uncalibrated | 0.138042 |
| sigmoid | 0.138334 |
| isotonic | 0.138276 |

Calibrators fit on training out-of-fold predictions using group-aware folds and ensemble=False; the base estimator is refit on all training data. Validation chooses minimum Brier. Original probabilities win; calibration was evaluated, **not demonstrated to improve this model**.

Thresholds 0.05–0.95 in increments of 0.01 were compared by validation F1; ties favor larger thresholds. Selected **0.28**. F1 is an educational precision/recall proxy, not a credit-loss optimum: no exposure, recovery or decision-cost data exists. Display bands [0.20, 0.50] are configurable at training and explicitly illustrative, independent of the classifier threshold.

## Final held-out results
| Metric | Held-out test | 95% group-bootstrap CI |
|---|---:|---|
| accuracy | 0.7937 | 0.7835–0.8046 |
| precision | 0.5268 | 0.5035–0.5546 |
| recall | 0.5904 | 0.5660–0.6142 |
| f1 | 0.5568 | 0.5362–0.5797 |
| macro_f1 | 0.7112 | 0.6996–0.7249 |
| roc_auc | 0.7930 | 0.7803–0.8076 |
| average_precision | 0.5708 | 0.5416–0.6013 |
| balanced_accuracy | 0.7206 | 0.7082–0.7338 |
| brier | 0.1309 | 0.1255–0.1361 |
| false_positive_rate | 0.1491 | 0.1385–0.1592 |
| false_negative_rate | 0.4096 | 0.3858–0.4340 |

Confusion matrix (truth rows, prediction columns, order 0/1): `[[3982, 698], [539, 777]]`. At threshold 0.50 the frozen model has accuracy 82.39%, recall 36.25% and F1 0.4746. Threshold 0.28 catches 300 more defaults while flagging 481 more non-defaults. This is descriptive, not subsequent tuning.

Bootstrap resamples entire profile/ID groups 300 times with seed 42. Intervals condition on this fitted model and cohort; they omit training/selection uncertainty and future drift. Validation is reused for several choices and therefore is selection-biased. Test is opened after freezing choices. CLI refuses overwriting runs/final reports, but cannot prevent intentional repeated experiments in new directories.

## Error analysis and legacy comparison
False positives: 698, mean p=0.4502, median limit NT$80,000, median PAY_0 code 1. False negatives: 539, mean p=0.1495, median limit NT$130,000, median PAY_0 code 0. These are associations, not explanations of why a person defaulted. A hypothetical approval policy could incur default losses from false negatives and unfairly deny non-defaulting customers through false positives. This demo implements no lending policy.

Historical accuracy says 81.32%, but the artifact is missing and its test was used for selection. No controlled before/after gain can be claimed.
