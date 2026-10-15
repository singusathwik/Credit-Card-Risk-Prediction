# Frozen-model test evaluation

Model: `uci-f2649c89-seed42-xgboost_depth4-uncalibrated`

Threshold: 0.28; chosen on validation F1.

| Metric | Test value | 95% cluster bootstrap interval |
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

Confusion matrix (true rows, predicted columns): `[[3982, 698], [539, 777]]`.

Intervals condition on this fitted model and cohort. They do not measure population drift or training uncertainty.
Error aggregates, subgroup metrics and calibration bins: `evaluation.json`. No applicant rows are published.