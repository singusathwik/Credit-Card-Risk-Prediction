# Accuracy improvement with recall constraints

## Decision
The challenger did not pass all prespecified guardrails. The existing application model is retained; an accuracy increase alone is insufficient to justify replacing it.

The original model/report are preserved. Original test data were **not scored again**. No model change is hidden behind the old test report. The final candidate is stored only in ignored `artifacts/native-002/model.joblib`.

## Comparison on the same evaluation rows
| Evaluation | Baseline accuracy | Challenger accuracy | Change | Baseline recall | Challenger recall | Change |
|---|---:|---:|---:|---:|---:|---:|
| Nested grouped CV (18,000 training records) | 78.74% | 77.91% | -0.83 pp | 58.57% | 60.43% | +1.86 pp |
| Reused validation audit (6,004 records) | 78.60% | 78.01% | -0.58 pp | 56.70% | 57.85% | +1.15 pp |

Nested CV estimates the entire selection procedure: each outer fold may select a different model/threshold. The final all-training selection is **incumbent_xgb**, threshold **0.265**. Validation numbers refer to that single fitted candidate and the exact saved incumbent.

Exploratory paired 95% group-bootstrap intervals on nested OOF differences:
- Accuracy change: -1.04 to -0.62 percentage points.
- Recall change: +1.43 to +2.33 percentage points.

Intervals condition on the fitted OOF predictions, ignore dependence from overlapping training sets, and do not capture retraining/model-search uncertainty. Do not interpret them as definitive population-level significance.

## Guardrail audit
| Evaluation | Prespecified requirement | Result |
|---|---|---|
| Nested CV | Accuracy gain at least 0.5 pp | Fail |
| Nested CV | Recall at least 56% | Pass |
| Nested CV | Recall loss at most 3 pp | Pass |
| Nested CV | Brier increase at most 0.005 | Pass |
| Reused validation | Accuracy gain at least 0.5 pp | Fail |
| Reused validation | Recall at least 56% | Pass |
| Reused validation | Recall loss at most 3 pp | Pass |
| Reused validation | Brier increase at most 0.005 | Pass |
| Nested CV | Accuracy-delta interval lower bound above zero | Fail |

## Protocol fixed before fitting
- Original training partition only: 18,000 records. Three grouped outer folds, each with three grouped inner folds. Identical profiles/IDs cannot cross any fold.
- Four fixed configurations: incumbent XGBoost and native CatBoost at depths 4, 5 and 6. CatBoost receives categorical codes as stable strings and numeric missing values natively. Its target-dependent categorical statistics are fitted only inside training folds. The raw input and row-local feature contract is shared.
- No preprocessing is fitted outside its training fold. No synthetic oversampling. Histogram boosting internal early stopping is disabled, avoiding an ungrouped internal validation split.
- For each candidate, select a threshold from 0.000–1.000 in 0.005 steps on inner OOF predictions only. Maximize accuracy while recall is at least max(60%, baseline inner recall minus 3 percentage points). Ties favor higher recall.
- Select the candidate by inner constrained accuracy. Refit it on outer training and score outer validation exactly once. The incumbent architecture is also freshly fitted on each outer training set and retains threshold 0.28.
- After nested evaluation, run the same procedure on all original training rows. Freeze candidate and threshold before a single reused-validation audit; do not adapt them afterward.
- Promotion requires at least +0.5 percentage points accuracy, recall >=56%, recall loss <=3 percentage points and Brier deterioration <=0.005 on both nested CV and reused validation. Nested paired accuracy interval lower bound must exceed zero. These checks are conservative project criteria, not proof of generalization.

Protocol: [protocol.json](protocol.json). Complete aggregate results: [comparison.json](comparison.json). Individual OOF predictions, row indices and model bundles remain in ignored artifacts.

## Outer fold outcomes
| Fold | Inner-selected model | Threshold | Baseline accuracy | Challenger accuracy | Baseline recall | Challenger recall |
|---|---|---:|---:|---:|---:|---:|
| 1 | incumbent_xgb | 0.270 | 78.53% | 77.66% | 57.62% | 58.54% |
| 2 | incumbent_xgb | 0.265 | 78.46% | 77.76% | 59.21% | 60.35% |
| 3 | catboost_depth4 | 0.245 | 79.23% | 78.31% | 58.89% | 62.40% |

## Final inner OOF selection (selection-biased; not test results)
| Candidate | Threshold | Accuracy | Recall | Average precision | Brier |
|---|---:|---:|---:|---:|---:|
| incumbent_xgb | 0.265 | 78.29% | 60.10% | 0.5512 | 0.1328 |
| catboost_depth4 | 0.255 | 78.14% | 60.23% | 0.5484 | 0.1329 |
| catboost_depth5 | 0.255 | 77.86% | 60.00% | 0.5479 | 0.1329 |
| catboost_depth6 | 0.255 | 78.06% | 60.20% | 0.5483 | 0.1330 |

![Accuracy and recall](accuracy_recall.png)

## Interpretation and limitations
Raising a threshold often reduces false positives and increases accuracy, but can miss more defaults. A recall constraint on fitting/OOF data does **not** guarantee recall on other observations. The outer and validation checks detect this failure rather than relaxing the criteria after observing results.

The original training and validation data have already influenced earlier project choices. This nested comparison prevents direct fold leakage, but it is exploratory development evidence, not a fresh unbiased evaluation of the whole research process. Reused validation is not an independent holdout. No claim is made that the published 79.37% test accuracy has improved. A new temporal/geographic cohort is needed to establish a new test claim.

## Reproduce
With Python 3.12, the pinned requirements, original workbook and run-001 artifact:
```bash
python -m credit_risk.improve --profile native --selection-recall-floor 0.60 --output artifacts/native-new --report reports/native-new
python -m pytest -q
```
Use a new directory for exact reproduction; never run repeated searches against the observed test outcome. `scripts/report_improvement.py` renders this run's aggregate outputs without fitting anything.

## Method references
- [Scikit-learn: nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).
- [Scikit-learn: decision-threshold tuning and overfitting](https://scikit-learn.org/stable/modules/classification_threshold.html).

This follow-up was designed after inspecting the earlier accuracy-recall-001 results. The higher inner recall target was fixed before this run. It does not undo adaptive development bias.

[CatBoost native categorical features](https://catboost.ai/docs/en/features/categorical-features).
