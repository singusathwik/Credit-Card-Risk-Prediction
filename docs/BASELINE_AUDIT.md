# Baseline audit — 15 October 2026

Completed before implementation changes. Original files are preserved in `legacy/`;
the Colab notebook remains unchanged as historical work, not a validated experiment.
The local folder has no `.git` directory. No remote publication has been performed.

## Architecture and functionality
Root Flask server serves a standalone HTML form and `/predict`, `/api/predict`,
`/accuracy`, `/api/accuracy`. A second React/Vite/Tailwind frontend exists in `src/`,
but root `index.html` does not mount it. Both accept 23 UCI input fields and display
probabilities and hard-coded risk categories. Vite proxies `/api` to Flask.
There are no dependency pins, CI, Dockerfile, or assertion-based unit tests.
The three test scripts are manual demos; HTTP failures can print and still exit zero.

## Data and algorithms
Local `default of credit card clients.xls` is available. Target is
`default payment next month` (1 = default), renamed `IsDefaulter` in training.
ID is excluded by the Python trainer. Notebook also contains Logistic Regression,
SVC, decision tree, RF and XGBoost, Colab paths and incompatible older pandas calls.
The script trains two large random forests, gradient boosting, optional XGBoost,
stacking and voting. Many ratios, delay aggregates and interactions are engineered.

## Evaluation and reproducible starting point
Script uses stratified 80/20 seed-42 splitting, then chooses the model with highest
**test accuracy**. Saved `model_accuracy.json` claims 0.8131666667 test accuracy,
0.9861698143 resampled training accuracy and 0.9785833333 original training accuracy.
No `model.pkl` accompanies this checkout: these are historical, unverified numbers,
not a reproducible deployable baseline or a fair before/after comparison.
The missing artifact prevents measuring its live inference performance. Re-running
the expensive legacy selection would reproduce a biased protocol, not an unbiased
baseline; the upgrade will measure fresh dummy and conventional baselines instead.

Executed workbook inspection confirmed 30,000 rows, 25 columns, 6,636 positives,
zero missing cells, unique IDs and 56 repeated raw predictor profiles. Historical
model performance remains unverified; these are measured dataset facts.

## Confirmed problems
- Inference silently fills every engineered feature with zero rather than computing it.
- Invalid and missing values are silently converted to zero; no finite/range checks.
- SMOTE occurs after the outer split, but before stacking CV: synthetic neighbors
  cross internal folds. Standard SMOTE interpolates nominal category codes.
- Notebook applies SMOTE and scaling before splitting, includes ID during SMOTE,
  reverses truth/prediction arguments in several metrics and computes AUC on labels.
- Test selection causes optimistic results; no calibration, threshold study,
  uncertainty, subgroup audit or explanation capability exists.
- Duplicate engineered columns (volatility/std, efficiency/ratio, debt/bill trend)
  inflate apparent feature count. Delay codes -2/-1/0 are not numeric arrears.
  Same-month payment minus bill is not proven outstanding debt; ratios with +1
  denominators are unstable near zero and signed balances.
- Flask exposes the entire project as static files, enables debug and wildcard CORS.
  Some exceptions expose internal messages. Pickle loading has no version checks.
- Manual model test calls predict_proba on a saved dictionary; documented Excel
  dependency openpyxl does not read the supplied .xls file (xlrd is required).
- README duplicates conflicting UI/setup instructions and overstates deployment,
  tuning and accuracy. React CSS references an undefined border utility.

## Improvements and risks
Use shared raw-input pipelines; one-hot categorical/status codes; conservative
row-local summaries; training-only imputation/scaling. Risk: predictions change,
so old models are retained rather than silently loaded with new features.
Group identical predictor profiles and repeated IDs across partitions, then use
train/validation/test with fold-local preprocessing. Risk: historical scores are
not directly comparable. Evaluate weighted models instead of categorical SMOTE.
Select on validation average precision, calibrate with training CV, choose a
validation threshold, freeze before test. Risk: validation selection is optimistic;
single-cohort test cannot establish prospective validity.
Explain exact served probability using permutation SHAP. Risk: correlated features,
background dependence, approximate allocations and latency.
Retain Flask/React and API aliases; consolidate duplicate frontends. Risk: response
validation becomes stricter. Restrict static serving and trusted artifact loading.
Add automated tests, bounded experiment logging, charts, model/data cards and
honest documentation. No production lending or causal claims are supported.
