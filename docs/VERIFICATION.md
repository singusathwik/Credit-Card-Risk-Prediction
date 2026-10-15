# Verification record — 15 October 2026

## Executed checks
- Original workbook inspection: 30,000 rows, 6,636 defaults, no missing cells, 56 repeated predictor profiles, no repeated IDs.
- Full training: ten configurations, three group-aware training folds, validation selection, three calibration methods and 91 validation thresholds.
- One frozen final evaluation: 5,996 test records, 300 profile-group bootstrap replicates, subgroup/error aggregates and ten plots.
- `python -m pytest -q`: **32 passed**, including train-only imputation, group separation, schema/range errors, batch/single prediction parity, missing values, probability bounds, artifact version/checksum rejection, SHAP additivity, safe errors, CORS and strict browser JSON.
- Fresh isolated **Python 3.12.14** environment: `pip install -r requirements.txt` succeeded; `pip check` reported no broken requirements; all 32 tests passed.
- Full training rerun in that clean environment reproduced the same selected model, calibration choice, threshold and every split index. All **6,004 validation probabilities matched exactly** (maximum absolute difference 0.0). The test set was not reopened. Evidence: [reproducibility.json](../reports/reproducibility.json).
- `python test_model.py` in the clean environment: actual saved-model/API probability equality and SHAP reconstruction passed. Synthetic example probability: 0.0613847114.
- `python test_server.py`: live health and prediction assertions passed.
- `python test_high_risk.py`: hypothetical delayed-payment profile returned valid probability 0.7171292305. This is an example, not a monotonicity guarantee.
- Training/evaluation CLI help commands executed successfully.
- `npm run build`: Vite production build passed.
- `npx playwright test`: **2 passed** against Waitress and the actual model, including prediction, local SHAP, evaluation charts, correct split labels, browser validation, API error display, retained form values and a 390px mobile viewport. Repeated against the clean Python 3.12 service.
- `npm audit`: **0 vulnerabilities** at final check. This is dependency-audit evidence, not a security certification.
- Desktop prediction, desktop evaluation and mobile screenshots captured; desktop and mobile layouts visually reviewed.

## Corrections discovered through verification
Browser testing found non-standard NaN in XGBoost parameter JSON; configuration sentinels now serialize as strings and existing logs were normalized without changing fitted models or numeric experiment results. Visual inspection found split sizes in JSON key order rather than train/validation/test order; an explicit order and browser assertion fix this. Returning from evaluation now restores the submitted profile.

## Environment and limits
Initial model fitting and evaluation ran on Python 3.14 with core ML versions recorded in metadata; its existing SHAP installation depended on beta numba/llvmlite packages. The supported installation is therefore Python 3.12 with stable pinned dependencies, verified in an isolated environment. The same saved model loads there and produces the same example probability and additive explanations. `requirements-lock.txt` records the full clean dependency snapshot before optional formatting tools were installed.

No Docker executable is installed, so image build/run is unverified. At the initial local verification, GitHub Actions had not run remotely and the source folder was a ZIP checkout without `.git`. Git history was subsequently restored from origin/main for publication. No cloud deployment was performed; remote CI status must be checked separately. The historical 81.32% model cannot be independently evaluated because its artifact is absent. Tests do not establish real lending suitability, causal explanations, external validity or fairness certification.
