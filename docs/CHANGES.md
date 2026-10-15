# Upgrade inventory

## Created
- `credit_risk/{features,data,train,metrics,evaluate,explain,artifacts}.py`: shared ML package.
- `tests/test_pipeline.py`, `tests/test_api.py`, `tests/browser.spec.js`, pytest and Playwright configurations.
- `src/components/Evaluation.jsx`: results and fairness view.
- `requirements.txt`, `requirements-lock.txt`, `.env.example`, Dockerfile, `.dockerignore`, GitHub Actions CI.
- `examples/applicant.json`: complete synthetic API input.
- `docs/BASELINE_AUDIT.md`, `EXPERIMENTS.md`, `MODEL_CARD.md`, `DATASET_CARD.md`, `TECHNICAL_REPORT.md`, `ARCHITECTURE.md`, `RESUME_BULLETS.md`, `VERIFICATION.md`, screenshots.
- `reports/run-001`: selection parameters, fold/validation results, test metrics/intervals, error and subgroup aggregates, CSV and ten charts.
- `scripts/write_docs.py`: measured documentation generation.
- `artifacts/run-001` and `artifacts/reproduction`: ignored local model bundles, manifests and split indices.
- `legacy/`: original training/server/UI/manual test and README snapshots, with warning notice.

## Modified
- `train_model.py`: compatibility wrapper for the corrected CLI.
- `server.py`: app factory, validated API, SHAP, report serving and Waitress.
- `src/App.jsx`, main entry, existing Header/RiskForm/RiskResults/StatsPanel and CSS.
- `index.html`: mounts React instead of a second standalone application.
- `package.json`, lockfile, Vite/PostCSS configuration: build and dependency fixes.
- `.gitignore`: datasets, artifacts and local environments excluded from new additions.
- README and four quick-start/status/troubleshooting guides: consolidated accurate instructions.
- Root manual tests: assertions rather than printing unsupported expectations.
- `model_accuracy.json`, `feature_columns.json`: current report/raw-input compatibility snapshots; original versions retained in legacy.

## Principal bugs corrected
Training-derived fields zeroed during inference; test-based model selection;
SMOTE contamination of stacking folds; inappropriate category interpolation;
silent invalid-input coercion; source/data exposure via Flask static root;
debug/wildcard CORS defaults; disconnected React mount; undefined CSS utility;
dictionary/model mismatch in manual tests; inconsistent risk thresholds;
non-standard NaN in parameter JSON; mislabeled split order caused by JSON key
sorting; and lost form state when returning from evaluation.

Original notebook is retained unchanged as historical, with its leakage and metric
problems documented. It is not the maintained training entry point. No remote
commit, code license, Docker execution or cloud deployment is implied.
