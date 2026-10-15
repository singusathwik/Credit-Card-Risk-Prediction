# Architecture

```text
Original UCI workbook -> schema validation + profile/ID connected groups
 -> train / validation / held-out test
 -> training-only three-fold comparisons
    [FeatureBuilder -> ColumnTransformer -> estimator]
 -> validation AP selection -> training-CV calibration -> validation threshold
 -> local full-pipeline bundle + checksum/dependency manifest
 -> one frozen evaluation -> aggregate JSON/CSV + PNG reports

React form -> Flask validation -> same saved pipeline -> probability
                               -> permutation SHAP -> contributions
React evaluation page <- version-matched reports and allowlisted PNGs
```

## Modules
- `credit_risk/features.py`: 23-field contract, eight derived fields, sklearn pipeline.
- `data.py`: Excel reader, profile/ID grouping and partition construction.
- `train.py`: bounded search, group CV, validation selection/calibration/threshold.
- `metrics.py`: classification/probability metrics, threshold grid, group bootstrap.
- `evaluate.py`: frozen evaluation, aggregate errors/subgroups and charts.
- `explain.py`: SHAP for the complete served probability function.
- `artifacts.py`: trusted local serialization, corruption and dependency checks.
- `server.py`: app factory, safe errors, configurable CORS, aliases and limited static serving.
- `src/`: retained React component structure with upgraded form/results and evaluation view.

## Feature decisions
Retained bill mean/std, payment mean/std, mean bill/limit utilization, positive-delay count and maximum positive delay. Added absolute latest-minus-oldest bill change. Nonpositive denominators become missing and use training-fitted imputation. Negative bills remain valid. These are historical summaries, not accounting balances or causal drivers.

Removed redundant summaries, arbitrary flags, +1 ratio denominators and unverified outstanding-balance claims. Raw repayment codes are nominal; delay summaries use only positive delays. Preprocessing supports missing values, but incomplete API submissions are rejected rather than silently treated as zero.

## Intentional compatibility changes
Flask, React, Vite, component boundaries, all raw inputs and predict/accuracy aliases remain. Root HTML now mounts React; production serves only `dist`. Old UI, code and README are preserved in `legacy/`. Risk labels use server metadata instead of conflicting browser rules. Tailwind/Vite were updated to resolve reported dependency vulnerabilities. No automatic loading of legacy models with incompatible features occurs. This ZIP-style local folder has no Git history; edits do not imply remote commits.
