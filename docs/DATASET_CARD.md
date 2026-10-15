# Dataset card

## Provenance and target
The supplied workbook is UCI's Default of Credit Card Clients, attributed to I-Cheng Yeh. It represents Taiwan credit-card customers with April–September 2005 histories and binary next-month default. See the [UCI dictionary](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), DOI 10.24432/C55S3H. Dataset license: CC BY 4.0; this does not establish a code license.

Local inspection: **30,000 rows**, 23 predictors, ID and target; **6,636 defaults (22.12%)**, 23,364 non-defaults, no missing cells, no repeated IDs and **56 duplicate predictor profiles** beyond first occurrences. Identical profiles are grouped even if labels differ; no target-based deduplication is performed.

SHA-256: `f2649c89a3d5f656ca7c7ede8246181f5897632b9b71b106270b405a6e6bbdd6`.

## Fields
- LIMIT_BAL: credit amount in NT dollars; AGE: years.
- SEX codes 1/2; EDUCATION codes 1–4 documented, observed 0/5/6 retained as undocumented categories; MARRIAGE 1–3 documented, observed 0 retained.
- PAY_0, PAY_2…PAY_6: September to April repayment status. -1 means paid duly; positive values indicate delay months. The official dictionary does not fully define observed -2/0, so these are nominal categories, not invented delay magnitudes.
- BILL_AMT1…6 and PAY_AMT1…6: statement amounts and previous payments, September to April, in NT dollars. Negative bills are retained; payments are nonnegative.
- `default payment next month`: 1 default, 0 non-default. ID is never a predictor.

Observed ages: 21–79; limits: NT$10,000–1,000,000. Broader API safety bounds do not establish model validity outside this range. Six-month histories are columns within a single observation; there are no separate outcome cohorts for chronological backtesting. Unique IDs do not exclude hidden family/account relationships.

## Acquisition and privacy
Use the existing workbook or obtain the exact workbook from UCI's Download button and place it at root. Read using xlrd and `header=1`. No alternative dataset is downloaded automatically. New Excel files and artifacts are ignored by Git; the existing public repository may already track the workbook, which ignore rules do not untrack.

Use synthetic form inputs, not personal records. Reports publish aggregates only. The local bundle includes 24 historical training rows as a SHAP background: treat it as data-bearing and review before redistribution. Historical geography, sampling and undocumented codes limit generalization.
