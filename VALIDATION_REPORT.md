# AeroLife deployment validation report

Date: 2026-11-10

## Source and artifacts

- Notebook source: `AeroLife_RUL_Thesis_JOURNAL_DEPLOYMENT_REVISION (1).ipynb`.
- Model archive: `artifacts-20261110T045805Z-1-001.zip`.
- ZIP integrity: PASS for both source artifact archive and deployment package.
- Four R1 ensemble JSON bundles, shared feature pipeline, CatBoost/XGBoost/LightGBM models, and four empirical interval tables: present.

## Actual NASA official-test inference and parity

The four uploaded NASA C-MAPSS `test_FD00x.txt` files were parsed as 26-column, whitespace-delimited, headerless raw histories. The runtime predicted one final-cycle RUL per engine. Predictions were compared with the notebook archive's `official_test_predictions.csv` for the matching dataset and engine IDs.

| Dataset | Engines | Rows | RMSE difference vs saved notebook predictions | Maximum absolute difference |
|---|---:|---:|---:|---:|
| FD001 | 100 | 13,096 | 6.24e-15 | 2.84e-14 |
| FD002 | 259 | 33,991 | 5.68e-15 | 2.84e-14 |
| FD003 | 100 | 16,596 | 5.06e-15 | 2.84e-14 |
| FD004 | 248 | 41,214 | 4.28e-15 | 2.84e-14 |

Result: PASS. Predictions match the saved notebook predictions to floating-point precision for all 707 engines. All predictions and interval bounds were finite, and interval ordering checks passed.

This is an independent runtime-vs-saved-output parity check, not a new accuracy evaluation against true RUL labels.

## Application checks and remaining work

- Python syntax compilation: PASS for app, runtime and smoke test.
- Application uploader supports CSV with required headers and original NASA whitespace-delimited `.txt` files (26 columns, no header).
- The runtime inference path was exercised on all four actual official test files.
- Streamlit is not installed in this build environment; a live browser/UI test has NOT been performed.
- Cloud deployment and public URL creation have NOT been performed. A GitHub repository and Streamlit Community Cloud deployment step are still required.
- Runtime environment has scikit-learn 1.8.0 while artifact metadata records 1.6.1; XGBoost also emitted an older-pickle warning. `requirements.txt` pins the recorded artifact versions. Use the pinned versions for deployment.

## Scope and methodology notes

This is a research demonstration using simulated NASA C-MAPSS benchmark data, not a certified real-aircraft maintenance system. The saved intervals are empirical residual-based intervals and are not claimed to provide formal conformal coverage guarantees.

The notebook's existing `crossfit_blend` / `crossfit_blend_bounded` functions fit fold-k blending weights on OOF rows from other folds. Since those other-fold base models may have trained on fold-k engines, this can transmit fold-k labels indirectly into the weights. A correct nested outer/inner cross-fitting re-evaluation is a separate research task; this deployment package does not claim to correct that methodology or replace the notebook's reported metrics.
