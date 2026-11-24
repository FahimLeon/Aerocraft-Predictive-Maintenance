# AeroLife — Public Streamlit Dashboard

## What visitors can do

The app has two workspaces:

1. **NASA Benchmark Explorer** — FD001, FD002, FD003 and FD004 test-engine histories are bundled in `data/nasa_test/`. Visitors can select a dataset and engine ID (no benchmark-file upload required), then generate a full RUL/health report with CatBoost, XGBoost, LightGBM and the weighted R1 hybrid predictions, empirical interval, risk band, sensor trend, global/local SHAP explanations, and CSV/PDF downloads.
2. **Custom Data Prediction** — visitors can upload a compatible external CSV or NASA whitespace TXT and run the same frozen R1 pipeline.

The matching NASA `RUL_FD001.txt` through `RUL_FD004.txt` ground-truth files are bundled under `data/nasa_rul/` and loaded automatically in Benchmark Explorer. Labels are validated against the selected subset's engine count and mapped in NASA test-engine order. Actual RUL and prediction error are never estimated or fabricated.

## Model and scope

The frozen model is the existing **CatBoost + XGBoost + LightGBM weighted hybrid ensemble**. No Extra Trees or ARCHE model is part of these artifacts. The app does not retrain the model.

This is an academic research demonstration using simulated NASA C-MAPSS benchmark data. It is not certified for operational aircraft maintenance and must not be used as a real-world maintenance decision system.

## Run locally

Use Python 3.12 (the same major/minor version selected for the Streamlit Cloud deployment):

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

Scientific-library versions are pinned to the versions recorded alongside the supplied trained artifacts. This matters because scikit-learn and XGBoost pickle/joblib compatibility can change between versions.

## Custom upload format

CSV with headers `unit`, `cycle`, `os1`, `os2`, `os3`, `s1` ... `s21`, or an original NASA whitespace-delimited headerless TXT file with 26 columns. Each engine must have cycles starting at 1 and contiguous, with one row per cycle. Raw readings are required; do not upload pre-engineered features.

## Metrics and uncertainty

Dataset MAE, RMSE, R² and NASA Score are displayed as previously recorded thesis official-test evaluation results, not recalculated from the currently selected engine. The saved prediction intervals are empirical residual-based intervals, not formal conformal coverage guarantees. SHAP values are weighted across the three tree members and explain model outputs before post-prediction clipping and interval post-processing; they are not causal effects.

## Validation notes

The runtime was exercised against the bundled official NASA test-input files for all four subsets in the current build environment. A SHAP smoke test was also run for FD001. A complete Streamlit browser test and a Python 3.12 clean-environment deployment test are not performed in this build environment; the cloud app should be checked after the GitHub update.


## Ground-truth RUL labels

The matching official NASA test-set labels are bundled under `data/nasa_rul/` as `RUL_FD001.txt` through `RUL_FD004.txt`. The dashboard loads the selected subset's labels automatically and maps each label to the corresponding test engine in NASA test-engine order. Actual RUL and prediction error are not inferred when matching labels are unavailable.

## Custom data input modes

Custom Data Prediction supports CSV/TXT upload and manual sensor entry. Manual entry includes a single-cycle form for recording one observation and a historical-cycle editor for valid predictions. The frozen R1 feature pipeline requires contiguous engine history from cycle 1; single-cycle-only prediction is therefore intentionally disabled.
