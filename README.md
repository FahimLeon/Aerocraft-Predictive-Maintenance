# AeroLife — Aircraft Engine Health Research Dashboard

## Interface and navigation

The dashboard uses a responsive Streamlit layout with a **persistent, always-visible sidebar**. It opens expanded by default and its collapse/expand controls are hidden. Select one of the four workspaces directly from the sidebar:

1. **Overview** — project introduction, four-dataset average research-performance snapshot, dataset-level details and links to the workspaces.
2. **NASA Benchmark Explorer** — FD001, FD002, FD003 and FD004 test-engine histories are bundled under `data/nasa_test/`. Visitors can select a dataset and engine ID, then generate a RUL/health report with CatBoost, XGBoost, LightGBM and the weighted Primary Research Model, empirical residual interval, risk band, sensor trend, global/local SHAP explanations, and CSV/PDF downloads.
3. **Custom Data Prediction** — visitors can upload a compatible CSV or NASA whitespace TXT, or use the existing manual single-cycle and historical-cycle input modes. Valid predictions require contiguous engine history beginning at cycle 1 because the model uses causal rolling, slope, drift and health-index features.
4. **About AeroLife** — project summary, model information and researcher/supervisor attribution.

## Primary Research Model

The active deployment is the **Primary Research Model**, a three-family weighted ensemble (CatBoost + XGBoost + LightGBM). The selection log contains 60 inner-only configuration records (3 seeds × 5 outer folds × 4 datasets). For the final all-development-data refit, the deployment configuration uses the modal selected target cap per dataset and the component-wise median inner-only blend weights among records selecting that cap, normalized to sum to one. The final members are fitted on all 709 development engines.

The active per-dataset target caps are stored in `artifacts/thesis/results/RESEARCH_SELECTION_DEPLOYMENT.json`. FD001, FD003 and FD004 use T200; FD002 uses cap 160. The three T200 base learners are reused from the historical baseline artifacts because their data fingerprint, feature pipeline, target and family hyperparameters match exactly. New cap-160 CatBoost, XGBoost and LightGBM members are trained for FD002. New weights are used for all four datasets.

The previous deployment artifacts are retained as the **Historical Baseline Reference** (`*_FrozenR1`). Original `final_*.joblib` artifacts are not overwritten.

## Research performance snapshot and metric provenance

The Overview displays unweighted arithmetic averages of the four dataset-specific metrics:

- **Mean R²:** 0.8992 — Primary Research Model, strict nested-validation estimates.
- **Mean RMSE:** 18.631 cycles — Primary Research Model, strict nested-validation estimates.
- **Mean MAE:** 12.348 cycles — Primary Research Model, strict nested-validation estimates.
- **Mean NASA Score:** 1,763.1 — Historical Baseline Reference, historical official NASA test-endpoint scores.

Each value is an unweighted mean across FD001–FD004, not a pooled-engine metric. NASA Score is kept as a separate reference and is not attributed to the Primary Research Model. The per-dataset source values are available in the dataset-level details and benchmark explorer.

The nested validation estimates are not the measured performance of the final refitted deployment artifacts and are not an individual engine's Actual RUL/error. Official NASA test labels remain bundled for benchmark-engine actual-RUL display only. NASA Score evaluates labelled benchmark predictions; it does not generate or adjust RUL predictions.

The empirical 80%/90% residual interval tables are recalibrated from the averaged repeated strict nested outer-validation predictions. They are empirical intervals, not formal conformal coverage guarantees.

## Project attribution

**Project:** Aircraft Engine Predictive Maintenance Using a Hybrid Ensemble Machine Learning Framework with Explainable Artificial Intelligence  
**Researcher:** MD FAHIM MUNTASIR LEON  
**Research Supervisor:** FATEMA TUJ JOHORA

## Validation and scope

- The strict nested outputs are retained under `artifacts/thesis/strict_nested/`.
- The deployment configuration and per-model metadata are retained under `artifacts/thesis/results/` and `artifacts/thesis/models/`.
- Runtime smoke tests should load all four dataset bundles and predict from synthetic and bundled NASA test histories.
- This is an academic research demonstration using simulated NASA C-MAPSS benchmark data. It is not certified for operational aircraft maintenance and must not be used as a real-world maintenance decision system.

## Run locally

Use Python 3.13 with the pinned dependencies in `requirements.txt`:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```

## Custom upload format

CSV with headers `unit`, `cycle`, `os1`, `os2`, `os3`, `s1` ... `s21`, or an original NASA whitespace-delimited headerless TXT file with 26 columns. Each engine must have cycles starting at 1 and contiguous, with one row per cycle. Raw readings are required; do not upload pre-engineered features.

## Ground-truth RUL labels

The matching official NASA test-set labels are bundled under `data/nasa_rul/` as `RUL_FD001.txt` through `RUL_FD004.txt`. The app validates label counts and maps them to the corresponding test engine in NASA order. Actual RUL and prediction error are never inferred or fabricated when matching labels are unavailable.
