# AeroLife — Research Selection Streamlit Dashboard

## What visitors can do

AeroLife keeps the existing two-workspace dashboard:

1. **NASA Benchmark Explorer** — FD001, FD002, FD003 and FD004 test-engine histories are bundled under `data/nasa_test/`. Visitors can select a dataset and engine ID, then generate a RUL/health report with CatBoost, XGBoost, LightGBM and the weighted Research Selection ensemble, empirical residual interval, risk band, sensor trend, global/local SHAP explanations, and CSV/PDF downloads.
2. **Custom Data Prediction** — visitors can upload a compatible CSV or NASA whitespace TXT, or use the existing manual single-cycle and historical-cycle input modes. Valid predictions require contiguous engine history beginning at cycle 1 because the model uses causal rolling, slope, drift and health-index features.

## Active model

The active deployment is the **Research Selection three-family ensemble** (CatBoost + XGBoost + LightGBM). The selection log contains 60 inner-only configuration records (3 seeds × 5 outer folds × 4 datasets). For the final all-development-data refit, the deployment configuration uses the modal selected target cap per dataset and the component-wise median inner-only blend weights among records selecting that cap, normalized to sum to one. The final members are fitted on all 709 development engines.

The active per-dataset target caps are stored in `artifacts/thesis/results/RESEARCH_SELECTION_DEPLOYMENT.json`. FD001, FD003 and FD004 use T200; FD002 uses cap 160. The three T200 base learners are reused from the previous Frozen R1 artifacts because their data fingerprint, feature pipeline, target and family hyperparameters match exactly. New cap-160 CatBoost, XGBoost and LightGBM members are trained for FD002. New Research Selection ensemble weights are used for all four datasets.

The previous Frozen R1 bundles and interval tables are retained as rollback copies (`*_FrozenR1`). The old `final_*.joblib` artifacts are not overwritten.

## Metrics shown in the dashboard

The dashboard preserves the original Research Selection strict nested outer-validation values for MAE, RMSE and R², along with the original validation-engine counts. The **NASA Score field alone** displays the separately verified Official NASA Test Endpoint score for the active Research Selection model, using one latest-cycle prediction per official test engine and NASA's supplied RUL labels. These values come from different evaluation protocols and are explicitly labelled; they must not be interpreted as one single evaluation run. NASA Score is an evaluation metric only and does not feed into or alter predictions.

The empirical 80%/90% residual interval tables are recalibrated from the averaged repeated strict nested outer-validation predictions. They are empirical intervals, not formal conformal coverage guarantees.

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
