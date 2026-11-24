# AeroLife Research Selection deployment — validation report

## Build contents

- Existing Streamlit dashboard structure retained (`app.py`, `aerolife_runtime.py`, benchmark data, custom input workflows, SHAP/report/download code).
- Active `ensemble_FD001.json` through `ensemble_FD004.json` now load Research Selection bundles.
- Previous Frozen R1 ensemble JSONs and interval tables are preserved with `_FrozenR1` suffixes; original `final_*.joblib` artifacts are unchanged.
- Research Selection selection log, nested metrics and outer predictions are bundled under `artifacts/thesis/strict_nested/`.
- Empirical 80%/90% interval tables are recalibrated using averaged repeated outer-validation predictions.

## Final refit configuration

The selection log contains 15 inner-only selections per dataset. The final deployment rule selects the modal cap per dataset and uses component-wise median weights among records selecting that cap, followed by normalization:

| Dataset | Final target | CatBoost | XGBoost | LightGBM |
|---|---|---:|---:|---:|
| FD001 | T200 | 0.7758 | 0.1121 | 0.1121 |
| FD002 | 160 | 0.4297 | 0.2227 | 0.3476 |
| FD003 | T200 | 0.7261 | 0.1071 | 0.1668 |
| FD004 | T200 | 0.3681 | 0.1915 | 0.4403 |

The four uploaded training files reproduce the original development-data fingerprint `2ee5a4b9b0f8fe1a06e0` (709 engines; 160,359 rows). The three T200 base learners and feature pipeline were reused from Frozen R1 after matching data fingerprint, target and family hyperparameter metadata. New CatBoost, XGBoost and LightGBM T160 members were fitted on all 709 development engines for FD002.

## Checks performed

- `python -m py_compile app.py aerolife_runtime.py tests/*.py` — PASS.
- `tests/smoke_test_runtime.py` — PASS for all four dataset bundles using synthetic histories (2 engines per dataset).
- `tests/test_research_selection_runtime.py` — PASS for all four bundled NASA test input datasets (100, 259, 100 and 248 engines); finite component/ensemble predictions and valid interval ordering verified.
- Bundle fingerprints, selected members, weights, artifact presence and model metadata are validated at runtime.

A LightGBM feature-name warning appears during inference because the runtime passes a NumPy feature matrix to an estimator that retains feature-name metadata; it does not prevent predictions or the checks above.

## Important limits

- The dashboard's primary metrics display is now the Official NASA Test Endpoint evaluation for the active Research Selection model. It uses one latest-cycle prediction per official test engine and the matching NASA RUL labels. Strict nested outer-validation results remain in the research artifacts and are not shown in the live dashboard.
- Official endpoint metrics (FD001 / FD002 / FD003 / FD004): NASA Score 503.3227 / 2520.7204 / 903.7590 / 2728.7461; RMSE 14.1326 / 19.1545 / 15.3597 / 17.5809; MAE 9.2433 / 12.1354 / 9.8369 / 11.6358; R² 0.8843 / 0.8731 / 0.8623 / 0.8960.
- NASA Score is an evaluation metric only. It does not feed into or alter the prediction pipeline. Custom-data NASA Score is not available unless true RUL labels are supplied independently.
- Runtime smoke tests verify artifact loading and prediction execution, not predictive accuracy. Endpoint metric values are the previously verified latest-cycle predictions for these Research Selection artifacts.
- A live Streamlit browser/hosted deployment test was not run in this build environment. After pushing, verify the app on the hosting platform.
- NASA C-MAPSS is simulated benchmark data. This academic demonstration is not certified for operational aircraft maintenance.
