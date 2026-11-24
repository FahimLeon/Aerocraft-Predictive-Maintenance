# AeroLife Research Selection deployment — validation report

## Build contents

- Existing Streamlit dashboard functionality retained (`aerolife_runtime.py`, benchmark data, custom input workflows, SHAP/report/download code); `app.py` received a presentation-only redesign.
- Added a native Streamlit slide-in navigation drawer, collapsed by default, with Overview, NASA Benchmark Explorer, Custom Data Prediction and About AeroLife sections.
- Added project attribution for MD FAHIM MUNTASIR LEON and supervisor FATEMA TUJ JOHORA.
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

## Dashboard metric provenance

- Research Selection R², RMSE and MAE remain the strict nested outer-validation results.
- The displayed NASA Score is explicitly the historical Frozen R1 official NASA test-endpoint reference, dataset by dataset. It is not attributed to Research Selection.
- Official Frozen R1 NASA scores: FD001 506.273; FD002 2951.298; FD003 863.645; FD004 2731.377.

## Checks performed

- `python -m py_compile app.py aerolife_runtime.py tests/*.py` — PASS.
- `tests/smoke_test_runtime.py` — PASS for all four dataset bundles using synthetic histories (2 engines per dataset).
- `tests/test_research_selection_runtime.py` — PASS for all four bundled NASA test input datasets (100, 259, 100 and 248 engines); finite component/ensemble predictions and valid interval ordering verified.
- Bundle fingerprints, selected members, weights, artifact presence and model metadata are validated at runtime.

A LightGBM feature-name warning appears during inference because the runtime passes a NumPy feature matrix to an estimator that retains feature-name metadata; it does not prevent predictions or the checks above.

## Important limits

- The displayed Research Selection metrics are strict nested outer-validation estimates for the selection procedure. They are not represented as measured accuracy of the final all-development-data refit artifacts.
- The runtime smoke tests verify artifact loading and prediction execution, not predictive accuracy. No official-test accuracy was recalculated as part of this deployment build.
- A live Streamlit browser/hosted deployment test was not run in this build environment. After pushing, verify the app on the hosting platform.
- NASA C-MAPSS is simulated benchmark data. This academic demonstration is not certified for operational aircraft maintenance.
