# AeroLife interface update — validation report

## Build contents

- The existing Streamlit dashboard functionality is retained (`aerolife_runtime.py`, benchmark data, custom input workflows, SHAP/report/download code); only `app.py` and project documentation were changed for this interface update.
- The native Streamlit sidebar is permanently expanded on load and its collapse/expand controls are hidden.
- The Overview now places the four-dataset Research Performance Snapshot after the workspace summary and before Explore AeroLife.
- The snapshot shows unweighted means: Primary Research Model mean R² 0.8992, mean RMSE 18.631 cycles and mean MAE 12.348 cycles; the mean NASA Score 1,763.1 is explicitly identified as the Historical Baseline Reference official-test value.
- User-facing labels consistently use **Primary Research Model** and **Historical Baseline Reference**. The researcher is identified as MD FAHIM MUNTASIR LEON and the Research Supervisor as FATEMA TUJ JOHORA.
- The active model bundles, historical baseline bundles, data files, training/model artifacts, ensemble weights, preprocessing, feature logic and runtime are unchanged.

## Average metric definitions

All four averages are unweighted arithmetic means of the four dataset-specific values currently displayed in the dashboard. They are not pooled-engine metrics. The first three use the strict nested-validation estimates for the Primary Research Model. The NASA average is calculated separately from historical official-test scores for the Historical Baseline Reference; it is not a Primary Research Model NASA score.

| Overview card | Source | Display value |
|---|---|---:|
| Mean R² | Primary Research Model strict nested validation | 0.8992 |
| Mean RMSE | Primary Research Model strict nested validation | 18.631 cycles |
| Mean MAE | Primary Research Model strict nested validation | 12.348 cycles |
| Mean NASA Score | Historical Baseline Reference official test | 1,763.1 |

## Checks

- `python -m py_compile app.py aerolife_runtime.py tests/*.py` — PASS.
- `tests/smoke_test_runtime.py` — PASS for all four dataset bundles using synthetic histories (2 engines per dataset).
- `tests/test_research_selection_runtime.py` — PASS for all four bundled NASA test input datasets (100, 259, 100 and 248 engines); finite component/ensemble predictions and valid interval ordering verified.
- The UI label helper changes only the displayed/exported `ensemble` name in prediction result tables; it does not change model predictions or inference inputs.
- Bundle fingerprints, selected members, weights, artifact presence and model metadata remain validated by the runtime tests.

A LightGBM feature-name warning may appear during inference because the runtime passes a NumPy feature matrix to an estimator that retains feature-name metadata; it does not prevent predictions or the checks above.

## Important limits

- The displayed Primary Research Model metrics are strict nested outer-validation estimates for the selection procedure. They are not represented as measured accuracy of the final all-development-data refit artifacts.
- Runtime smoke tests verify artifact loading and prediction execution, not predictive accuracy. No new accuracy metric was calculated as part of this interface update.
- A live Streamlit browser/hosted deployment test was not run in this build environment. After pushing, verify the permanently visible sidebar on the hosting platform.
- NASA C-MAPSS is simulated benchmark data. This academic demonstration is not certified for operational aircraft maintenance.
