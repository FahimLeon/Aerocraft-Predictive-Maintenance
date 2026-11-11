# AeroLife public dashboard update — validation notes

## Included benchmark inputs
- `test_FD001.txt`: 100 engines, 13,096 rows.
- `test_FD002.txt`: 259 engines, 33,991 rows.
- `test_FD003.txt`: 100 engines, 16,596 rows.
- `test_FD004.txt`: 248 engines, 41,214 rows.
- All four files were parsed as 26-column NASA C-MAPSS test histories.

## Runtime checks completed
- Python syntax compilation: PASS for `app.py` and `aerolife_runtime.py`.
- Frozen model bundle loading and actual benchmark-input inference: PASS for FD001–FD004.
- Detailed predictions expose CatBoost, XGBoost, LightGBM and weighted hybrid output columns.
- FD001 Engine 99 report inference: PASS (latest cycle 97; all component and hybrid predictions finite; interval and risk output returned).
- SHAP TreeExplainer smoke test: PASS for FD001 Engine 99.
- Global SHAP smoke test: PASS for a 30-engine FD001 sample (30 × 253 feature attribution matrix).
- Existing synthetic runtime smoke test: PASS for FD001–FD004.
- ZIP integrity check: PASS (`ZipFile.testzip()` returned no corrupt member).

## Not yet verified
- Streamlit is not installed in this build environment, so a live browser interaction test was not possible.
- Clean install and live execution under Python 3.12 on Streamlit Community Cloud must be confirmed after the GitHub update.
- Exact notebook-to-app parity against all official test predictions was not recalculated in this update.
- The four NASA `RUL_FD00x.txt` ground-truth label files are not bundled. Actual RUL/error appear only if the visitor supplies the matching label file.

## Research limitations
- Dataset-level MAE/RMSE/R²/NASA Score are displayed as previously recorded thesis evaluation values, not recalculated in the web app.
- Empirical residual intervals are not formal conformal coverage guarantees.
- SHAP values combine the three tree-member attributions using frozen ensemble weights and describe tree outputs before member clipping and interval post-processing. They are not causal explanations.
- This is a simulated benchmark research demonstration, not a certified real-aircraft maintenance system.
- Previously documented OOF blending leakage concerns are not addressed by this dashboard update. No retraining or methodological correction is claimed.
