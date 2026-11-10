# AeroLife Streamlit Deployment

## Important scope
This is a research demonstration using simulated NASA C-MAPSS benchmark data. It is not a certified real-aircraft maintenance system.

## Run locally
1. Use Python 3.13.18 (the version recorded in the model artifact metadata).
2. Open a terminal in this folder.
3. Create an environment: `python -m venv .venv`
4. Activate it, then run: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`

## Install dependencies
The scientific-library versions are pinned to the versions recorded alongside the supplied trained artifacts. This matters because scikit-learn and XGBoost pickle/joblib compatibility can change between versions.

## Input files
Upload either:
- A CSV with headers `unit`, `cycle`, `os1`, `os2`, `os3`, `s1` ... `s21`; or
- An original NASA C-MAPSS whitespace-delimited, headerless `test_FD00x.txt` file (26 columns).

Each engine's cycles must start at 1 and be contiguous. No missing/non-finite values are accepted. Use raw readings, not engineered features. Select the matching FD001–FD004 subset before predicting.

## Artifact layout
The `artifacts/thesis/models` folder contains the four frozen ensemble bundles and shared fitted feature/model objects. `artifacts/thesis/results` contains the four empirical interval tables and the frozen ensemble selection record used by bundle validation.

## Uncertainty note
The saved intervals are empirical residual-based intervals. They are not claimed to provide formal conformal coverage guarantees.

## Validation status
The deployment package includes a runtime smoke-test report. Model bundle loading and synthetic-input inference were exercised for all four subsets in the build environment. This is not an official-test parity test or a full Streamlit browser test.
