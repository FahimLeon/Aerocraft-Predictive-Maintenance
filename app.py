from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from aerolife_runtime import RULPredictor, RAW_COLS, SUBSETS

st.set_page_config(page_title="AeroLife | RUL Prediction", page_icon="✈️", layout="wide")
st.title("✈️ AeroLife — Turbofan Engine RUL Prediction")
st.caption("NASA C-MAPSS benchmark demonstration • Frozen R1 CatBoost + XGBoost + LightGBM ensemble")
st.info("This application uses simulated NASA C-MAPSS benchmark data. It is a research demonstration, not a certified aircraft maintenance or real-aircraft deployment system.")

with st.expander("How to use this app", expanded=True):
    st.markdown("""
    1. Choose the C-MAPSS subset that matches your CSV.
    2. Upload a CSV containing the full engine history: `unit`, `cycle`, `os1`–`os3`, and `s1`–`s21`.
    3. Each engine must have cycles starting at 1 with no gaps, and one row per cycle.
    4. Click **Predict RUL**. The app reports one result per engine using its latest cycle.
    5. Download the results as CSV.

    The uploaded file must contain raw sensor readings, not pre-engineered features and not official test labels.
    """)

@st.cache_resource(show_spinner="Loading the frozen ensemble artifacts...")
def load_predictor(dataset):
    return RULPredictor(dataset, root=HERE)

@st.cache_data(show_spinner=False)
def validate_and_read(file_bytes, filename):
    import io
    name = (filename or "").lower()
    if name.endswith(".txt"):
        # NASA C-MAPSS official test files are whitespace-delimited and headerless.
        df = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None)
        if df.shape[1] != len(RAW_COLS):
            raise ValueError(f"NASA TXT file must contain {len(RAW_COLS)} columns; found {df.shape[1]}.")
        df.columns = RAW_COLS
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    return df

left, right = st.columns([1, 2])
with left:
    dataset = st.selectbox("Select dataset", SUBSETS, index=0)
with right:
    uploaded = st.file_uploader("Upload engine-history CSV or NASA C-MAPSS TXT", type=["csv", "txt"])

st.caption("Required columns: " + ", ".join(RAW_COLS))

if uploaded is not None:
    try:
        raw = validate_and_read(uploaded.getvalue(), uploaded.name)
        st.write(f"Uploaded: **{uploaded.name}** · {len(raw):,} rows · {raw['unit'].nunique() if 'unit' in raw else '—'} engines")
        missing = [c for c in RAW_COLS if c not in raw.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            # Validate numeric fields before calling the frozen notebook predictor.
            numeric_cols = ["unit", "cycle", "os1", "os2", "os3"] + [f"s{i}" for i in range(1, 22)]
            non_numeric = []
            for col in numeric_cols:
                converted = pd.to_numeric(raw[col], errors="coerce")
                if converted.isna().any() or not converted.map(lambda x: pd.notna(x) and abs(float(x)) != float("inf")).all():
                    non_numeric.append(col)
                else:
                    raw[col] = converted
            if non_numeric:
                st.error("These columns must contain finite numeric values: " + ", ".join(non_numeric))
            elif (raw[["unit", "cycle"]] < 1).any().any() or not (raw[["unit", "cycle"]] % 1 == 0).all().all():
                st.error("Engine IDs (unit) and cycle values must be positive whole numbers.")
            elif raw.duplicated(["unit", "cycle"]).any():
                st.error("Duplicate engine-cycle rows found. Each engine must have exactly one row per cycle.")
            else:
                st.dataframe(raw.head(8), use_container_width=True)
            if not non_numeric and not (raw[["unit", "cycle"]] < 1).any().any() and (raw[["unit", "cycle"]] % 1 == 0).all().all() and not raw.duplicated(["unit", "cycle"]).any() and st.button("Predict RUL", type="primary", use_container_width=True):
                try:
                    predictor = load_predictor(dataset)
                    with st.spinner("Checking input and calculating predictions..."):
                        result = predictor.predict(raw)
                    st.success(f"Prediction completed for {len(result)} engines.")
                    a,b,c = st.columns(3)
                    a.metric("Engines", f"{len(result):,}")
                    b.metric("Mean predicted RUL", f"{result.predicted_rul.mean():.1f} cycles")
                    c.metric("Critical / High", int(result.risk_band.str.contains(r"^(Critical|High)").sum()))
                    st.subheader("Prediction results")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    chart = px.bar(result, x="engine", y="predicted_rul", color="risk_band", title="Predicted RUL by engine", labels={"engine":"Engine ID", "predicted_rul":"Predicted RUL (cycles)"})
                    st.plotly_chart(chart, use_container_width=True)
                    interval = px.scatter(result, x="engine", y="predicted_rul", error_y=(result.rul_upper_90-result.predicted_rul), error_y_minus=(result.predicted_rul-result.rul_lower_90), color="risk_band", title="Predicted RUL with empirical 90% intervals", labels={"engine":"Engine ID", "predicted_rul":"RUL (cycles)"})
                    st.plotly_chart(interval, use_container_width=True)
                    counts = result.risk_band.value_counts().rename_axis("Risk level").reset_index(name="Engines")
                    st.subheader("Risk distribution")
                    st.dataframe(counts, use_container_width=True, hide_index=True)
                    csv = result.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions (CSV)", csv, file_name=f"aerolife_{dataset}_predictions.csv", mime="text/csv", use_container_width=True)
                    st.caption("Intervals are empirical residual-based intervals from the saved notebook artifacts; they are not presented as formal conformal coverage guarantees.")
                except Exception as e:
                    st.error(f"Prediction could not be completed: {e}")
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

st.divider()
st.markdown("**Model:** Frozen R1 three-family ensemble · **Datasets:** FD001–FD004 · **Purpose:** academic research and demonstration")
