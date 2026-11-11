from pathlib import Path
import io
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from aerolife_runtime import RULPredictor, RAW_COLS, SUBSETS

st.set_page_config(page_title="AeroLife | Aircraft Engine Health", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# Previously recorded official-test results from the thesis evaluation. These are
# clearly separated from the live engine prediction and are not recomputed here.
THESIS_TEST_METRICS = {
    "FD001": {"engines": 100, "r2": 0.8843, "rmse": 14.133, "mae": 9.268, "nasa": 506},
    "FD002": {"engines": 259, "r2": 0.8711, "rmse": 19.310, "mae": 12.356, "nasa": 2951},
    "FD003": {"engines": 100, "r2": 0.8629, "rmse": 15.330, "mae": 9.870, "nasa": 864},
    "FD004": {"engines": 248, "r2": 0.8960, "rmse": 17.586, "mae": 11.643, "nasa": 2731},
}

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(115deg,#10243b,#1b4965); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2rem; margin: 0 0 .35rem 0; color: white;}
.hero p {margin: 0; color: #d9eaf4;}
.small-note {font-size: .85rem; color: #64748b;}
</style>
<div class="hero"><h1>✈️ AeroLife</h1><p>Aircraft Engine Health Intelligence · Remaining Useful Life Prediction · Explainable AI</p></div>
""", unsafe_allow_html=True)
st.caption("NASA C-MAPSS research benchmark · Frozen R1 CatBoost + XGBoost + LightGBM weighted ensemble")
st.warning("Research demonstration only. NASA C-MAPSS data are simulated benchmark data, not live aircraft telemetry. Outputs must not be used as certified or operational maintenance instructions.")

@st.cache_resource(show_spinner="Loading frozen R1 models...")
def load_predictor(dataset: str):
    return RULPredictor(dataset, root=HERE)

@st.cache_data(show_spinner=False)
def load_benchmark(dataset: str) -> pd.DataFrame:
    path = HERE / "data" / "nasa_test" / f"test_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Bundled benchmark file is missing: {path.name}")
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] != len(RAW_COLS):
        raise ValueError(f"Expected {len(RAW_COLS)} columns; found {df.shape[1]}.")
    df.columns = RAW_COLS
    return df

@st.cache_data(show_spinner=False)
def read_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".txt"):
        frame = pd.read_csv(io.BytesIO(file_bytes), sep=r"\s+", header=None)
        if frame.shape[1] != len(RAW_COLS):
            raise ValueError(f"NASA TXT must contain {len(RAW_COLS)} columns; found {frame.shape[1]}.")
        frame.columns = RAW_COLS
        return frame
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def ensemble_shap(dataset: str, X: np.ndarray):
    """Weighted SHAP values for the three frozen tree models (pre-output clipping)."""
    import shap
    predictor = load_predictor(dataset)
    total_values = None
    total_base = 0.0
    names = predictor.bundle.get("feature_columns", [])
    for weight, _cap, model in predictor.members:
        explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        values = explainer.shap_values(X, check_additivity=False)
        if isinstance(values, list):
            values = values[0]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, 0]
        total_values = weight * values if total_values is None else total_values + weight * values
        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = float(np.asarray(base).reshape(-1)[0])
        total_base += weight * float(base)
    if not names or len(names) != X.shape[1]:
        names = [f"feature_{i}" for i in range(X.shape[1])]
    return total_values, total_base, names

def render_metrics(dataset: str):
    m = THESIS_TEST_METRICS[dataset]
    st.subheader("Dataset Performance")
    st.caption("Previously recorded thesis official-test evaluation metrics. These are dataset-level results, not the selected engine's individual actual RUL/error.")
    cols = st.columns(5)
    cols[0].metric("Test engines", f"{m['engines']:,}")
    cols[1].metric("MAE", f"{m['mae']:.3f} cycles")
    cols[2].metric("RMSE", f"{m['rmse']:.3f} cycles")
    cols[3].metric("R²", f"{m['r2']:.4f}")
    cols[4].metric("NASA Score", f"{m['nasa']:,}")

def render_engine_report(dataset: str, engine_id: int, history: pd.DataFrame, predictor: RULPredictor, actual_rul=None, show_shap: bool = True):
    pred = predictor.predict(history)
    row = pred.iloc[0]
    st.markdown(f"## Aircraft Engine Health Report — {dataset}")
    st.caption(f"Engine ID: {engine_id} · Latest observed cycle: {int(row.last_cycle)} · Ensemble: {row.ensemble}")
    risk = str(row.risk_band)
    if risk.startswith("Critical"):
        st.error(f"Risk assessment: {risk}")
    elif risk.startswith("High"):
        st.warning(f"Risk assessment: {risk}")
    elif risk.startswith("Medium"):
        st.info(f"Risk assessment: {risk}")
    else:
        st.success(f"Risk assessment: {risk}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Hybrid RUL", f"{row.predicted_rul:.2f} cycles")
    k2.metric("Lower 90% interval", f"{row.rul_lower_90:.2f}")
    k3.metric("Upper 90% interval", f"{row.rul_upper_90:.2f}")
    k4.metric("Current cycle", f"{int(row.last_cycle)}")

    st.markdown("### Individual model predictions")
    comp_cols = st.columns(4)
    comp_cols[0].metric("CatBoost", f"{row.catboost_prediction:.2f} cycles")
    comp_cols[1].metric("XGBoost", f"{row.xgboost_prediction:.2f} cycles")
    comp_cols[2].metric("LightGBM", f"{row.lightgbm_prediction:.2f} cycles")
    comp_cols[3].metric("Weighted Hybrid", f"{row.predicted_rul:.2f} cycles")

    if actual_rul is not None:
        error = float(row.predicted_rul) - float(actual_rul)
        a1, a2 = st.columns(2)
        a1.metric("Actual RUL", f"{float(actual_rul):.2f} cycles")
        a2.metric("Prediction error (predicted − actual)", f"{error:+.2f} cycles")
    else:
        st.info("Actual RUL and prediction error are unavailable because the matching NASA RUL_FD00x ground-truth label file was not supplied. They are deliberately not estimated or fabricated.")

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(row.predicted_rul), number={"suffix":" cycles", "valueformat":".1f"}, title={"text":"Predicted Remaining Useful Life"}, gauge={"axis":{"range":[0,200]}, "bar":{"color":"#1d7897"}, "steps":[{"range":[0,25],"color":"#fee2e2"},{"range":[25,50],"color":"#ffedd5"},{"range":[50,100],"color":"#fef3c7"},{"range":[100,200],"color":"#dcfce7"}] }))
    gauge.update_layout(height=280, margin=dict(l=20,r=20,t=60,b=10))
    st.plotly_chart(gauge, use_container_width=True)

    with st.expander("Prediction details and ensemble weights"):
        weights = pd.DataFrame([{"Model": m[2].__class__.__module__.split('.')[0], "Weight": m[0], "RUL cap": m[1]} for m in predictor.members])
        st.dataframe(weights, hide_index=True, use_container_width=True)
        st.dataframe(pred, hide_index=True, use_container_width=True)

    st.markdown("### Sensor trend analysis")
    sensor_options = [f"s{i}" for i in range(1,22)]
    chosen_sensor = st.selectbox("Choose sensor", sensor_options, index=3, key=f"sensor_{dataset}_{engine_id}")
    trend = px.line(history, x="cycle", y=chosen_sensor, markers=False, title=f"Engine {engine_id} · {chosen_sensor} across operating cycles")
    trend.update_layout(xaxis_title="Cycle", yaxis_title="Sensor reading")
    st.plotly_chart(trend, use_container_width=True)

    st.markdown("### Explainable AI — SHAP")
    if show_shap:
        try:
            X, feature_names, ids, cycles = predictor.latest_feature_matrix(history)
            engine_X = X[:1]
            # Dataset-wide latest-cycle sample for global explanation.
            all_history = load_benchmark(dataset)
            X_all, _, _, _ = predictor.latest_feature_matrix(all_history)
            rng = np.random.default_rng(42)
            take = np.arange(len(X_all)) if len(X_all) <= 50 else np.sort(rng.choice(len(X_all), 50, replace=False))
            global_X = X_all[take]
            shap_global, base_global, names = ensemble_shap(dataset, global_X)
            shap_local, base_local, _ = ensemble_shap(dataset, engine_X)
            mean_abs = np.mean(np.abs(shap_global), axis=0)
            top_idx = np.argsort(mean_abs)[-18:][::-1]
            global_df = pd.DataFrame({"Feature": np.array(names)[top_idx], "Mean |SHAP value|": mean_abs[top_idx]})
            fig_global = px.bar(global_df.sort_values("Mean |SHAP value|"), x="Mean |SHAP value|", y="Feature", orientation="h", title="Global SHAP feature importance · weighted ensemble")
            st.plotly_chart(fig_global, use_container_width=True)
            local_values = shap_local[0]
            local_idx = np.argsort(np.abs(local_values))[-18:][::-1]
            local_df = pd.DataFrame({"Feature": np.array(names)[local_idx], "SHAP contribution": local_values[local_idx]})
            local_df = local_df.sort_values("SHAP contribution")
            fig_local = px.bar(local_df, x="SHAP contribution", y="Feature", orientation="h", color="SHAP contribution", color_continuous_scale="RdBu", title=f"Local SHAP explanation · Engine {engine_id}")
            st.plotly_chart(fig_local, use_container_width=True)
            st.caption("SHAP contributions combine the three tree models using the frozen ensemble weights. They explain the weighted tree outputs before the runtime's member clipping and interval post-processing; they are feature attributions, not causal effects.")
        except ImportError:
            st.warning("SHAP package is not installed in this deployment. Install the pinned requirements and redeploy to enable explanations.")
        except Exception as exc:
            st.warning(f"SHAP calculation could not be completed for this engine: {exc}")

    csv_bytes = pred.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download this engine report (CSV)", csv_bytes, file_name=f"aerolife_{dataset}_engine_{engine_id}.csv", mime="text/csv")
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story = [Paragraph("AeroLife — Aircraft Engine Health Report", styles["Title"]), Spacer(1, 10), Paragraph(f"Dataset: {dataset} | Engine ID: {engine_id} | Latest cycle: {int(row.last_cycle)}", styles["BodyText"]), Spacer(1, 12)]
        rows = [["Measure", "Value"], ["CatBoost RUL", f"{row.catboost_prediction:.2f} cycles"], ["XGBoost RUL", f"{row.xgboost_prediction:.2f} cycles"], ["LightGBM RUL", f"{row.lightgbm_prediction:.2f} cycles"], ["Weighted Hybrid RUL", f"{row.predicted_rul:.2f} cycles"], ["Empirical 90% interval", f"[{row.rul_lower_90:.2f}, {row.rul_upper_90:.2f}]"], ["Risk band", risk]]
        if actual_rul is not None:
            rows.extend([["Actual RUL", f"{float(actual_rul):.2f} cycles"], ["Prediction error", f"{float(row.predicted_rul)-float(actual_rul):+.2f} cycles"]])
        table = Table(rows, colWidths=[210, 260])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F5F9")]),("PADDING",(0,0),(-1,-1),8)]))
        story.extend([table, Spacer(1, 14), Paragraph("Research demonstration only. Empirical intervals are not formal conformal guarantees. Not certified for operational aircraft maintenance.", styles["Italic"])])
        doc.build(story)
        st.download_button("⬇ Download report (PDF)", pdf_buffer.getvalue(), file_name=f"aerolife_{dataset}_engine_{engine_id}.pdf", mime="application/pdf")
    except Exception as exc:
        st.caption(f"PDF export unavailable: {exc}")
    return pred

mode = st.radio("Choose a workspace", ["NASA Benchmark Explorer", "Custom Data Prediction"], horizontal=True, index=0)

if mode == "NASA Benchmark Explorer":
    st.markdown("### Explore the preloaded NASA benchmark")
    st.write("Select a subset and engine ID. The benchmark engine histories are already included in this app—no upload is needed.")
    dataset = st.selectbox("NASA dataset", SUBSETS, key="benchmark_dataset")
    try:
        benchmark = load_benchmark(dataset)
        engine_ids = sorted(benchmark.unit.astype(int).unique().tolist())
        c1, c2, c3 = st.columns([1,1,2])
        with c1:
            engine_id = st.selectbox("Engine ID", engine_ids, index=min(98, len(engine_ids)-1), key=f"engine_{dataset}")
        with c2:
            st.metric("Engines in subset", f"{len(engine_ids):,}")
        with c3:
            selected_history = benchmark[benchmark.unit == engine_id].copy()
            st.metric("Available engine cycles", f"{len(selected_history):,}")
        predictor = load_predictor(dataset)
        render_metrics(dataset)
        label_upload = st.file_uploader("Optional: upload matching NASA RUL_FD00x.txt ground-truth labels to show Actual RUL and Prediction Error", type=["txt"], key=f"rul_labels_{dataset}")
        actual_labels = None
        if label_upload is not None:
            try:
                actual_labels = pd.read_csv(io.BytesIO(label_upload.getvalue()), header=None).iloc[:, 0].astype(float).to_numpy()
                if len(actual_labels) != len(engine_ids):
                    st.error(f"Ground-truth label count ({len(actual_labels)}) does not match engine count ({len(engine_ids)}). Labels ignored.")
                    actual_labels = None
                else:
                    st.success("Ground-truth labels matched to engine IDs in official NASA order.")
            except Exception as label_exc:
                st.error(f"Could not read RUL labels: {label_exc}")
        st.divider()
        if st.button("Generate Full Engine Health Report", type="primary", use_container_width=True):
            st.session_state["report_request"] = (dataset, int(engine_id))
        request = st.session_state.get("report_request")
        if request and request == (dataset, int(engine_id)):
            selected_history = benchmark[benchmark.unit == request[1]].copy()
            with st.spinner("Generating report, prediction and SHAP explanations..."):
                selected_actual = None if actual_labels is None else float(actual_labels[int(request[1]) - 1])
                render_engine_report(dataset, request[1], selected_history, predictor, actual_rul=selected_actual)
        st.markdown("### Dataset overview")
        overview = benchmark.groupby("unit").agg(first_cycle=("cycle","min"), latest_cycle=("cycle","max"), observations=("cycle","count")).reset_index().rename(columns={"unit":"Engine ID"})
        st.dataframe(overview, use_container_width=True, hide_index=True, height=260)
    except Exception as exc:
        st.error(f"Could not load this benchmark subset: {exc}")

else:
    st.markdown("### Predict using your own engine data")
    st.write("Upload a compatible full engine-history CSV or NASA C-MAPSS whitespace TXT. The selected dataset must match the data's operating regime/fault subset.")
    dataset = st.selectbox("Select the matching model subset", SUBSETS, key="custom_dataset")
    uploaded = st.file_uploader("Upload external engine history", type=["csv","txt"], help="Required raw columns: unit, cycle, os1–os3, s1–s21.")
    if uploaded is not None:
        try:
            raw = read_upload(uploaded.getvalue(), uploaded.name)
            missing = [c for c in RAW_COLS if c not in raw.columns]
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                for col in RAW_COLS:
                    raw[col] = pd.to_numeric(raw[col], errors="coerce")
                if raw[RAW_COLS].isna().any().any() or not np.isfinite(raw[RAW_COLS].to_numpy(float)).all():
                    st.error("All required input columns must contain finite numeric values; missing/invalid values were found.")
                elif (raw[["unit","cycle"]] < 1).any().any() or not (raw[["unit","cycle"]] % 1 == 0).all().all():
                    st.error("Engine IDs and cycle values must be positive whole numbers.")
                elif raw.duplicated(["unit","cycle"]).any():
                    st.error("Duplicate engine-cycle rows found.")
                else:
                    st.success(f"Loaded {len(raw):,} rows from {raw.unit.nunique():,} engine(s).")
                    st.dataframe(raw.head(8), use_container_width=True)
                    if st.button("Predict external engine data", type="primary"):
                        predictor = load_predictor(dataset)
                        with st.spinner("Running frozen R1 prediction pipeline..."):
                            result = predictor.predict(raw)
                        st.subheader("External data prediction results")
                        st.dataframe(result, use_container_width=True, hide_index=True)
                        st.metric("Mean hybrid RUL", f"{result.predicted_rul.mean():.2f} cycles")
                        fig = px.bar(result, x="engine", y="predicted_rul", color="risk_band", title="Predicted RUL by external engine")
                        st.plotly_chart(fig, use_container_width=True)
                        st.download_button("Download all predictions (CSV)", result.to_csv(index=False).encode("utf-8"), file_name=f"aerolife_custom_{dataset}_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Could not process uploaded data: {exc}")

with st.expander("Dataset metrics and model notes"):
    metric_rows = [{"Dataset": d, "Test engines": m["engines"], "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"], "NASA Score": m["nasa"]} for d,m in THESIS_TEST_METRICS.items()]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown("**Frozen model:** CatBoost + XGBoost + LightGBM weighted hybrid. No Extra Trees or ARCHE model is included in this R1 artifact set.")
    st.markdown("**Intervals:** empirical residual-based 90% intervals from saved artifacts; not formal conformal coverage guarantees.")
    st.markdown("**Evaluation:** historical thesis test-set metrics are displayed as reported; this web app does not recompute them from the selected engine.")
    st.markdown("**Data labels:** official NASA test input files do not contain the per-engine true RUL values. Actual RUL and prediction error are shown only if matching ground-truth labels are supplied in a future version.")

st.divider()
st.caption("AeroLife · Academic research and demonstration · Frozen R1 artifacts · Not certified for operational aircraft maintenance")
