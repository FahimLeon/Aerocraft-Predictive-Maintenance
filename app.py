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
def load_ground_truth(dataset: str, expected_engine_count: int) -> np.ndarray:
    """Load the matching official NASA test-set RUL labels bundled with AeroLife."""
    path = HERE / "data" / "nasa_rul" / f"RUL_{dataset}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Ground-truth labels are missing: {path.name}")
    labels = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0].astype(float).to_numpy()
    if len(labels) != expected_engine_count:
        raise ValueError(
            f"{path.name} contains {len(labels)} labels, but {dataset} has "
            f"{expected_engine_count} test engines. Labels were not applied."
        )
    if not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError(f"{path.name} contains invalid RUL labels.")
    return labels

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
        absolute_error = abs(float(row.predicted_rul) - float(actual_rul))
        signed_error = float(row.predicted_rul) - float(actual_rul)
        pred["actual_rul"] = float(actual_rul)
        pred["absolute_prediction_error"] = absolute_error
        pred["signed_prediction_error"] = signed_error
        a1, a2 = st.columns(2)
        a1.metric("Actual RUL", f"{float(actual_rul):.2f} cycles")
        a2.metric("Absolute prediction error", f"{absolute_error:.2f} cycles", help="Absolute difference between predicted and ground-truth RUL. Lower is better.")
    else:
        st.info("Actual RUL and prediction error are shown only when the matching official ground-truth label is available and validated. They are never estimated or fabricated.")

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
            absolute_error = abs(float(row.predicted_rul) - float(actual_rul))
            rows.extend([["Actual RUL", f"{float(actual_rul):.2f} cycles"], ["Absolute prediction error", f"{absolute_error:.2f} cycles"]])
        table = Table(rows, colWidths=[210, 260])
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#16324F")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#CBD5E1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#F1F5F9")]),("PADDING",(0,0),(-1,-1),8)]))
        explanation = (
            f"<b>Overall Explanation</b><br/><br/>The AeroLife AI model predicts that Engine {engine_id} "
            f"has approximately {float(row.predicted_rul):.2f} cycles of Remaining Useful Life (RUL). "
            f"This means the engine is estimated to have about {float(row.predicted_rul):.2f} operating cycles "
            "remaining based on the available sensor data and the AI model's prediction. The prediction is "
            "generated using a weighted combination of CatBoost, XGBoost, and LightGBM models. The estimated "
            "RUL represents the model's prediction of the engine's remaining useful life."
        )
        story.extend([
            table,
            Spacer(1, 14),
            Paragraph("Research demonstration only. Empirical intervals are not formal conformal guarantees. Not certified for operational aircraft maintenance.", styles["Italic"]),
            Spacer(1, 16),
            Paragraph(explanation, styles["BodyText"]),
        ])
        doc.build(story)
        st.download_button("⬇ Download report (PDF)", pdf_buffer.getvalue(), file_name=f"aerolife_{dataset}_engine_{engine_id}.pdf", mime="application/pdf")
    except Exception as exc:
        st.caption(f"PDF export unavailable: {exc}")

    st.markdown("### Overall Explanation")
    st.write(
        f"The AeroLife AI model predicts that Engine {engine_id} has approximately "
        f"{float(row.predicted_rul):.2f} cycles of Remaining Useful Life (RUL). "
        f"This means the engine is estimated to have about {float(row.predicted_rul):.2f} "
        "operating cycles remaining based on the available sensor data and the AI model's prediction. "
        "The prediction is generated using a weighted combination of CatBoost, XGBoost, and LightGBM models. "
        "The estimated RUL represents the model's prediction of the engine's remaining useful life."
    )
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
        actual_labels = None
        try:
            actual_labels = load_ground_truth(dataset, len(engine_ids))
            st.success(f"Official {dataset} ground-truth RUL labels loaded and matched to test engines in NASA order.")
        except Exception as label_exc:
            st.warning(f"Actual RUL labels are unavailable: {label_exc}")
        st.divider()
        if st.button("Generate Full Engine Health Report", type="primary", use_container_width=True):
            st.session_state["report_request"] = (dataset, int(engine_id))
        request = st.session_state.get("report_request")
        if request and request == (dataset, int(engine_id)):
            selected_history = benchmark[benchmark.unit == request[1]].copy()
            with st.spinner("Generating report, prediction and SHAP explanations..."):
                selected_actual = None if actual_labels is None else float(actual_labels[engine_ids.index(int(request[1]))])
                render_engine_report(dataset, request[1], selected_history, predictor, actual_rul=selected_actual)
        st.markdown("### Dataset overview")
        overview = benchmark.groupby("unit").agg(first_cycle=("cycle","min"), latest_cycle=("cycle","max"), observations=("cycle","count")).reset_index().rename(columns={"unit":"Engine ID"})
        st.dataframe(overview, use_container_width=True, hide_index=True, height=260)
    except Exception as exc:
        st.error(f"Could not load this benchmark subset: {exc}")

else:
    st.markdown("### Predict using your own engine data")
    st.write("Choose upload-based prediction or enter sensor readings manually. The selected dataset must match the data's operating regime/fault subset.")
    dataset = st.selectbox("Select the matching model subset", SUBSETS, key="custom_dataset")
    input_method = st.radio("Custom data input method", ["Upload Dataset", "Manual Sensor Input"], horizontal=True, key="custom_input_method")

    if input_method == "Upload Dataset":
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
                        if st.button("Predict external engine data", type="primary", key="predict_uploaded_custom"):
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

    else:
        manual_mode = st.radio("Manual input mode", ["Single Cycle Input", "Historical Cycle Input"], horizontal=True, key="manual_input_mode")
        if manual_mode == "Single Cycle Input":
            st.info("A single observation can be entered here, but the frozen R1 model uses historical rolling, slope, drift, and health-index features. A valid prediction cannot be generated from one cycle alone. Enter the engine's full cycle history in Historical Cycle Input to generate a prediction.")
            engine_id_manual = st.number_input("Engine ID", min_value=1, value=1, step=1, key="single_engine_id")
            cycle_manual = st.number_input("Current cycle", min_value=1, value=1, step=1, key="single_cycle")
            st.markdown("**Operating settings**")
            op_cols = st.columns(3)
            op_values = [op_cols[i].number_input(f"OS{i+1}", value=0.0, format="%.6f", key=f"single_os{i+1}") for i in range(3)]
            st.markdown("**Sensor readings**")
            sensor_values = []
            for base in range(0, 21, 7):
                cols = st.columns(7)
                for j in range(base, min(base + 7, 21)):
                    sensor_values.append(cols[j-base].number_input(f"S{j+1}", value=0.0, format="%.6f", key=f"single_s{j+1}"))
            st.caption("The values are collected for review, but prediction is intentionally disabled for a single cycle to avoid inventing missing historical features.")
            st.button("Generate Prediction", disabled=True, help="A full contiguous history is required by the frozen R1 feature pipeline.", key="single_cycle_blocked")

        else:
            st.write("Enter the engine's raw history from cycle 1 through its latest observed cycle. The frozen feature pipeline will calculate historical features using the same saved preprocessing and feature logic as the trained models.")
            engine_id_manual = st.number_input("Engine ID", min_value=1, value=1, step=1, key="history_engine_id")
            manual_cols = ["cycle"] + RAW_COLS[2:]
            template = pd.DataFrame([{c: None for c in manual_cols} for _ in range(5)])
            st.caption("Add one row per cycle. Cycles must start at 1 and be contiguous; all 3 operating settings and 21 sensor readings are required for every row.")
            edited = st.data_editor(template, num_rows="dynamic", use_container_width=True, key=f"manual_history_editor_{dataset}")
            template_csv = pd.DataFrame(columns=manual_cols).to_csv(index=False).encode("utf-8")
            st.download_button("Download manual input CSV template", template_csv, file_name="aerolife_manual_history_template.csv", mime="text/csv")
            if st.button("Predict from manual historical data", type="primary", key="predict_manual_history"):
                try:
                    manual = edited.copy()
                    manual = manual.dropna(how="all")
                    if manual.empty:
                        raise ValueError("Enter at least one complete historical cycle.")
                    for col in manual_cols:
                        manual[col] = pd.to_numeric(manual[col], errors="coerce")
                    if manual[manual_cols].isna().any().any() or not np.isfinite(manual[manual_cols].to_numpy(float)).all():
                        raise ValueError("Every entered cycle must include finite numeric values for cycle, all 3 operating settings, and all 21 sensors. Remove incomplete rows or fill every required cell.")
                    if (manual["cycle"] < 1).any() or not (manual["cycle"] % 1 == 0).all():
                        raise ValueError("Cycle values must be positive whole numbers.")
                    manual = manual.sort_values("cycle").reset_index(drop=True)
                    expected_cycles = np.arange(1, len(manual) + 1)
                    if not np.array_equal(manual["cycle"].to_numpy(dtype=int), expected_cycles):
                        raise ValueError("Historical cycles must start at 1 with no gaps or duplicates.")
                    manual.insert(0, "unit", int(engine_id_manual))
                    manual = manual[RAW_COLS]
                    predictor = load_predictor(dataset)
                    with st.spinner("Calculating historical features and running the frozen R1 prediction pipeline..."):
                        result = predictor.predict(manual)
                    st.subheader("Manual historical prediction result")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.metric("Predicted Hybrid RUL", f"{float(result.iloc[0].predicted_rul):.2f} cycles")
                    st.download_button("Download manual prediction (CSV)", result.to_csv(index=False).encode("utf-8"), file_name=f"aerolife_manual_{dataset}_engine_{int(engine_id_manual)}.csv", mime="text/csv")
                except Exception as exc:
                    st.error(f"Manual historical prediction could not be generated: {exc}")

with st.expander("Dataset metrics and model notes"):
    metric_rows = [{"Dataset": d, "Test engines": m["engines"], "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"], "NASA Score": m["nasa"]} for d,m in THESIS_TEST_METRICS.items()]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
    st.markdown("**Frozen model:** CatBoost + XGBoost + LightGBM weighted hybrid. No Extra Trees or ARCHE model is included in this R1 artifact set.")
    st.markdown("**Intervals:** empirical residual-based 90% intervals from saved artifacts; not formal conformal coverage guarantees.")
    st.markdown("**Evaluation:** historical thesis test-set metrics are displayed as reported; this web app does not recompute them from the selected engine.")
    st.markdown("**Ground truth:** matching NASA RUL_FD001–RUL_FD004 label files are bundled and automatically matched to the selected test subset and engine ID.")

st.divider()
st.caption("AeroLife · Academic research and demonstration · Frozen R1 artifacts · Not certified for operational aircraft maintenance")
