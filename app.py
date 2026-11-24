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

# Primary-model regression metrics are strict nested-validation estimates.
# The NASA score is a separate historical official-test reference for the
# Historical Baseline Reference; it is not attributed to the active predictor.
RESEARCH_SELECTION_METRICS = {
    "FD001": {"engines": 100, "r2": 0.905409, "rmse": 17.744212, "mae": 11.556675},
    "FD002": {"engines": 260, "r2": 0.883943, "rmse": 19.623551, "mae": 13.133444},
    "FD003": {"engines": 100, "r2": 0.926304, "rmse": 16.341627, "mae": 10.958765},
    "FD004": {"engines": 249, "r2": 0.881011, "rmse": 20.815945, "mae": 13.741695},
}

FROZEN_R1_OFFICIAL_NASA = {
    "FD001": {"test_engines": 100, "nasa": 506.273},
    "FD002": {"test_engines": 259, "nasa": 2951.298},
    "FD003": {"test_engines": 100, "nasa": 863.645},
    "FD004": {"test_engines": 248, "nasa": 2731.377},
}

# Unweighted arithmetic means of the four dataset-specific values shown below.
# They are descriptive averages, not pooled-engine metrics.
PRIMARY_MODEL_AVERAGES = {
    "r2": float(np.mean([v["r2"] for v in RESEARCH_SELECTION_METRICS.values()])),
    "rmse": float(np.mean([v["rmse"] for v in RESEARCH_SELECTION_METRICS.values()])),
    "mae": float(np.mean([v["mae"] for v in RESEARCH_SELECTION_METRICS.values()])),
}
HISTORICAL_BASELINE_MEAN_NASA = float(
    np.mean([v["nasa"] for v in FROZEN_R1_OFFICIAL_NASA.values()])
)

st.markdown("""
<style>
:root {
  --al-ink: #13243a;
  --al-muted: #63758a;
  --al-blue: #176b87;
  --al-cyan: #42a5b8;
  --al-line: #dce6ef;
  --al-surface: #ffffff;
}
.block-container {padding-top: 1.15rem; padding-bottom: 2.4rem; max-width: 1480px;}
[data-testid="stAppViewContainer"] {background: #f5f8fc;}
[data-testid="stHeader"] {background: rgba(245,248,252,.86);}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#10243b 0%,#14364c 58%,#17465a 100%); border-right: 1px solid rgba(255,255,255,.10); min-width: 285px !important; width: 285px !important;}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarExpandButton"] {display:none !important;}
[data-testid="stSidebar"] button[aria-label*="Collapse"], [data-testid="stSidebar"] button[aria-label*="Expand"] {display:none !important;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {color:#eef6fb;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#d8e6ef;}
[data-testid="stSidebar"] [data-testid="stRadio"] label {color:#edf5fa !important;}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {font-weight:700; letter-spacing:.04em;}
.sidebar-brand {padding: .6rem .15rem 1rem .15rem; border-bottom:1px solid rgba(255,255,255,.16); margin-bottom:1rem;}
.sidebar-brand .mark {font-size:1.75rem; font-weight:800; letter-spacing:.04em; color:#fff;}
.sidebar-brand .tag {font-size:.77rem; color:#b9d2df; line-height:1.45;}
.sidebar-card {padding:.85rem .9rem; margin:.6rem 0; border:1px solid rgba(255,255,255,.16); border-radius:13px; background:rgba(255,255,255,.07);}
.sidebar-card .k {font-size:.68rem; text-transform:uppercase; letter-spacing:.09em; color:#a9c7d7;}
.sidebar-card .v {font-size:.9rem; font-weight:650; color:#fff; margin-top:.18rem;}
.profile-role {font-size:.78rem; font-weight:650; color:#e5f1f6; margin-top:.24rem;}
.profile-sub {font-size:.72rem; line-height:1.4; color:#c2d7e2; margin-top:.22rem;}
.hero {padding:1.35rem 1.55rem; border-radius:20px; background:linear-gradient(112deg,#10243b 0%,#174e68 62%,#247f91 100%); color:white; margin:.15rem 0 1rem 0; box-shadow:0 12px 30px rgba(18,55,78,.13);}
.hero .eyebrow {font-size:.72rem; letter-spacing:.14em; text-transform:uppercase; color:#b8d9e6; font-weight:750; margin-bottom:.45rem;}
.hero h1 {font-size:2.05rem; line-height:1.13; margin:0 0 .42rem 0; color:white;}
.hero p {margin:0; color:#e0eef4; font-size:.96rem;}
.hero .hero-foot {margin-top:.9rem; font-size:.78rem; color:#b9d8e4;}
.section-kicker {font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:#24748a; font-weight:800; margin-bottom:.15rem;}
.page-title {font-size:1.45rem; font-weight:780; color:var(--al-ink); margin:.2rem 0 .25rem 0;}
.page-subtitle {color:var(--al-muted); margin-bottom:1.15rem;}
.small-note {font-size:.84rem; color:var(--al-muted);}
div[data-testid="stMetric"] {background:#fff; border:1px solid var(--al-line); border-radius:15px; padding:.8rem .95rem; box-shadow:0 3px 10px rgba(25,55,80,.035);}
div[data-testid="stMetricLabel"] p {color:#62758b; font-size:.78rem; font-weight:650;}
div[data-testid="stMetricValue"] {color:#142c45; font-weight:780;}
div.stButton > button, div[data-testid="stDownloadButton"] > button {border-radius:11px; font-weight:650; min-height:2.6rem;}
div.stButton > button[kind="primary"] {background:linear-gradient(100deg,#176b87,#218ca0); border:0;}
div[data-testid="stExpander"] {border:1px solid var(--al-line); border-radius:13px; background:#fff;}
[data-testid="stDataFrame"] {border:1px solid var(--al-line); border-radius:12px; overflow:hidden;}
hr {border-color:#dce6ef;}
@media (max-width: 720px) {
  .block-container {padding: .7rem .75rem 1.8rem .75rem;}
  .hero {padding:1.15rem 1.1rem; border-radius:15px;}
  .hero h1 {font-size:1.65rem;}
  [data-testid="stSidebar"] {min-width: 255px !important; width: 255px !important;}
}
</style>
<div class="hero">
  <div class="eyebrow">Predictive Maintenance · Research Intelligence</div>
  <h1>✈️ AeroLife</h1>
  <p>Aircraft engine Remaining Useful Life prediction with a weighted ensemble and explainable AI.</p>
  <div class="hero-foot">NASA C-MAPSS · FD001–FD004 · CatBoost + XGBoost + LightGBM</div>
</div>
""", unsafe_allow_html=True)

# Persistent navigation sidebar: expanded on load with collapse controls hidden.
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <div class="mark">✈ AeroLife</div>
      <div class="tag">Aircraft Engine Health Intelligence<br/>Research dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATION",
        ["Overview", "NASA Benchmark Explorer", "Custom Data Prediction", "About AeroLife"],
        index=0,
        key="aerolife_navigation",
    )
    st.markdown("""
    <div class="sidebar-card">
      <div class="k">Active deployment</div>
      <div class="v">Primary Research Model</div>
    </div>
    <div class="sidebar-card">
      <div class="k">Benchmark scope</div>
      <div class="v">4 NASA C-MAPSS subsets</div>
    </div>
    <div class="sidebar-card">
      <div class="k">Researcher</div>
      <div class="v">MD FAHIM MUNTASIR LEON</div>
      <div class="profile-sub">Dept. of CSE, Daffodil International University</div>
    </div>
    <div class="sidebar-card">
      <div class="k">Research Supervisor</div>
      <div class="v">FATEMA TUJ JOHORA</div>
      <div class="profile-role">Asst. Professor</div>
      <div class="profile-sub">Dept. of CSE, Daffodil International University</div>
    </div>
    <div class="tag" style="font-size:.72rem;padding:.65rem .15rem 0 .15rem;">
      Academic research prototype · Not for operational maintenance decisions
    </div>
    """, unsafe_allow_html=True)

st.caption("The Primary Research Model powers live RUL predictions · NASA C-MAPSS simulated benchmark data")
st.warning("Academic research prototype. NASA C-MAPSS is simulated benchmark data, not live aircraft telemetry. Outputs are not certified or intended as standalone operational maintenance instructions.")

@st.cache_resource(show_spinner="Loading the Primary Research Model...")
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

def apply_public_model_label(results: pd.DataFrame) -> pd.DataFrame:
    """Use the public-facing model name in rendered and downloaded results."""
    output = results.copy()
    if "ensemble" in output.columns:
        output["ensemble"] = "Primary Research Model · Weighted Ensemble"
    return output


def render_metrics(dataset: str):
    research = RESEARCH_SELECTION_METRICS[dataset]
    official = FROZEN_R1_OFFICIAL_NASA[dataset]
    st.subheader("Dataset Performance")
    st.caption("R², RMSE and MAE are strict nested-validation estimates for the Primary Research Model. NASA Score is the historical official-test reference for the Historical Baseline Reference; it is not produced by the active predictor.")
    cols = st.columns(5)
    cols[0].metric("Validation engines", f"{research['engines']:,}")
    cols[1].metric("MAE · Primary Model", f"{research['mae']:.3f} cycles")
    cols[2].metric("RMSE · Primary Model", f"{research['rmse']:.3f} cycles")
    cols[3].metric("R² · Primary Model", f"{research['r2']:.4f}")
    cols[4].metric("NASA Score · Historical Reference", f"{official['nasa']:,.1f}", help=f"Historical official NASA test score for the Historical Baseline Reference on {official['test_engines']} official test engines. It is not a score for the Primary Research Model.")
    st.caption(f"Historical reference population: {official['test_engines']} official test engines · Primary-model nested-validation population: {research['engines']} development engines.")

def render_engine_report(dataset: str, engine_id: int, history: pd.DataFrame, predictor: RULPredictor, actual_rul=None, show_shap: bool = True):
    pred = apply_public_model_label(predictor.predict(history))
    row = pred.iloc[0]
    st.markdown(f"## Aircraft Engine Health Report — {dataset}")
    st.caption(f"Engine ID: {engine_id} · Latest observed cycle: {int(row.last_cycle)} · Model: {row.ensemble}")
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
            st.caption("SHAP contributions combine the three tree models using the saved ensemble weights. They explain the weighted tree outputs before the runtime's member clipping and interval post-processing; they are feature attributions, not causal effects.")
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

if page == "Overview":
    st.markdown('<div class="section-kicker">Workspace overview</div><div class="page-title">Aircraft Health Intelligence</div><div class="page-subtitle">A unified workspace for benchmark exploration, RUL prediction and explainable model analysis.</div>', unsafe_allow_html=True)
    overview_cols = st.columns(4)
    overview_cols[0].metric("Benchmark subsets", "04")
    overview_cols[1].metric("Development engines", "709")
    overview_cols[2].metric("Official test engines", "707")
    overview_cols[3].metric("Sensor channels", "21")

    st.markdown("### Research Performance Snapshot")
    st.caption("Unweighted arithmetic averages across FD001–FD004. R², RMSE and MAE summarise strict nested validation for the Primary Research Model; NASA Score is a separate historical official-test reference for the Historical Baseline Reference.")
    performance_cols = st.columns(4)
    performance_cols[0].metric("Mean R²", f"{PRIMARY_MODEL_AVERAGES['r2']:.4f}")
    performance_cols[1].metric("Mean RMSE", f"{PRIMARY_MODEL_AVERAGES['rmse']:.3f} cycles")
    performance_cols[2].metric("Mean MAE", f"{PRIMARY_MODEL_AVERAGES['mae']:.3f} cycles")
    performance_cols[3].metric("Mean NASA Score", f"{HISTORICAL_BASELINE_MEAN_NASA:,.1f}", help="Unweighted mean of the four historical official-test NASA scores for the Historical Baseline Reference. Not a pooled-engine score and not a Primary Research Model score.")
    st.caption("The first three averages are calculated from the four dataset-level nested-validation estimates. The NASA average is calculated separately from the four historical official-test reference scores. Each is an unweighted mean across datasets, not a pooled-engine metric.")

    with st.expander("Dataset-level performance details", expanded=False):
        overview_rows = []
        for subset in SUBSETS:
            r = RESEARCH_SELECTION_METRICS[subset]
            f = FROZEN_R1_OFFICIAL_NASA[subset]
            overview_rows.append({
                "Dataset": subset,
                "Validation engines": r["engines"],
                "Primary Model R²": r["r2"],
                "Primary Model RMSE": r["rmse"],
                "Primary Model MAE": r["mae"],
                "Historical Reference NASA Score": f["nasa"],
                "Historical Reference Test Engines": f["test_engines"],
            })
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    st.markdown("### Explore AeroLife")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**NASA Benchmark Explorer**")
        st.write("Inspect the preloaded FD001–FD004 test engines, generate per-engine RUL reports, review sensor trends and explore SHAP explanations.")
        st.caption("Open this workspace from the persistent navigation sidebar.")
    with c2:
        st.markdown("**Custom Data Prediction**")
        st.write("Upload a CSV/NASA TXT file or enter historical sensor readings manually to generate predictions with the Primary Research Model.")
        st.caption("Open this workspace from the persistent navigation sidebar.")

elif page == "NASA Benchmark Explorer":
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

elif page == "Custom Data Prediction":
    st.markdown('<div class="section-kicker">Prediction workspace</div><div class="page-title">Custom Data Prediction</div><div class="page-subtitle">Run the Primary Research Model on uploaded files or complete historical sensor records.</div>', unsafe_allow_html=True)
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
                            with st.spinner("Running the Primary Research Model prediction pipeline..."):
                                result = apply_public_model_label(predictor.predict(raw))
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
            st.info("A single observation can be entered here, but the Primary Research Model uses historical rolling, slope, drift, and health-index features. A valid prediction cannot be generated from one cycle alone. Enter the engine's full cycle history in Historical Cycle Input to generate a prediction.")
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
            st.button("Generate Prediction", disabled=True, help="A full contiguous history is required by the Primary Research Model feature pipeline.", key="single_cycle_blocked")

        else:
            st.write("Enter the engine's raw history from cycle 1 through its latest observed cycle. The saved feature pipeline will calculate historical features using the same saved preprocessing and feature logic as the trained models.")
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
                    with st.spinner("Calculating historical features and running the Primary Research Model prediction pipeline..."):
                        result = apply_public_model_label(predictor.predict(manual))
                    st.subheader("Manual historical prediction result")
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.metric("Predicted Hybrid RUL", f"{float(result.iloc[0].predicted_rul):.2f} cycles")
                    st.download_button("Download manual prediction (CSV)", result.to_csv(index=False).encode("utf-8"), file_name=f"aerolife_manual_{dataset}_engine_{int(engine_id_manual)}.csv", mime="text/csv")
                except Exception as exc:
                    st.error(f"Manual historical prediction could not be generated: {exc}")

elif page == "About AeroLife":
    st.markdown('<div class="section-kicker">Project information</div><div class="page-title">About AeroLife</div><div class="page-subtitle">An explainable machine-learning research prototype for turbofan engine prognostics.</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero" style="margin-top:.6rem;">
      <div class="eyebrow">Research project</div>
      <h1 style="font-size:1.55rem;">Aircraft Engine Predictive Maintenance</h1>
      <p>Remaining Useful Life estimation using a hybrid ensemble machine-learning framework with Explainable Artificial Intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    about_left, about_right = st.columns([1.15, .85], gap="large")
    with about_left:
        st.markdown("### Project overview")
        st.write("AeroLife investigates data-driven Remaining Useful Life (RUL) estimation for turbofan engines using the NASA C-MAPSS benchmark. It combines complementary tree-based learners and explainability methods to make model predictions easier to inspect and interpret.")
        st.markdown("### Methodology")
        st.write("The Primary Research Model uses a weighted ensemble of CatBoost, XGBoost and LightGBM, supported by sensor-history feature engineering and SHAP-based explanations. The dashboard provides benchmark exploration, custom-data inference, sensor trend analysis and downloadable engine reports.")
        st.markdown("### Dataset and scope")
        st.write("The study uses NASA C-MAPSS subsets FD001, FD002, FD003 and FD004, including three operating settings and 21 sensor channels. Benchmark data are simulated and are used here for academic research and demonstration.")
    with about_right:
        st.markdown("### Researcher")
        st.markdown("""
        <div class="sidebar-card" style="background:#fff;border:1px solid #dce6ef;">
          <div class="k" style="color:#61758b;">Researcher</div>
          <div class="v" style="color:#142c45;font-size:1.05rem;">MD FAHIM MUNTASIR LEON</div>
          <div style="font-size:.78rem;line-height:1.45;color:#61758b;margin-top:.28rem;">Dept. of CSE, Daffodil International University</div>
        </div>
        <div class="sidebar-card" style="background:#fff;border:1px solid #dce6ef;">
          <div class="k" style="color:#61758b;">Research Supervisor</div>
          <div class="v" style="color:#142c45;font-size:1.0rem;">FATEMA TUJ JOHORA</div>
          <div style="font-size:.82rem;font-weight:650;color:#334b63;margin-top:.24rem;">Asst. Professor</div>
          <div style="font-size:.78rem;line-height:1.45;color:#61758b;margin-top:.18rem;">Dept. of CSE, Daffodil International University</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("### Model information")
        st.markdown("""
        - **Primary model:** Weighted ensemble of CatBoost, XGBoost and LightGBM
        - **Algorithms:** CatBoost · XGBoost · LightGBM
        - **Explainability:** SHAP
        - **Prediction target:** Remaining Useful Life (cycles)
        """)
    st.warning("Research prototype only. AeroLife is not a medical, aviation certification or operational aircraft-maintenance system. Predictions must not be used as standalone maintenance decisions. Unusual inputs can produce unreliable estimates, and benchmark performance does not guarantee real-world performance.")

if page in ("NASA Benchmark Explorer", "Overview"):
    with st.expander("Dataset metrics and model notes"):
        metric_rows = []
        for d in SUBSETS:
            research = RESEARCH_SELECTION_METRICS[d]
            official = FROZEN_R1_OFFICIAL_NASA[d]
            metric_rows.append({
                "Dataset": d,
                "Validation engines": research["engines"],
                "Primary Model MAE": research["mae"],
                "Primary Model RMSE": research["rmse"],
                "Primary Model R²": research["r2"],
                "Historical Reference NASA Score": official["nasa"],
                "Historical Reference Test Engines": official["test_engines"],
            })
        st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)
        st.markdown("**Primary prediction system:** CatBoost + XGBoost + LightGBM weighted ensemble. The Historical Baseline Reference is retained separately for historical benchmark comparison.")
        st.markdown("**Metric provenance:** R², RMSE and MAE are strict nested-validation results for the Primary Research Model. NASA Score is the historical official-test score for the Historical Baseline Reference; it is not attributed to the Primary Research Model.")
        st.markdown("**Prediction rule:** NASA Score evaluates labelled benchmark predictions; it does not generate or adjust an engine's predicted RUL. Custom data cannot receive a NASA Score unless matching ground-truth RUL labels are independently available.")

st.divider()
st.caption("AeroLife · Researcher: MD FAHIM MUNTASIR LEON · Research Supervisor: FATEMA TUJ JOHORA · Academic research prototype, not certified for operational aircraft maintenance")
