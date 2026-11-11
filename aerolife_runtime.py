"""AeroLife R1 runtime extracted from the supplied thesis notebook.
Prediction preprocessing and post-processing are kept aligned with notebook RULPredictor.
"""
from __future__ import annotations
import os, sys, json, hashlib
from pathlib import Path
from dataclasses import dataclass, field
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge

# Library (definitions only, no training): imports and constants
"""AeroLife core library: data loading, causal feature engineering,
engine-level cross-validation and metrics for NASA C-MAPSS RUL prediction.

Every feature at cycle t uses only cycles 1..t of the same engine, so the
same code produces training rows, out-of-fold rows and the single last-cycle
row used for each official test engine.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
OP_COLS = ["os1", "os2", "os3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
RAW_COLS = ["unit", "cycle"] + OP_COLS + SENSOR_COLS
# Sensors that carry degradation information. s1, s5, s6, s16, s18 and s19 are
# constant (or quantised noise) within an operating condition in FD001/FD003,
# so they are excluded a priori from domain knowledge, not from labels.
USEFUL_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s10", "s11", "s12",
                  "s13", "s14", "s15", "s17", "s20", "s21"]
N_REGIMES = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
N_FAULT_MODES = {"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2}

# Library (definitions only, no training): fold-fitted preprocessing: operating regimes and per-regime normalisation
@dataclass
class RegimeNormalizer:
    """Clusters operating settings into regimes and z-scores each sensor within
    its regime. Fitted on training-fold rows only."""
    n_regimes: int
    sensors: list = field(default_factory=lambda: list(USEFUL_SENSORS))
    seed: int = 0

    def fit(self, df: pd.DataFrame) -> "RegimeNormalizer":
        ops = df[OP_COLS].to_numpy()
        if self.n_regimes > 1:
            self.km_ = KMeans(n_clusters=self.n_regimes, n_init=10, random_state=self.seed).fit(ops)
            reg = self.km_.labels_
        else:
            self.km_ = None
            reg = np.zeros(len(df), dtype=int)
        x = df[self.sensors].to_numpy(dtype=float)
        self.mu_ = np.zeros((self.n_regimes, len(self.sensors)))
        self.sd_ = np.ones((self.n_regimes, len(self.sensors)))
        for r in range(self.n_regimes):
            xr = x[reg == r]
            self.mu_[r] = xr.mean(0)
            sd = xr.std(0)
            self.sd_[r] = np.where(sd > 1e-8, sd, 1.0)
        return self

    def regime(self, df: pd.DataFrame) -> np.ndarray:
        if self.km_ is None:
            return np.zeros(len(df), dtype=int)
        return self.km_.predict(df[OP_COLS].to_numpy())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        reg = self.regime(df)
        x = df[self.sensors].to_numpy(dtype=float)
        z = (x - self.mu_[reg]) / self.sd_[reg]
        out = df[["unit", "cycle"] + (["rul"] if "rul" in df else [])].copy()
        out["regime"] = reg
        for j, s in enumerate(self.sensors):
            out[s] = z[:, j]
        return out

# Library (definitions only, no training): causal window features (computed per engine, past cycles only)
def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    n = len(x)
    idx = np.arange(1, n + 1)
    lo = np.maximum(idx - w, 0)
    cnt = (idx - lo)[:, None]
    return (c[idx] - c[lo]) / cnt


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    m = _rolling_mean(x, w)
    m2 = _rolling_mean(x * x, w)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def _rolling_slope(x: np.ndarray, w: int) -> np.ndarray:
    """OLS slope of x against cycle index over the trailing window (0 for one point)."""
    n = len(x)
    t = np.arange(n, dtype=float)[:, None]
    mt, my = _rolling_mean(t, w), _rolling_mean(x, w)
    mty, mtt = _rolling_mean(t * x, w), _rolling_mean(t * t, w)
    var = mtt - mt * mt
    return np.where(var > 1e-12, (mty - mt * my) / np.where(var > 1e-12, var, 1.0), 0.0)


def _ewm(x: np.ndarray, alpha: float) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def _expanding_mean(x: np.ndarray) -> np.ndarray:
    return np.cumsum(x, axis=0) / np.arange(1, len(x) + 1)[:, None]


@dataclass
class FeatureConfig:
    name: str = "full"
    windows: tuple = (5, 10, 20, 30)
    slope_windows: tuple = (10, 20, 50)
    ewm_alphas: tuple = (0.1, 0.3)
    baseline_cycles: int = 10
    raw_only: bool = False
    health_index: bool = True
    fault_axis: bool = False


def engine_features(z: pd.DataFrame, sensors: list, cfg: FeatureConfig) -> pd.DataFrame:
    """z: normalised sensors for one or more engines, sorted by unit, cycle."""
    blocks = []
    for _, g in z.groupby("unit", sort=False):
        x = g[sensors].to_numpy(dtype=float)
        feats = {"cycle": g["cycle"].to_numpy(dtype=float)}
        for j, s in enumerate(sensors):
            feats[s] = x[:, j]
        if not cfg.raw_only:
            for w in cfg.windows:
                m, sd = _rolling_mean(x, w), _rolling_std(x, w)
                for j, s in enumerate(sensors):
                    feats[f"{s}_mean{w}"] = m[:, j]
                    if w in (10, 30):
                        feats[f"{s}_std{w}"] = sd[:, j]
            for w in cfg.slope_windows:
                sl = _rolling_slope(x, w)
                for j, s in enumerate(sensors):
                    feats[f"{s}_slope{w}"] = sl[:, j]
            for a in cfg.ewm_alphas:
                e = _ewm(x, a)
                for j, s in enumerate(sensors):
                    feats[f"{s}_ewm{a}"] = e[:, j]
            # Engine-relative drift: smoothed current level minus the engine's own
            # early-life baseline (first k cycles, or fewer if not yet observed).
            k = cfg.baseline_cycles
            base = _expanding_mean(x)
            base[k:] = base[k - 1] if len(x) >= k else base[k:]
            cur = _rolling_mean(x, 10)
            exp_mean = _expanding_mean(x)
            for j, s in enumerate(sensors):
                feats[f"{s}_drift"] = cur[:, j] - base[:, j]
                feats[f"{s}_expmean"] = exp_mean[:, j]
        blocks.append(pd.DataFrame(feats, index=g.index))
    return pd.concat(blocks).loc[z.index]


class FeaturePipeline:
    """Fold-safe feature builder: fit() learns regime normalisation and an
    optional linear health index on training rows; transform() is causal."""

    def __init__(self, subset_regimes: int, cfg: FeatureConfig, cap: float = 125.0, seed: int = 0):
        self.cfg, self.cap, self.seed = cfg, cap, seed
        self.norm = RegimeNormalizer(subset_regimes, seed=seed)
        self.sensors = list(USEFUL_SENSORS)

    def fit(self, train_df: pd.DataFrame) -> "FeaturePipeline":
        self.norm.fit(train_df)
        if self.cfg.health_index and not self.cfg.raw_only:
            z = self.norm.transform(train_df)
            # Health index: ridge projection of smoothed sensors onto the capped
            # target, scaled to roughly [0, 1] (1 = healthy). Fitted on train fold only.
            sm = self._smooth(z)
            y = np.minimum(train_df["rul"].to_numpy(dtype=float), self.cap) / self.cap
            self.hi_ = Ridge(alpha=1.0).fit(sm, y)
        if self.cfg.fault_axis:
            # Fault-mode axis (FD003/FD004 have two fault modes). Unsupervised:
            # cluster training-fold engines by their end-of-life sensor drift and
            # keep the unit vector between the two centroids. No RUL labels used.
            z = self.norm.transform(train_df)
            g = z.groupby("unit")
            drift = (g.tail(10).groupby("unit")[self.sensors].mean()
                     - g.head(10).groupby("unit")[self.sensors].mean()).to_numpy()
            km = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit(drift)
            v = km.cluster_centers_[1] - km.cluster_centers_[0]
            # orient the axis so the result does not depend on cluster labelling
            self.fault_dir_ = v / np.linalg.norm(v) * np.sign(v[np.argmax(np.abs(v))])
        return self

    def _smooth(self, z: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(z), len(self.sensors)))
        pos = {i: k for k, i in enumerate(z.index)}
        for _, g in z.groupby("unit", sort=False):
            m = _rolling_mean(g[self.sensors].to_numpy(dtype=float), 10)
            out[[pos[i] for i in g.index]] = m
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        z = self.norm.transform(df)
        F = engine_features(z, self.sensors, self.cfg)
        if self.norm.n_regimes > 1:
            F["regime"] = z["regime"].to_numpy()
        if self.cfg.health_index and not self.cfg.raw_only:
            hi_inst = self.hi_.predict(z[self.sensors].to_numpy(dtype=float))
            tmp = pd.DataFrame({"unit": df["unit"].to_numpy(), "hi": hi_inst}, index=df.index)
            hi_cols = {k: np.empty(len(df)) for k in ["hi", "hi_mean10", "hi_mean30", "hi_slope20", "hi_slope50", "hi_ewm0.1", "hi_drift"]}
            pos = np.arange(len(df))
            for _, g in tmp.groupby("unit", sort=False):
                ii = pos[df.index.get_indexer(g.index)]
                h = g["hi"].to_numpy()[:, None]
                hi_cols["hi"][ii] = h[:, 0]
                hi_cols["hi_mean10"][ii] = _rolling_mean(h, 10)[:, 0]
                hi_cols["hi_mean30"][ii] = _rolling_mean(h, 30)[:, 0]
                hi_cols["hi_slope20"][ii] = _rolling_slope(h, 20)[:, 0]
                hi_cols["hi_slope50"][ii] = _rolling_slope(h, 50)[:, 0]
                hi_cols["hi_ewm0.1"][ii] = _ewm(h, 0.1)[:, 0]
                b = _expanding_mean(h)
                if len(h) >= 10:
                    b[10:] = b[9]
                hi_cols["hi_drift"][ii] = (_rolling_mean(h, 10) - b)[:, 0]
            for k, v in hi_cols.items():
                F[k] = v
        if self.cfg.fault_axis:
            D = F[[f"{s}_drift" for s in self.sensors]].to_numpy()
            proj = D @ self.fault_dir_
            F["fault_axis"] = proj
            F["fault_axis_ratio"] = proj / (np.linalg.norm(D, axis=1) + 1e-6)
        return F

# Library (definitions only, no training): fixed study configuration (declared before any new experiment was run)
LONG = FeatureConfig(name="long_windows", windows=(5, 10, 20, 30, 50), slope_windows=(10, 20, 50, 100))
HI_CAP = 130.0          # cap used only to fit the ridge health-index feature (feature hyper-parameter)
N_FOLDS = 5
DS_CODE = {s: i + 1 for i, s in enumerate(SUBSETS)}
PROXY_MAX_RUL = 200     # primary selection population: truncation RUL uniform-ish in [1, 200]
PROXY_MIN_CYCLE = 20
# Target options screened in Stage 1: piecewise-linear caps, and "T200", the uncapped RUL
# learned only from rows inside the 200-cycle prognostic horizon (population-matched target).
CAP_GRID = (130.0, 145.0, 160.0, 175.0, 200.0, "T200")

# One fixed configuration per family. All configurations come from the earlier
# campaign (aerolife_core / campaign.py); none was re-tuned for this study.
FAMILY_SPECS = {
    "CAT": dict(name="CatBoost", family="CatBoost", maker="catboost",
                params=dict(depth=8, iterations=3000, learning_rate=0.03, l2_leaf_reg=3.0, rsm=0.5),
                origin="configuration of earlier experiment Q8 (large CatBoost)"),
    "XGB": dict(name="XGBoost", family="XGBoost", maker="xgb",
                params=dict(n_estimators=600, learning_rate=0.03, max_depth=6, min_child_weight=20,
                            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0),
                origin="configuration of earlier experiments M2 / Q7"),
    "LGB": dict(name="LightGBM", family="LightGBM", maker="lgbm",
                params=dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                            subsample=0.8, colsample_bytree=0.5, reg_lambda=1.0),
                origin="configuration of earlier experiment F1"),
    "ET": dict(name="Extra Trees", family="Extra Trees", maker="et",
               params=dict(n_estimators=300, min_samples_leaf=10, max_features=0.3),
               origin="configuration of earlier experiment M3"),
    "RF": dict(name="Random Forest", family="Random Forest", maker="rf",
               params=dict(n_estimators=200, min_samples_leaf=10, max_features=0.3, max_samples=0.5),
               origin="earlier experiment B0 used 300 trees / max_features 0.5 on 24 raw inputs; on 253 features "
                      "that costs >20 min per fold, so trees, feature fraction and bootstrap size were reduced "
                      "for feasibility before any result was seen (not tuned)"),
}
SCREEN_FAMILIES = ("XGB", "LGB")   # fast families used to choose the target cap


def make_model(key: str, seed: int = 0, n_jobs: int | None = None):
    spec = FAMILY_SPECS[key]
    nj = n_jobs or int(os.environ.get("AEROLIFE_N_JOBS", os.cpu_count() or 2))
    p = dict(spec["params"])
    if spec["maker"] == "catboost":
        return make_catboost(seed, thread_count=nj, **p)
    if spec["maker"] == "xgb":
        return make_xgb(seed, n_jobs=nj, **p)
    if spec["maker"] == "lgbm":
        return make_lgbm(seed, n_jobs=nj, **p)
    if spec["maker"] == "et":
        return make_extratrees(seed, n_jobs=nj, **p)
    if spec["maker"] == "rf":
        return make_rf(seed, n_jobs=nj, **p)
    raise KeyError(key)


def cap_tag(cap) -> str:
    return cap if isinstance(cap, str) else str(int(cap))


def parse_cap(tag):
    tag = str(tag)
    return tag if tag.startswith("T") else float(tag)


def target_spec(cap):
    """(clip_max, horizon) for a target option: a numeric cap c trains on min(RUL, c)
    over all rows; 'T<B>' trains on the uncapped RUL of rows with RUL <= B only."""
    if isinstance(cap, str) and cap.startswith("T"):
        b = float(cap[1:])
        return b, b
    return float(cap), None


def target_label(cap) -> str:
    if isinstance(cap, str) and cap.startswith("T"):
        return f"uncapped target, rows with RUL <= {cap[1:]}"
    return f"cap {int(cap)}"


def make_target(y, cap):
    """Returns (target vector, row mask) for training."""
    clip_max, horizon = target_spec(cap)
    if horizon is None:
        return np.minimum(y, clip_max), np.ones(len(y), dtype=bool)
    return y, y <= horizon


def member_id(key: str, cap) -> str:
    return f"{key}@{cap_tag(cap)}"


def member_label(mid: str) -> str:
    key, cap = mid.split("@")
    return f"{FAMILY_SPECS[key]['name']} ({target_label(parse_cap(cap))})"


def clip_max_of(mid: str) -> float:
    return target_spec(parse_cap(mid.split("@")[1]))[0]

# Library (definitions only, no training): features
class PooledFeatures:
    """Fold-safe feature builder for the pooled data: regime normalisation and
    ridge health index (aerolife_core.FeaturePipeline, long windows) fitted on
    training rows only, plus a one-hot dataset indicator (a known input at
    prediction time, never a label)."""

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.fp = FeaturePipeline(max(N_REGIMES.values()), LONG, HI_CAP, seed)

    def fit(self, df: pd.DataFrame) -> "PooledFeatures":
        self.fp.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        F = self.fp.transform(df)
        for s in SUBSETS:
            F[f"ds_{s}"] = (df["ds"].to_numpy() == s).astype(float)
        self.columns_ = list(F.columns)
        return F

# Library (definitions only, no training): prediction uncertainty (empirical, cross-fitted) and policy risk bands
PRED_BINS = (0, 25, 50, 75, 100, 125, 150, 1e9)


def _bin(pred):
    return np.digitize(pred, PRED_BINS[1:-1])


def interval_table(pred, y, levels=(0.8, 0.9)):
    """Empirical residual quantiles (true - predicted) within bins of the
    predicted RUL (Mondrian split-conformal style)."""
    b = _bin(pred)
    r = y - pred
    rows = []
    for k in range(len(PRED_BINS) - 1):
        rk = r[b == k]
        row = {"bin": k, "pred_from": PRED_BINS[k], "pred_to": PRED_BINS[k + 1], "n": int(len(rk))}
        for lv in levels:
            a = (1 - lv) / 2
            row[f"lo{int(lv*100)}"] = float(np.quantile(rk, a)) if len(rk) >= 30 else np.nan
            row[f"hi{int(lv*100)}"] = float(np.quantile(rk, 1 - a)) if len(rk) >= 30 else np.nan
        rows.append(row)
    t = pd.DataFrame(rows)
    # bins with too few rows borrow the nearest populated bin
    t = t.ffill().bfill()
    return t


def apply_interval(pred, table, level=0.9):
    b = _bin(pred)
    lo = table.set_index("bin")[f"lo{int(level*100)}"].reindex(b).to_numpy()
    hi = table.set_index("bin")[f"hi{int(level*100)}"].reindex(b).to_numpy()
    return np.maximum(pred + lo, 0), pred + hi


def crossfit_coverage(pred, y, fold, mask, levels=(0.8, 0.9)):
    """Coverage of the intervals when the residual table for fold k is built only
    from the other folds (honest estimate on training engines)."""
    rows = []
    for lv in levels:
        cov, width = [], []
        for k in np.unique(fold):
            tr, te = mask & (fold != k), mask & (fold == k)
            t = interval_table(pred[tr], y[tr], levels=(lv,))
            lo, hi = apply_interval(pred[te], t, lv)
            cov.append(((y[te] >= lo) & (y[te] <= hi)).astype(float))
            width.append(hi - lo)
        rows.append({"nominal": lv, "empirical_coverage": float(np.concatenate(cov).mean()),
                     "mean_width": float(np.concatenate(width).mean())})
    return pd.DataFrame(rows)


# Policy thresholds (an operational choice, NOT learned from data)
RISK_BANDS = [(0, 25, "Critical: plan removal now"), (25, 50, "High: schedule maintenance"),
              (50, 100, "Medium: increase monitoring"), (100, 1e9, "Low: routine operation")]


def risk_band(rul_value: float) -> str:
    for lo, hi, name in RISK_BANDS:
        if lo <= rul_value < hi:
            return name
    return RISK_BANDS[-1][2]

def fingerprint(*objs) -> str:
    h = hashlib.sha256()
    for o in objs:
        if isinstance(o, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(o, index=False).to_numpy().tobytes())
        elif isinstance(o, np.ndarray):
            h.update(np.ascontiguousarray(o).tobytes())
        else:
            h.update(json.dumps(o, sort_keys=True, default=str).encode())
    return h.hexdigest()[:20]



MIN_MEMBER_WEIGHT = 0.10
REVISION_ID = "R1-three-family-2026-11-06"
MIN_FINAL_FAMILIES = 3
ENSEMBLE_BUNDLE_VERSION = "ensemble-bundle-v2"   # v1 = 31 Oct bundles (no key); bump when the bundle format changes


def candidate_families(cid: str) -> set:
    return {FAMILY_SPECS[k]["family"] for k in ENSEMBLE_CANDIDATES[cid]["members"]}


def select_ensemble_min_families(cand: pd.DataFrame, min_families: int = MIN_FINAL_FAMILIES):
    """The declared rule of select_ensemble, restricted to learned-weight candidates
    with at least `min_families` distinct model families."""
    c = cand[cand.kind.str.startswith("ensemble candidate (learned")].copy()
    c = c[c["id"].map(lambda i: len(candidate_families(i)) >= min_families)]
    if c.empty:
        raise RuntimeError(f"no pre-declared candidate has >= {min_families} families")
    best = c.loc[c.p200_rmse.idxmin()]
    near = c[c.p200_rmse <= best.p200_rmse + best.p200_rmse_se]
    return near.sort_values(["n_members", "p200_rmse"]).iloc[0]["id"]


def _jsonable(o):
    return json.loads(json.dumps(o, default=str))


def ensemble_bundle_key(final_entry: dict, member_keys: dict) -> dict:
    """Everything that defines a deployed ensemble: dataset, candidate, target, every member's
    family, hyper-parameters and frozen weight, and the cache key of every fitted member
    artifact (which itself contains data, code, pipeline and library-version fingerprints)."""
    return _jsonable({
        "what": "ensemble_bundle", "bundle_version": ENSEMBLE_BUNDLE_VERSION, "revision": REVISION_ID,
        "code": CODE_VERSION, "dataset": final_entry["dataset"], "candidate_id": final_entry["candidate_id"],
        "target": cap_tag(final_entry["cap"]),
        "members": [{"member_id": m["member_id"], "family": m["family"], "hyper_parameters": m["hyper_parameters"],
                     "weight": m["weight"], "artifact_key": fingerprint(_jsonable(member_keys[m["member_id"]]))}
                    for m in final_entry["members"]]})


def validate_ensemble_bundle(bundle: dict, root: Path) -> None:
    """Raises ValueError unless the bundle is a current-format bundle whose members and weights
    equal the frozen selection and whose member artifacts on disk carry the recorded keys.
    A two-model bundle from the 31 Oct version is therefore never accepted."""
    ds = bundle.get("dataset")
    if bundle.get("bundle_version") != ENSEMBLE_BUNDLE_VERSION:
        raise ValueError(f"{ds}: stale ensemble bundle (version {bundle.get('bundle_version')!r}, expected "
                         f"{ENSEMBLE_BUNDLE_VERSION!r}); re-run Section 17")
    key = bundle.get("bundle_key") or {}
    if fingerprint(key) != bundle.get("bundle_fingerprint"):
        raise ValueError(f"{ds}: bundle fingerprint does not match its key")
    frozen = json.loads((Path(root) / bundle["frozen_file"]).read_text())["final"][ds]
    want = [(m["member_id"], m["weight"]) for m in frozen["members"]]
    have = [(m["member_id"], m["weight"]) for m in bundle["members"]]
    keyed = [(m["member_id"], m["weight"]) for m in key.get("members", [])]
    if not (want == have == keyed) or frozen["candidate_id"] != bundle["candidate_id"]:
        raise ValueError(f"{ds}: bundle members/weights {have} differ from the frozen selection {want}")
    w = np.array([x for _, x in have])
    if abs(w.sum() - 1) > 1e-9 or w.min() < MIN_MEMBER_WEIGHT - 1e-9:
        raise ValueError(f"{ds}: weights must be >= {MIN_MEMBER_WEIGHT} and sum to 1")
    if len({m["family"] for m in bundle["members"]}) < MIN_FINAL_FAMILIES:
        raise ValueError(f"{ds}: fewer than {MIN_FINAL_FAMILIES} model families")
    for m in key["members"]:
        meta = (Path(root) / bundle["artifacts"][m["member_id"]]).with_suffix(".meta.json")
        if fingerprint(json.loads(meta.read_text())["key"]) != m["artifact_key"]:
            raise ValueError(f"{ds}: artifact of {m['member_id']} was refitted or replaced after the bundle was written")




def _register_pickle_classes():
    """The supplied notebook serialized these classes from __main__; register exact classes for joblib compatibility."""
    main = sys.modules.get("__main__")
    if main is not None:
        for name in ("FeatureConfig", "RegimeNormalizer", "FeaturePipeline", "PooledFeatures"):
            setattr(main, name, globals()[name])

def resolve_project_root(root):

    root = Path(root).resolve()

    # ROOT is already the project root
    if (root / "artifacts" / "thesis").is_dir():
        return root

    # ROOT points to artifacts/thesis
    if root.name == "thesis" and root.parent.name == "artifacts":

        project_root = root.parent.parent

        if (project_root / "artifacts" / "thesis").is_dir():
            return project_root

    raise FileNotFoundError(
        f"Cannot resolve AeroLife project root from: {root}"
    )


# ============================================================
# 2. FROZEN PREDICTION INTERFACE
# ============================================================

class RULPredictor:

    """
    Predict RUL for engines of one C-MAPSS dataset
    using frozen R1 ensemble artifacts.

    Input:
        DataFrame with unit, cycle, os1-os3, s1-s21.

    Output:
        One row per engine containing:
        predicted RUL, empirical 90% interval and risk band.
    """

    REQUIRED = RAW_COLS

    def __init__(self, dataset: str, root: Path | None = None):

        if dataset not in SUBSETS:
            raise ValueError(
                f"dataset must be one of {SUBSETS}, got {dataset!r}"
            )

        self.dataset = dataset

        if root is None:
            root = Path(__file__).resolve().parent
        self.project_root = resolve_project_root(root)

        self.thesis_dir = (
            self.project_root / "artifacts" / "thesis"
        )

        # ----------------------------------------------------
        # Load frozen ensemble bundle
        # ----------------------------------------------------

        bundle_path = (
            self.thesis_dir
            / "models"
            / f"ensemble_{dataset}.json"
        )

        if not bundle_path.is_file():
            raise FileNotFoundError(
                f"Ensemble bundle not found: {bundle_path}"
            )

        b = json.loads(bundle_path.read_text())

        # IMPORTANT:
        # Validator receives PROJECT ROOT.
        validate_ensemble_bundle(b, self.project_root)

        self.bundle = b

        # ----------------------------------------------------
        # Resolve artifact paths
        # ----------------------------------------------------

        def resolve_artifact(relative_path):

            p = Path(relative_path)

            if p.is_absolute():
                return p

            # Bundle artifact paths are project-relative.
            return self.project_root / p

        # ----------------------------------------------------
        # Load feature pipeline
        # ----------------------------------------------------

        pipeline_path = resolve_artifact(
            b["artifacts"]["feature_pipeline"]
        )

        if not pipeline_path.is_file():
            raise FileNotFoundError(
                f"Feature pipeline not found: {pipeline_path}"
            )

        _register_pickle_classes()
        self.pipeline = joblib.load(pipeline_path)

        # ----------------------------------------------------
        # Load frozen ensemble members
        # ----------------------------------------------------

        self.members = []

        for m in b["members"]:

            member_id = m["member_id"]

            model_path = resolve_artifact(
                b["artifacts"][member_id]
            )

            if not model_path.is_file():
                raise FileNotFoundError(
                    f"Model artifact not found for {member_id}: "
                    f"{model_path}"
                )

            model = joblib.load(model_path)

            self.members.append(
                (
                    float(m["weight"]),
                    clip_max_of(member_id),
                    model
                )
            )

        # ----------------------------------------------------
        # Load interval table
        # ----------------------------------------------------

        interval_path = (
            self.thesis_dir
            / "results"
            / f"interval_table_{dataset}.csv"
        )

        if not interval_path.is_file():
            raise FileNotFoundError(
                f"Interval table not found: {interval_path}"
            )

        self.intervals = pd.read_csv(interval_path)

    # ========================================================
    # 3. INPUT VALIDATION
    # ========================================================

    def _validate(self, h: pd.DataFrame) -> pd.DataFrame:

        if not isinstance(h, pd.DataFrame):
            raise TypeError(
                "history must be a pandas DataFrame"
            )

        missing = [
            c for c in self.REQUIRED
            if c not in h.columns
        ]

        if missing:
            raise ValueError(
                f"missing columns: {missing}"
            )

        h = h[self.REQUIRED].copy()

        bad = h.columns[h.isna().any()].tolist()

        if bad:
            raise ValueError(
                f"missing values in columns {bad}; "
                "impute or drop those cycles explicitly before predicting"
            )

        if not np.isfinite(
            h[OP_COLS + SENSOR_COLS].to_numpy(float)
        ).all():

            raise ValueError(
                "non-finite sensor or setting values"
            )

        for u, g in h.groupby("unit"):

            c = g["cycle"].to_numpy()

            if not np.array_equal(
                np.sort(c),
                np.arange(1, len(c) + 1)
            ):

                raise ValueError(
                    f"engine {u}: cycles must be contiguous "
                    f"and start at 1 "
                    f"(got {c.min()}..{c.max()}, n={len(c)})"
                )

        return (
            h.sort_values(["unit", "cycle"])
             .reset_index(drop=True)
        )

    # ========================================================
    # 4. PREDICTION
    # ========================================================

    def latest_feature_matrix(self, history: pd.DataFrame):
        """Return the exact frozen-pipeline feature row for each engine's latest cycle."""
        h = self._validate(history)
        h.insert(0, "ds", self.dataset)
        h["engine"] = h["unit"]
        h["unit"] = DS_CODE[self.dataset] * 1000 + h["unit"]
        F = self.pipeline.transform(h)
        last = h.groupby("unit").tail(1).index
        X_frame = F.loc[last].copy()
        X = np.ascontiguousarray(X_frame.to_numpy(np.float32))
        engine_ids = h.loc[last, "engine"].to_numpy()
        cycles = h.loc[last, "cycle"].to_numpy()
        return X, list(X_frame.columns), engine_ids, cycles

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        X, _, engine_ids, cycles = self.latest_feature_matrix(history)
        member_predictions = {}
        p = np.zeros(len(X), dtype=float)
        for weight, cap, model in self.members:
            member_name = model.__class__.__module__.split(".")[0]
            if member_name == "catboost":
                label = "catboost_prediction"
            elif member_name == "xgboost":
                label = "xgboost_prediction"
            elif member_name == "lightgbm":
                label = "lightgbm_prediction"
            else:
                label = f"{member_name}_prediction"
            component = np.clip(model.predict(X), 0, cap)
            member_predictions[label] = component
            p += weight * component

        lo, hi = apply_interval(p, self.intervals, 0.9)
        out = pd.DataFrame({
            "dataset": self.dataset,
            "engine": engine_ids,
            "last_cycle": cycles,
            **member_predictions,
            "predicted_rul": p,
            "rul_lower_90": lo,
            "rul_upper_90": hi,
            "risk_band": [risk_band(v) for v in lo],
            "ensemble": self.bundle["name"]
        })
        return out

