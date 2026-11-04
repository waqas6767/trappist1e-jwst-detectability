"""TRAPPIST-1e atmosphere detection lab: uses the trained pickles directly."""
import pickle, warnings
from pathlib import Path
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
st.set_page_config(page_title="TRAPPIST-1e Detection Lab", page_icon="🪐", layout="wide")

BASE = Path(__file__).parent
FILES = {
    "combined": "TRAPPIST1e_Climate_Detection_AI_Combined.pkl",
    "complete": "TRAPPIST1e_Detection_AI_COMPLETE.pkl",
    "stats": "TRAPPIST1e_Detection_Statistics_RF.pkl",
}
SAFE = ("sklearn", "numpy", "pandas", "scipy", "collections", "builtins", "joblib", "_codecs", "copyreg")
BLOCKED = {"eval", "exec", "compile", "open", "__import__", "getattr", "system"}
GAS = ["P_N2", "P_CO2", "P_CH4"]
LABEL = {"P_N2": "N₂", "P_CO2": "CO₂", "P_CH4": "CH₄"}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":  # the saved helper function is not needed here
            return lambda *a, **k: None
        if module.split(".")[0] not in SAFE or (module == "builtins" and name in BLOCKED):
            raise pickle.UnpicklingError(f"Blocked global: {module}.{name}")
        return super().find_class(module, name)


import os as _os, urllib.request as _ur
_RELEASE = "https://github.com/waqas6767/trappist1e-jwst-detectability/releases/download/v1.0.0"
_MDIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "models")

def _ensure_models():
    """Fetch model pickles from the GitHub release if they are not on disk."""
    _os.makedirs(_MDIR, exist_ok=True)
    for _f in ("TRAPPIST1e_Climate_Detection_AI_Combined.pkl",
               "TRAPPIST1e_Detection_AI_COMPLETE.pkl",
               "TRAPPIST1e_Detection_Statistics_RF.pkl"):
        _dst = _os.path.join(_MDIR, _f)
        if not _os.path.exists(_dst):
            with st.spinner(f"Downloading {_f} (first run only)…"):
                _ur.urlretrieve(f"{_RELEASE}/{_f}", _dst + ".part")
                _os.replace(_dst + ".part", _dst)

_ensure_models()

@st.cache_resource(show_spinner="Loading trained models…")
def load_all():
    out = {}
    for key, fn in FILES.items():
        path = next((d / fn for d in (BASE / "models", BASE) if (d / fn).exists()), None)
        if path is None:
            st.error(f"Missing {fn}. Put it in the `models/` folder next to app.py.")
            st.stop()
        with open(path, "rb") as fh:
            out[key] = SafeUnpickler(fh).load()
    return out


M = load_all()
climate, detect = M["combined"]["climate_model"], M["combined"]["detection_model"]
RANGES = M["combined"]["training_ranges"]
LOG = {k: (np.log10(RANGES[k]["min"]), np.log10(RANGES[k]["max"])) for k in GAS}
TLO, THI = RANGES["T_surf_climate"]["min"], RANGES["T_surf_climate"]["max"]


# ---------------- cross-validated calibration ----------------
@st.cache_resource(show_spinner="Cross-validating models (runs once, ~30s)…")
def calibrate():
    """10-fold CV on the 446 labelled samples: honest accuracy, residual quantiles
    for calibrated intervals, and a normalized training-point cloud for a
    nearest-neighbour reliability check. Fits fresh clones only, never the loaded
    models, so this never mutates or depends on training-time state."""
    X = pd.concat([M["complete"]["X_train"], M["complete"]["X_test"]]).reset_index(drop=True)
    Y = pd.concat([M["complete"]["y_train"], M["complete"]["y_test"]]).reset_index(drop=True)
    kf = KFold(n_splits=10, shuffle=True, random_state=0)

    det_pred = cross_val_predict(clone(detect), X.to_numpy(float), Y.to_numpy(float), cv=kf)
    det_resid = np.abs(det_pred - Y.to_numpy(float))
    det = {
        "r2": [float(r2_score(Y.iloc[:, i], det_pred[:, i])) for i in range(3)],
        "mae": det_resid.mean(0).tolist(),
        "q90": np.quantile(det_resid, 0.9, axis=0).tolist(),
    }

    Xc, yc = X[GAS].to_numpy(float), X["T_surf_climate"].to_numpy(float)
    clim_pred = cross_val_predict(clone(climate), Xc, yc, cv=kf)
    clim_resid = np.abs(clim_pred - yc)
    clim = {"r2": float(r2_score(yc, clim_pred)), "mae": float(clim_resid.mean()),
            "q90": float(np.quantile(clim_resid, 0.9))}

    lo = np.array([LOG[k][0] for k in GAS]); hi = np.array([LOG[k][1] for k in GAS])
    Xn = (np.log10(X[GAS].to_numpy(float)) - lo) / (hi - lo)
    self_d = np.sqrt(((Xn[:, None, :] - Xn[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(self_d, np.inf)
    nn_thresh = float(np.quantile(self_d.min(1), 0.95))

    return dict(det=det, clim=clim, Xn=Xn, nn_thresh=nn_thresh, n=len(X))


CAL = calibrate()


def reliability(n2, co2, ch4):
    lo = np.array([LOG[k][0] for k in GAS]); hi = np.array([LOG[k][1] for k in GAS])
    q = (np.log10([n2, co2, ch4]) - lo) / (hi - lo)
    if (q < 0).any() or (q > 1).any():
        return "out", float("nan")
    d = float(np.sqrt(((CAL["Xn"] - q) ** 2).sum(1)).min())
    return ("sparse" if d > CAL["nn_thresh"] else "ok"), d


def run(n2, co2, ch4, T=None):
    if T is None:
        T = float(climate.predict(np.array([[n2, co2, ch4]]))[0])
    x = np.array([[n2, co2, ch4, T]])
    trees = np.array([t.predict(x)[0] for t in detect.estimators_])
    m, s = trees.mean(0), trees.std(0)
    return dict(T=T, chi=m[0], red=m[1], n=m[2], chi_sd=s[0], n_sd=s[2], n_trees=trees[:, 2])


def batch(df):
    X = df[GAS].to_numpy(float)
    T = df["T_surf_climate"].to_numpy(float) if "T_surf_climate" in df else climate.predict(X)
    P = detect.predict(np.column_stack([X, T]))
    res = df.copy()
    res["T_surf_climate"] = T
    res["chi2"], res["reduced_chi2"], res["n_transits_for_detection"] = P.T
    res["reliability"] = [reliability(*row)[0] for row in X]
    return res


def fmt(v):
    return f"{v:.2f}" if v >= 1 else f"{v:.3f}" if v >= 0.01 else f"{v:.1e}"


PRESETS = {
    "Thin, CO₂-poor": (0.5, 0.002, 2e-6),
    "Earth-like N₂": (0.8, 0.01, 1e-4),
    "CO₂ greenhouse": (10.0, 0.5, 1e-3),
    "Methane-rich": (5.0, 0.05, 8e-3),
    "Dense N₂": (18.0, 0.1, 5e-4),
}
for k, v in zip(GAS, PRESETS["Earth-like N₂"]):
    st.session_state.setdefault(k, float(np.log10(v)))
    st.session_state.setdefault(f"{k}_num", float(v))


def apply_preset(name):
    for k, v in zip(GAS, PRESETS[name]):
        st.session_state[k] = float(np.log10(v))
        st.session_state[f"{k}_num"] = float(v)


def sync_from_slider(k):
    st.session_state[f"{k}_num"] = float(10 ** st.session_state[k])


def sync_from_number(k):
    lo, hi = LOG[k]
    val = max(st.session_state[f"{k}_num"], 10 ** lo * 0.999999)
    st.session_state[k] = float(np.clip(np.log10(val), lo, hi))
    st.session_state[f"{k}_num"] = float(10 ** st.session_state[k])


# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Atmosphere")
    for name in PRESETS:
        st.button(name, on_click=apply_preset, args=(name,), width='stretch')
    st.divider()
    for k in GAS:
        lo, hi = LOG[k]
        st.caption(f"{LABEL[k]} pressure (bar) — training range {fmt(10**lo)} to {fmt(10**hi)}")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.slider(f"log₁₀ {LABEL[k]}", float(lo), float(hi), key=k, step=0.01,
                       on_change=sync_from_slider, args=(k,), label_visibility="collapsed")
        with c2:
            st.number_input(f"{LABEL[k]} bar", key=f"{k}_num", step=0.0, format="%.6g",
                              on_change=sync_from_number, args=(k,), label_visibility="collapsed")
    st.divider()
    manual = st.toggle("Set surface temperature myself")
    T_in = st.slider("Surface temperature (K)", float(TLO), float(THI), 300.0, disabled=not manual)
    budget = st.slider("Your transit budget", 1, 100, 20)

n2, co2, ch4 = (st.session_state[f"{k}_num"] for k in GAS)
T_used = T_in if manual else None
r = run(n2, co2, ch4, T_used)
rel, nn_dist = reliability(n2, co2, ch4)

st.title("TRAPPIST-1e atmosphere detection lab")
st.caption("Predictions come straight from your trained random forests (500 trees each). "
           f"Accuracy figures below are from fresh 10-fold cross-validation on all {CAL['n']} labelled samples, "
           "not the single train/test split stored in the files.")
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Explore", "Batch CSV", "About the models"])

# ---------------- prediction ----------------
with tab1:
    if rel == "sparse":
        st.warning(f"This atmosphere is inside the training ranges but far from any training sample "
                   f"(nearest-neighbour distance {nn_dist:.2f}, vs a typical gap of {CAL['nn_thresh']:.2f} "
                   "among training points). Treat the prediction as less certain.")
    elif rel == "out":
        st.error("Outside the training ranges — the model is extrapolating and should not be trusted here.")
    else:
        st.success(f"Inside a well-sampled region of the training data (nearest neighbour at distance {nn_dist:.2f}).")

    qn = CAL["det"]["q90"][2]  # 90% CV residual quantile for transits
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transits needed", f"{r['n']:.1f}", f"90% interval ± {qn:.1f}", delta_color="off")
    c2.metric("Surface temperature", f"{r['T']:.0f} K",
               ("set by you" if manual else f"± {CAL['clim']['q90']:.1f} K (90%)"), delta_color="off")
    c3.metric("χ²", f"{r['chi']:.1f}", f"± {CAL['det']['q90'][0]:.1f} (90%)", delta_color="off")
    c4.metric("Reduced χ²", f"{r['red']:.3f}", f"± {CAL['det']['q90'][1]:.3f} (90%)", delta_color="off")
    st.caption("Intervals are cross-validated 90% residual bounds (cross-conformal style): on the 446 labelled "
               "samples, about 90% of true values fell within this distance of the prediction. They are a "
               "calibration check across the whole dataset, not an exact interval for this one point, and they "
               "widen further outside the training ranges or in sparse regions flagged above.")

    lo_n, hi_n = r["n"] - qn, r["n"] + qn
    if hi_n <= budget:
        st.success(f"Within your budget of {budget} transits even in the worst case of the 90% interval.")
    elif lo_n > budget:
        st.error(f"Over your budget of {budget} transits even in the best case of the 90% interval.")
    else:
        st.info(f"Uncertain: the 90% interval ({lo_n:.0f} to {hi_n:.0f} transits) straddles your budget of {budget}.")
    st.progress(min(1.0, r["n"] / max(100, budget)), text=f"{r['n']:.0f} of {max(100, budget)} transits on the scale")

    st.subheader("How much the 500 trees agree (not the same as the interval above)")
    h, e = np.histogram(r["n_trees"], bins=25)
    hist = pd.DataFrame({"transits": (e[:-1] + e[1:]) / 2, "trees": h})
    st.altair_chart(alt.Chart(hist).mark_bar().encode(x=alt.X("transits:Q", title="Transits needed (per tree)"),
                    y=alt.Y("trees:Q", title="Trees")).properties(height=220), width='stretch')

# ---------------- explore ----------------
with tab2:
    st.subheader("Change one gas, hold the rest")
    cols = st.columns(3)
    cur = [n2, co2, ch4]
    for col, k in zip(cols, GAS):
        grid = np.linspace(*LOG[k], 80)
        X = np.tile(cur, (80, 1)).astype(float)
        X[:, GAS.index(k)] = 10 ** grid
        T = np.full(80, T_used) if manual else climate.predict(X)
        y = detect.predict(np.column_stack([X, T]))[:, 2]
        d = pd.DataFrame({"pressure": 10 ** grid, "transits": y})
        line = alt.Chart(d).mark_line().encode(x=alt.X("pressure:Q", scale=alt.Scale(type="log"), title=f"{LABEL[k]} (bar)"),
                                               y=alt.Y("transits:Q", scale=alt.Scale(zero=False), title="Transits"))
        pt = alt.Chart(pd.DataFrame({"pressure": [cur[GAS.index(k)]], "transits": [r["n"]]})).mark_point(size=120, filled=True, color="#d9891a").encode(x="pressure:Q", y="transits:Q")
        col.altair_chart((line + pt).properties(height=240), width='stretch')
    st.subheader("CO₂ × CH₄ map (N₂ held at your value)")
    g = np.linspace(*LOG["P_CO2"], 40), np.linspace(*LOG["P_CH4"], 40)
    A, B = np.meshgrid(*g)
    X = np.column_stack([np.full(A.size, n2), 10 ** A.ravel(), 10 ** B.ravel()])
    T = np.full(A.size, T_used) if manual else climate.predict(X)
    z = detect.predict(np.column_stack([X, T]))[:, 2]
    hm = pd.DataFrame({"lcO2": A.ravel(), "lCH4": B.ravel(), "transits": z})
    st.altair_chart(alt.Chart(hm).mark_rect().encode(
        x=alt.X("lcO2:Q", bin=alt.Bin(maxbins=40), title="log₁₀ CO₂ (bar)"),
        y=alt.Y("lCH4:Q", bin=alt.Bin(maxbins=40), title="log₁₀ CH₄ (bar)"),
        color=alt.Color("mean(transits):Q", scale=alt.Scale(scheme="viridis", reverse=True), title="Transits")).properties(height=380),
        width='stretch')

# ---------------- batch ----------------
with tab3:
    st.write("Upload a CSV with columns `P_N2`, `P_CO2`, `P_CH4` (bar). `T_surf_climate` (K) is optional; "
             "if missing, the climate model predicts it. Each row also gets a `reliability` label: "
             "`ok` (well-sampled), `sparse` (inside range but far from training data) or `out` (outside "
             "the training ranges — do not trust).")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        miss = [c for c in GAS if c not in df]
        if miss:
            st.error(f"Missing columns: {', '.join(miss)}")
        else:
            out = batch(df)
            n_out = (out["reliability"] == "out").sum()
            n_sparse = (out["reliability"] == "sparse").sum()
            if n_out:
                st.error(f"{n_out} row(s) are outside the training ranges. Their predictions are extrapolation.")
            if n_sparse:
                st.warning(f"{n_sparse} row(s) are in sparse regions of the training data.")
            st.dataframe(out, width='stretch')
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "trappist1e_predictions.csv", "text/csv")
    else:
        ex = pd.DataFrame({k: [PRESETS[p][i] for p in PRESETS] for i, k in enumerate(GAS)})
        st.download_button("Download an example CSV", ex.to_csv(index=False).encode(), "example_input.csv", "text/csv")

# ---------------- about ----------------
with tab4:
    st.subheader("Cross-validated accuracy (10-fold, all 446 samples)")
    st.caption("Computed fresh each time the app starts, by cloning and refitting the loaded models on folds "
               "they were not trained on. This is more honest than a single stored train/test split.")
    cv_tbl = pd.DataFrame({"Output": ["χ²", "Reduced χ²", "Transits needed"],
                            "R²": CAL["det"]["r2"], "MAE": CAL["det"]["mae"], "90% residual": CAL["det"]["q90"]})
    st.dataframe(cv_tbl.set_index("Output"), width='stretch')
    st.write(f"Climate model (temperature): R² = {CAL['clim']['r2']:.3f}, "
             f"MAE = {CAL['clim']['mae']:.2f} K, 90% residual = {CAL['clim']['q90']:.2f} K.")

    st.subheader("Metrics stored in the original files (single 80/20 split)")
    met = M["complete"]["metrics"]
    st.dataframe(pd.DataFrame(met).T.rename(columns={"R2": "R²"}), width='stretch')

    st.subheader("What the detection model relies on")
    imp = pd.Series(detect.feature_importances_, index=M["combined"]["detection_feature_columns"]).sort_values()
    st.bar_chart(imp)

    st.subheader("Training coverage")
    st.write(f"A query is flagged **sparse** if its nearest labelled training point (in normalized log-pressure "
             f"space) is farther than {CAL['nn_thresh']:.2f} — the 95th percentile of how far training points "
             "are from each other. This does not guarantee accuracy inside that radius, only that similar "
             "inputs were seen during training.")
    st.dataframe(pd.DataFrame(RANGES).T, width='stretch')

    same = np.allclose(M["complete"]["model"].predict(M["complete"]["X_test"]), detect.predict(M["complete"]["X_test"].to_numpy()))
    st.caption(f"Detection model in the Combined file matches the COMPLETE file on the test set: {same}. "
               f"Statistics_RF file: {M['stats']['metadata'].get('total_valid_samples')} samples, "
               f"{M['stats']['metadata'].get('training_samples')} train / {M['stats']['metadata'].get('testing_samples')} test.")
