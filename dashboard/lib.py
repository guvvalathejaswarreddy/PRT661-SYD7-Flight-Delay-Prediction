"""Shared helpers for the PRT661 Flight Delay Prediction dashboard.

Every page imports from here so paths, styling, and data loading stay
consistent. All file paths are resolved relative to this file, so the app
runs the same way from any working directory (VS Code "Run", a terminal in
the repo root, or `streamlit run dashboard/streamlit_app.py`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
ASSETS_DIR = DASHBOARD_DIR / "assets"
FIGURES_DIR = ASSETS_DIR / "figures"
DATA_DIR = DASHBOARD_DIR / "data"

# --- Assessment 4 assets (exported by section 24 of the analysis notebook) ---
A4_DIR = ASSETS_DIR / "a4"
A4_SAMPLE_PATH = DATA_DIR / "a4_predictions_sample.parquet"
A4_CLF_PATH = A4_DIR / "a4_classifier_bundle.joblib"
A4_DUR_PATH = A4_DIR / "a4_duration_quantile_bundle.joblib"

BRAND = "#2563EB"
ACCENT = "#DD8452"
OK_GREEN = "#16A34A"
PLOTLY_TEMPLATE = "plotly_white"

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
_CSS = """
<style>
  .block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1300px;}
  [data-testid="stMetricValue"] {font-size: 1.55rem;}
  [data-testid="stMetricLabel"] {opacity: .72;}
  h1 {font-size: 2.1rem;}
  h1, h2, h3 {letter-spacing: -0.01em;}
  [data-testid="stSidebarNav"] {padding-top: .4rem;}
</style>
"""


def page_config() -> None:
    """Called once, in the navigation entry point (streamlit_app.py)."""
    st.set_page_config(
        page_title="Flight Delay Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def page(title: str = "") -> None:
    """Back-compat shim for running a view file on its own (e.g. in tests)."""
    try:
        page_config()
    except Exception:
        st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    p = ASSETS_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    p = ASSETS_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def figure(name: str) -> Path | None:
    p = FIGURES_DIR / name
    return p if p.exists() else None


def show_figure(name: str, caption: str = "") -> None:
    p = figure(name)
    if p is None:
        st.warning(f"Figure not found: `{name}` — run the analysis notebook to regenerate `assets/figures/`.")
        return
    if p.suffix == ".html":
        html = p.read_text()
        try:
            import streamlit.components.v1 as components
            components.html(html, height=520, scrolling=True)
        except Exception:  # pragma: no cover - API drift / sandbox
            st.download_button(f"Open {name}", html, file_name=name, mime="text/html")
        if caption:
            st.caption(caption)
    else:
        st.image(str(p), caption=caption or None, width="stretch")





# --------------------------------------------------------------------------
# Assessment 4: calibrated probability, quantile duration, thresholds
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_a4_sample() -> pd.DataFrame:
    """300k stratified 2023 flights with calibrated probability and duration range."""
    if not A4_SAMPLE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(A4_SAMPLE_PATH)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    return df


@st.cache_data(show_spinner=False)
def a4_csv(name: str) -> pd.DataFrame:
    p = A4_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def a4_json(name: str) -> dict:
    p = A4_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_resource(show_spinner="Loading the Assessment 4 models…")
def load_a4_models():
    """(classifier bundle, duration bundle) or (None, None)."""
    import warnings

    import joblib
    if A4_CLF_PATH.exists() and A4_DUR_PATH.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")       # scikit-learn minor-version pickle notice
            return joblib.load(A4_CLF_PATH), joblib.load(A4_DUR_PATH)
    return None, None


def calibrate(raw, bundle):
    """Raw LightGBM score -> calibrated probability, using the calibrator fitted on held-out validation data."""
    import numpy as np
    raw = np.asarray(raw, dtype=float).ravel()
    if bundle["calibration_method"] == "platt":
        z = np.log(np.clip(raw, 1e-6, 1 - 1e-6) / (1 - np.clip(raw, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        return bundle["calibrator"].predict_proba(z)[:, 1]
    return bundle["calibrator"].predict(raw)


def a4_operating_levels() -> list[dict]:
    """Alert levels offered to the user: threshold chosen on validation data + its 2023 backtest result."""
    thr = a4_json("a4_thresholds.json").get("thresholds", {})
    ops = a4_csv("a4_operating_points.csv")
    if not thr or ops.empty:
        return []
    want = [
        ("Balanced (max F1)", thr["max_f1"], "max F1 on validation"),
        ("Cost-weighted (a miss costs 3x a false alarm)", thr["min_cost"], "min expected cost on validation"),
        ("Alert budget: top 10% of flights", thr["budget_10pct"], "alert budget: top 10% of flights"),
        ("High precision (about 1 in 2 alerts is right)", None, "smallest threshold with validation precision >= 50%"),
    ]
    out = []
    for label, t, key in want:
        row = ops[ops["operating point"].str.startswith(key)]
        if row.empty:
            continue
        r = row.iloc[0]
        out.append({"label": label, "threshold": float(r["threshold"] if t is None else t),
                    "flagged": float(r["flights flagged"]), "precision": float(r["precision"]),
                    "recall": float(r["recall"]), "alerts_per_day": float(r["alerts per day (network)"])})
    return out
