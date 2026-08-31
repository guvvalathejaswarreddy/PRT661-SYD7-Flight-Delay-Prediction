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

PREDICTIONS_PATH = DATA_DIR / "predictions.parquet"
PREDICTIONS_FULL_PATH = DATA_DIR / "predictions_full.parquet"
METRICS_PATH = ASSETS_DIR / "model_metrics.json"
MODEL_CLF_PATH = ASSETS_DIR / "model_classifier.joblib"
MODEL_REG_PATH = ASSETS_DIR / "model_regressor.joblib"
PREDICT_REF_PATH = ASSETS_DIR / "predict_reference.json"

# Model input schema (must match dashboard/generate_predictions.py).
NUMERIC_FEATURES = [
    "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
    "distance", "carrier_delay_rate_7d", "origin_delay_rate_7d",
    "origin_degree_centrality",
]
CATEGORICAL_FEATURES = ["carrier", "origin_hub_tier"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

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
def load_predictions(prefer_full: bool = False) -> pd.DataFrame:
    path = PREDICTIONS_FULL_PATH if (prefer_full and PREDICTIONS_FULL_PATH.exists()) else PREDICTIONS_PATH
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    if "actual_delayed" in df.columns:
        df["actual_delayed"] = df["actual_delayed"].astype(bool)
    if "predicted_delayed" in df.columns:
        df["predicted_delayed"] = df["predicted_delayed"].astype(bool)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


@st.cache_resource(show_spinner="Loading trained models…")
def load_models():
    """(classifier_pipeline, regressor_pipeline) or (None, None) if not built yet."""
    import joblib
    if MODEL_CLF_PATH.exists() and MODEL_REG_PATH.exists():
        return joblib.load(MODEL_CLF_PATH), joblib.load(MODEL_REG_PATH)
    return None, None


@st.cache_data(show_spinner=False)
def load_predict_reference() -> dict:
    if PREDICT_REF_PATH.exists():
        return json.loads(PREDICT_REF_PATH.read_text())
    return {}


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


def no_predictions_banner() -> bool:
    """Returns True when prediction data is present (caller should continue)."""
    if load_predictions().empty:
        st.error("No prediction data. Run: `python dashboard/generate_predictions.py`")
        return False
    return True


def equalised_odds_gap(frame: pd.DataFrame) -> float:
    return max(
        frame["tpr"].max() - frame["tpr"].min(),
        frame["fpr"].max() - frame["fpr"].min(),
    )


def demographic_parity_gap(frame: pd.DataFrame) -> float:
    return frame["positive_pred_rate"].max() - frame["positive_pred_rate"].min()
