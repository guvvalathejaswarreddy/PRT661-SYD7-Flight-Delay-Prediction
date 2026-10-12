"""About — pipeline, method, model card, data availability and governance."""
from __future__ import annotations

import streamlit as st

import lib

st.title("About")
summary = lib.a4_json("a4_summary.json")
cal = lib.a4_csv("a4_calibration_summary.csv")
avail = lib.a4_csv("feature_availability.csv")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Pipeline")
    st.markdown(
        """
| Layer | Component |
|---|---|
| Ingestion | BTS on-time performance, OpenFlights airports, NOAA GHCN-Daily weather, OpenSky live context |
| Storage | Parquet (raw BTS, filtered NOAA, engineered features, checkpoints) |
| Processing | DuckDB clean + join; calendar-window delay rates; training-window graph features; nearest-station weather join |
| Modelling | LightGBM classifier (calibrated), LightGBM quantile regressors, baselines: LogReg, Random Forest, XGBoost |
| Evaluation | Walk-forward CV, ablation, calibration, thresholds and workload, leakage tests, fairness |
| Dashboard | Streamlit |
"""
    )
with col2:
    st.subheader("Method")
    st.markdown(
        """
| Item | Value |
|---|---|
| Training years | 2018, 2019, 2022 (2020–2021 excluded as COVID-distorted) |
| Test year | 2023 (used once) |
| Validation | last 15% of training dates (from 2022-07-11) for calibration and thresholds |
| Rolling delay rates | flight-weighted, calendar days [d-7, d-1] |
| Hub tier, centrality | computed from training years only |
| Weather | NOAA GHCN-Daily, previous day (d-1), nearest station within 50 km; the Predict page fetches it live from NOAA's Access Data Service (manual scenario as fallback) |
| Calibration | Platt scaling, chosen against isotonic on held-out data |
"""
    )

st.subheader("Model card — deployed classifier")
if summary and not cal.empty:
    c = cal.iloc[1]
    st.markdown(
        f"""
| | |
|---|---|
| Model | LightGBM, 26 inputs (11 BTS-derived + 15 prior-day weather), balanced class weights, Platt-calibrated |
| Training rows | {summary['n_train']:,} (fit on the first 85% of dates; last 15% held out for calibration and thresholds) |
| Test | {summary['n_test']:,} flights, {summary['test_year']} |
| AUC-ROC / average precision | {c['auc_roc']:.3f} / {c['avg_precision']:.3f} (delay base rate 0.206) |
| Calibration error (ECE) | {c['ECE']:.3f} (raw score: {cal.iloc[0]['ECE']:.3f}) |
| Duration model | quantile regression q10 / q50 / q90; 80% interval coverage {summary['duration_interval_coverage_80']:.1%}, median width {summary['duration_median_interval_width_min']:.0f} min; **no skill over a naive baseline** |
"""
    )

st.subheader("When is each input available?")
st.caption("Every deployed input is known before departure. Same-day weather is not, and is used only in a labelled upper-bound experiment.")
if not avail.empty:
    st.dataframe(avail, width="stretch", hide_index=True)

st.subheader("Known limitations")
st.markdown(
    """
- **Modest discrimination.** AUC ≈ 0.68: the model ranks flights better than chance and the probabilities are honest, but many delays are not predictable from schedule, recent delays and prior-day weather alone.
- **Duration is not reliable.** The minutes range is wide and no better than a naive guess; do not use it for individual decisions.
- **No weather forecast.** Same-day weather would raise AUC to about 0.71 but is not available before departure; a forecast feed (NWS) is future work.
- **Recent-rate inputs need a nightly refresh** of delay outcomes; the form defaults to 2023 averages.
- **Carrier is not a protected attribute**, but group gaps should still be monitored (see Fairness).
- **No inbound-aircraft information** (the previous leg of the same aircraft). It is a plausible extra signal and is listed as future work; it was not tested here.
"""
)

st.subheader("Data, privacy and governance")
st.markdown(
    """
- **Sources:** BTS On-Time Performance (US Government, public), NOAA GHCN-Daily (public domain), OpenFlights airport data (Open Database Licence), OpenSky Network (live context only; rate-limited, not used by the model).
- **Privacy:** the data describe flights, airports and weather. There are no passenger, crew or ticket records and no personal data is stored or processed.
- **Security:** no credentials in the repository (`.env` is git-ignored); the app reads local files only apart from the optional OpenSky call.
- **Reproducibility:** one notebook reproduces every number; all intermediate steps are checkpointed; automated leakage tests stop the run if a feature could use same-day or future information.
"""
)
