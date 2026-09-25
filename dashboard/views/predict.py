"""Predict a Flight — enter flight details, get a delay prediction from the
trained tuned-LightGBM classifier + duration regressor."""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

sys.path.insert(0, str(lib.PROJECT_ROOT))
from ingestion.opensky_live import OpenSkyError, get_live_snapshot  # noqa: E402


@st.cache_data(show_spinner=False, ttl=30)
def _cached_live_snapshot(iata: str):
    """Cached for 30s so re-running the form doesn't hammer the OpenSky API."""
    return get_live_snapshot(iata)


st.title("Predict a Flight")

clf, reg = lib.load_models()
ref = lib.load_predict_reference()

if clf is None or reg is None or not ref:
    st.error("Model files not found — run: `python dashboard/generate_predictions.py`")
    st.stop()

carriers = sorted(ref["carriers"])
airports = sorted(ref["origins"])
defaults = ref["defaults"]

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.form("predict_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        carrier = st.selectbox("Carrier", carriers,
                               index=carriers.index("AA") if "AA" in carriers else 0)
    with c2:
        origin = st.selectbox("Origin airport", airports,
                              index=airports.index("ATL") if "ATL" in airports else 0)
    with c3:
        dests = ref["origins"].get(origin, {}).get("destinations") or airports
        destination = st.selectbox("Destination airport", dests)

    c4, c5, c6 = st.columns(3)
    with c4:
        dep_date = st.date_input("Scheduled departure date", value=dt.date(2024, 6, 15))
    with c5:
        dep_time = st.time_input("Scheduled departure time", value=dt.time(8, 0))
    with c6:
        threshold = st.slider("Decision threshold  P(delay) ≥", 0.05, 0.95, 0.50, 0.05)

    o_ref = ref["origins"].get(origin, {})
    c_ref = ref["carriers"].get(carrier, {})
    route_key = f"{origin}-{destination}"
    dist_default = float(ref["routes"].get(route_key, defaults["distance"]))

    with st.expander("Advanced"):
        a1, a2, a3 = st.columns(3)
        with a1:
            distance = st.number_input("Route distance (miles)", min_value=30.0,
                                       max_value=6000.0, value=dist_default, step=25.0)
        with a2:
            carrier_rate = st.slider(
                "Carrier's recent 7-day delay rate", 0.0, 1.0,
                float(c_ref.get("delay_rate_7d", defaults["carrier_delay_rate_7d"])), 0.01)
        with a3:
            origin_rate = st.slider(
                "Origin's recent 7-day delay rate", 0.0, 1.0,
                float(o_ref.get("delay_rate_7d", defaults["origin_delay_rate_7d"])), 0.01)

    submitted = st.form_submit_button("Predict delay", type="primary", width="stretch")

if not submitted:
    st.stop()

# --------------------------------------------------------------------------
# Build the exact feature row the model was trained on
# --------------------------------------------------------------------------
try:
    import holidays
    is_holiday = int(dep_date in holidays.country_holidays("US", years=[dep_date.year]))
except Exception:
    is_holiday = 0

dow = dep_date.weekday()  # Mon=0 … Sun=6  (matches isodow-1 used in training)
row = {
    "crs_dep_hour": int(dep_time.hour),
    "day_of_week": int(dow),
    "is_weekend": int(dow in (5, 6)),
    "month": int(dep_date.month),
    "is_holiday": is_holiday,
    "distance": float(distance),
    "carrier_delay_rate_7d": float(carrier_rate),
    "origin_delay_rate_7d": float(origin_rate),
    "origin_degree_centrality": float(o_ref.get("degree_centrality", 0.0)),
    "carrier": str(carrier),
    "origin_hub_tier": str(o_ref.get("hub_tier", "medium")),
}
X = pd.DataFrame([row])[lib.ALL_FEATURES]

prob = float(clf.predict_proba(X)[0, 1])
minutes_if_delayed = float(max(0.0, reg.predict(X)[0]))
will_delay = prob >= threshold

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
st.divider()
left, right = st.columns([1.1, 1])

with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": " %", "font": {"size": 40}},
        title={"text": "Probability of a 15+ min arrival delay"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": lib.BRAND},
            "steps": [
                {"range": [0, 20], "color": "#DCFCE7"},
                {"range": [20, 40], "color": "#FEF9C3"},
                {"range": [40, 100], "color": "#FEE2E2"},
            ],
            "threshold": {"line": {"color": lib.ACCENT, "width": 4},
                          "value": threshold * 100},
        },
    ))
    gauge.update_layout(height=300, margin=dict(t=60, b=10, l=30, r=30),
                        template=lib.PLOTLY_TEMPLATE)
    st.plotly_chart(gauge, width="stretch")

with right:
    st.metric("Verdict at threshold", "LIKELY DELAYED" if will_delay else "LIKELY ON TIME")
    st.metric("If delayed, estimated arrival delay", f"≈ {minutes_if_delayed:.0f} min")
    if will_delay:
        st.warning(f"P(delay) {prob:.0%} ≥ threshold {threshold:.0%}")
    else:
        st.success(f"P(delay) {prob:.0%} < threshold {threshold:.0%}")

st.subheader("Model inputs")
st.dataframe(
    pd.DataFrame({"feature": lib.ALL_FEATURES,
                  "value": [row[k] for k in lib.ALL_FEATURES]}),
    width="stretch", hide_index=True,
)

# --------------------------------------------------------------------------
# Live airport activity (PRT661-16/17) — OpenSky ADS-B snapshot.
# Purely informational context alongside the prediction above: it does not
# feed the model, so nothing here changes ALL_FEATURES or requires retraining.
# --------------------------------------------------------------------------
st.divider()
st.subheader("Live airport activity (OpenSky)")
st.caption(
    "Live ADS-B snapshot near the selected origin, for operational context "
    "only — the prediction above always uses the 11 trained features and is "
    "unaffected by this panel."
)
try:
    snap = _cached_live_snapshot(origin)
except OpenSkyError as exc:
    st.info(f"Live data unavailable right now ({exc})")
else:
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric(f"Aircraft near {origin}", snap.total_aircraft)
    lc2.metric("Airborne", snap.airborne)
    lc3.metric("On ground", snap.on_ground)
    as_of = dt.datetime.utcfromtimestamp(snap.generated_unix).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"As of {as_of} · source: OpenSky Network")
