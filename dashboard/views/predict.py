"""Predict a Flight (Assessment 4) — calibrated delay probability, an alert level whose workload is known,
a duration *range* with a reliability warning, and the top reasons in plain language."""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import lib

sys.path.insert(0, str(lib.PROJECT_ROOT))
from ingestion.opensky_live import OpenSkyError, get_live_snapshot  # noqa: E402
import noaa_live  # noqa: E402  (dashboard/noaa_live.py)


@st.cache_data(show_spinner=False, ttl=30)
def _cached_live_snapshot(iata: str):
    """Cached for 30s so re-running the form doesn't hammer the OpenSky API."""
    return get_live_snapshot(iata)


# Weather "yesterday" presets -> the NOAA-style inputs the model was trained on (mm, deg C, m/s, 0/1).
# "Not available" leaves every weather input missing, which LightGBM handles natively.
_NA = dict(prcp=np.nan, snow=np.nan, snwd=np.nan, tmax=np.nan, tmin=np.nan, awnd=np.nan, wsf2=np.nan,
           thunder=np.nan, fog=np.nan, prcp3=np.nan)
WEATHER = {
    "Not available (unknown)": _NA,
    "Dry and calm": dict(prcp=0, snow=0, snwd=0, tmax=22, tmin=11, awnd=3, wsf2=8, thunder=0, fog=0, prcp3=0),
    "Light rain": dict(prcp=5, snow=0, snwd=0, tmax=18, tmin=11, awnd=4, wsf2=10, thunder=0, fog=0, prcp3=8),
    "Heavy rain": dict(prcp=25, snow=0, snwd=0, tmax=17, tmin=12, awnd=5, wsf2=13, thunder=0, fog=0, prcp3=40),
    "Thunderstorms": dict(prcp=15, snow=0, snwd=0, tmax=27, tmin=19, awnd=5, wsf2=16, thunder=1, fog=0, prcp3=20),
    "Snow": dict(prcp=10, snow=100, snwd=150, tmax=-2, tmin=-8, awnd=5, wsf2=12, thunder=0, fog=0, prcp3=15),
    "Strong wind": dict(prcp=0, snow=0, snwd=0, tmax=15, tmin=6, awnd=9, wsf2=20, thunder=0, fog=0, prcp3=0),
    "Fog": dict(prcp=1, snow=0, snwd=0, tmax=12, tmin=8, awnd=2, wsf2=6, thunder=0, fog=1, prcp3=2),
}

@st.cache_data(show_spinner="Fetching NOAA observations…", ttl=3600)
def _noaa_prior_day(iata: str, flight_date: dt.date):
    """Prior-day observations for one airport, or (None, reason) when they cannot be obtained."""
    try:
        res = noaa_live.fetch_prior_day(iata, flight_date, lib.a4_csv("airport_weather_station_map.csv"))
        return res, ""
    except noaa_live.NoaaUnavailable as exc:
        return None, str(exc)


st.title("Predict a Flight")
st.caption("Probability that a flight arrives 15+ minutes late, from information available before departure.")

bundle, dur = lib.load_a4_models()
ref = lib.a4_json("a4_predict_reference.json")
sample = lib.load_a4_sample()
phrases = lib.a4_json("a4_feature_phrases.json")
levels = lib.a4_operating_levels()
summary = lib.a4_json("a4_summary.json")

if bundle is None or dur is None or not ref or not levels:
    st.error("Assessment 4 model files not found in `dashboard/assets/a4/` — export them from the notebook (section 24).")
    st.stop()

base_rate = float(sample["actual_delayed"].mean()) if not sample.empty else 0.206
carriers_2023 = sorted(sample["carrier"].unique()) if not sample.empty else sorted(ref["carriers"])
airports = sorted(ref["origins"])
routes = ref["routes"]
rate_by_carrier = sample.groupby("carrier")["actual_delayed"].mean().to_dict() if not sample.empty else {}
rate_by_origin = sample.groupby("origin")["actual_delayed"].mean().to_dict() if not sample.empty else {}
fallback_rate = ref["recent_rate_defaults"]["carrier_delay_rate_7d"]

# --------------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------------
with st.container():
    c1, c2, c3 = st.columns(3)
    with c1:
        carrier = st.selectbox("Carrier", carriers_2023, index=carriers_2023.index("AA") if "AA" in carriers_2023 else 0)
    with c2:
        origin = st.selectbox("Origin airport", airports, index=airports.index("ATL") if "ATL" in airports else 0)
    with c3:
        dests = sorted({k.split("-")[1] for k in routes if k.startswith(origin + "-")}) or airports
        destination = st.selectbox("Destination airport", dests)

    c4, c5, c6 = st.columns(3)
    with c4:
        dep_date = st.date_input("Scheduled departure date", value=dt.date(2023, 7, 14))
    with c5:
        dep_time = st.time_input("Scheduled departure time", value=dt.time(17, 0))
    with c6:
        level_label = st.selectbox("Alert level", [lv["label"] for lv in levels], index=0,
                                   help="Each level uses a probability threshold chosen on validation data. "
                                        "The table below the result shows what it means for workload.")

    weather_mode = st.radio(
        "Weather inputs", ["Automatic: NOAA observations from the day before the flight", "Manual scenario"], horizontal=True,
        help="Automatic mode looks up the nearest NOAA GHCN-Daily station for each airport and uses day d-1 "
             "(same-day weather is not known before departure).")
    if weather_mode == "Manual scenario":
        w1, w2 = st.columns(2)
        with w1:
            w_orig = st.selectbox("Weather yesterday at origin", list(WEATHER), index=0)
        with w2:
            w_dest = st.selectbox("Weather yesterday at destination", list(WEATHER), index=0)

    o_ref = ref["origins"].get(origin, {})
    dist_default = float(routes.get(f"{origin}-{destination}", 800))
    with st.expander("Advanced"):
        a1, a2, a3 = st.columns(3)
        with a1:
            distance = st.number_input("Route distance (miles)", min_value=30.0, max_value=6000.0, value=dist_default, step=25.0)
        with a2:
            carrier_rate = st.slider("Carrier's delay rate over the last 7 days", 0.0, 1.0,
                                     float(round(rate_by_carrier.get(carrier, fallback_rate), 2)), 0.01,
                                     help="Default = this carrier's 2023 average. In production this is refreshed nightly.")
        with a3:
            origin_rate = st.slider("Origin's delay rate over the last 7 days", 0.0, 1.0,
                                    float(round(rate_by_origin.get(origin, fallback_rate), 2)), 0.01)

# --------------------------------------------------------------------------
# Build the feature row exactly as in training
# --------------------------------------------------------------------------
try:
    import holidays
    is_holiday = int(dep_date in holidays.country_holidays("US", years=[dep_date.year]))
except Exception:
    is_holiday = 0

dow = dep_date.weekday()
row = {
    "crs_dep_hour": float(dep_time.hour), "day_of_week": float(dow), "is_weekend": float(dow in (5, 6)),
    "month": float(dep_date.month), "is_holiday": float(is_holiday), "distance": float(distance),
    "carrier_delay_rate_7d": float(carrier_rate), "origin_delay_rate_7d": float(origin_rate),
    "origin_degree_centrality": float(o_ref.get("degree_centrality", 0.0)),
    "carrier": str(carrier), "origin_hub_tier": str(o_ref.get("hub_tier", "small")),
}
weather_note, weather_tables = None, {}
if weather_mode == "Manual scenario":
    wo, wd = WEATHER[w_orig], WEATHER[w_dest]
else:
    wo, wd = dict(_NA), dict(_NA)
    problems = []
    for tag, iata in (("origin", origin), ("destination", destination)):
        res, why = _noaa_prior_day(iata, dep_date)
        if res is None:
            problems.append(f"{tag} {iata}: {why}")
            continue
        inputs = noaa_live.to_model_inputs(res)
        if all(np.isnan(inputs[k]) for k in ("prcp", "tmax", "tmin", "awnd")):
            problems.append(f"{tag} {iata}: no NOAA observations for {res['date']} yet")
        weather_tables[tag] = (iata, res)
        (wo if tag == "origin" else wd).update(inputs)
    if problems:
        weather_note = "Weather treated as unknown where NOAA data was not available — " + "; ".join(problems) + "."
row.update({
    "o_prcp_d1": wo["prcp"], "o_snow_d1": wo["snow"], "o_snwd_d1": wo["snwd"], "o_tmax_d1": wo["tmax"], "o_tmin_d1": wo["tmin"],
    "o_awnd_d1": wo["awnd"], "o_wsf2_d1": wo["wsf2"], "o_thunder_d1": wo["thunder"], "o_fog_d1": wo["fog"], "o_prcp_3d": wo["prcp3"],
    "d_prcp_d1": wd["prcp"], "d_snow_d1": wd["snow"], "d_awnd_d1": wd["awnd"], "d_wsf2_d1": wd["wsf2"], "d_thunder_d1": wd["thunder"],
})
feats = bundle["features"]
X = pd.DataFrame([row])[feats]
for c in feats:
    if c not in ("carrier", "origin_hub_tier"):
        X[c] = X[c].astype("float64")
X["carrier"] = pd.Categorical(X["carrier"], categories=bundle["carrier_categories"])
X["origin_hub_tier"] = pd.Categorical(X["origin_hub_tier"], categories=bundle["tiers"])

model = bundle["model"]
raw = float(model.predict_proba(X)[0, 1])
prob = float(lib.calibrate([raw], bundle)[0])
qs = np.sort([float(dur["models"][q].predict(X)[0]) for q in dur["quantiles"]])
contrib = pd.Series(model.predict(X, pred_contrib=True)[0][:-1], index=feats)

level = next(lv for lv in levels if lv["label"] == level_label)
threshold = level["threshold"]
flagged = prob >= threshold

# --------------------------------------------------------------------------
# Result
# --------------------------------------------------------------------------
st.divider()
left, right = st.columns([1.1, 1])
with left:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": " %", "font": {"size": 42}},
        title={"text": "Probability of a 15+ min arrival delay (calibrated)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": lib.BRAND},
               "steps": [{"range": [0, 20], "color": "#DCFCE7"}, {"range": [20, 40], "color": "#FEF9C3"},
                         {"range": [40, 100], "color": "#FEE2E2"}],
               "threshold": {"line": {"color": lib.ACCENT, "width": 4}, "value": threshold * 100}},
    ))
    gauge.update_layout(height=300, margin=dict(t=70, b=10, l=30, r=30), template=lib.PLOTLY_TEMPLATE)
    st.plotly_chart(gauge, width="stretch")
    st.caption(f"Orange line = alert threshold {threshold:.0%} for “{level_label}”. "
               f"A typical flight has a {base_rate:.0%} chance of a 15+ minute arrival delay.")

with right:
    st.metric("Alert at this level", "ALERT: likely delayed" if flagged else "No alert")
    st.metric("Compared with an average flight", f"{prob / base_rate:.1f}x the delay risk",
              delta=f"{(prob - base_rate) * 100:+.0f} percentage points", delta_color="inverse")
    if flagged:
        st.warning(f"Probability {prob:.0%} is at or above the alert threshold {threshold:.0%}.")
    else:
        st.success(f"Probability {prob:.0%} is below the alert threshold {threshold:.0%}.")
    st.markdown(
        f"**What this alert level means (2023 backtest):** {level['precision']:.0%} of alerts were truly delayed, "
        f"{level['recall']:.0%} of all delays were caught, and {level['flagged']:.0%} of flights were flagged "
        f"(about {level['alerts_per_day']:,.0f} alerts per day across the network)."
    )

st.subheader("If it is delayed: how long?")
d1, d2 = st.columns([1.2, 1])
with d1:
    fig = go.Figure()
    fig.add_shape(type="line", x0=qs[0], x1=qs[2], y0=0, y1=0, line=dict(color=lib.BRAND, width=10))
    fig.add_trace(go.Scatter(x=[qs[0], qs[1], qs[2]], y=[0, 0, 0], mode="markers+text", text=["10%", "median", "90%"],
                             textposition="top center", marker=dict(size=[12, 18, 12], color=["#1E3A8A", lib.ACCENT, "#1E3A8A"]),
                             hovertemplate="%{x:.0f} min<extra></extra>"))
    fig.update_layout(height=170, margin=dict(t=20, b=30, l=10, r=10), template=lib.PLOTLY_TEMPLATE, showlegend=False,
                      xaxis=dict(title="Minutes late on arrival", rangemode="tozero"), yaxis=dict(visible=False, range=[-1, 1]))
    st.plotly_chart(fig, width="stretch")
with d2:
    st.metric("Likely range (80% interval)", f"{qs[0]:.0f}–{qs[2]:.0f} min", help=f"Median {qs[1]:.0f} min")
st.warning(
    "**Low reliability.** On 2023 test flights this range contained the true delay about "
    f"{summary.get('duration_interval_coverage_80', 0.79):.0%} of the time, but it is very wide "
    f"(median width {summary.get('duration_median_interval_width_min', 112):.0f} min) and the median estimate is no more accurate "
    "than guessing the typical delay (mean error 43.7 vs 44.4 min). Use it as an order of magnitude, not a promise. "
    "It applies only if the flight is delayed."
)

st.subheader("Why this probability?")
top = contrib.abs().sort_values(ascending=False).index[:5]
rows = [{"": "▲ raises risk" if contrib[f] > 0 else "▼ lowers risk",
         "Reason": phrases.get(f, f).capitalize(), "Strength": round(abs(float(contrib[f])), 3)} for f in top]
st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
st.caption("Contributions are exact tree contributions from the model (log-odds scale), shown in plain language.")

if weather_mode != "Manual scenario":
    with st.expander("Weather used (NOAA GHCN-Daily, day before the flight)"):
        if weather_note:
            st.info(weather_note)
        for tag, (iata, res) in weather_tables.items():
            st.markdown(f"**{tag.capitalize()}: {iata}**: observations of {res['date']}")
            t = res["stations"].rename(columns={"station": "station_id", "station_name": "station"}).copy()
            t.loc[len(t)] = ["PRCP 3-day total", "", "(same station)", np.nan, res["prcp_3d"]]
            t["value"] = t["value"].astype(float)
            st.dataframe(t[["element", "value", "station", "distance_km"]], width="stretch", hide_index=True)
        st.caption("Units: precipitation, snow and snow depth in mm; temperature in °C; wind in m/s; fog and thunder are 0/1 flags. "
                   "Missing values are passed to the model as missing.")

with st.expander("Model inputs and when each one is known"):
    st.dataframe(pd.DataFrame({"input": feats, "value": [str(row[k]) for k in feats]}), width="stretch", hide_index=True)
    avail = lib.a4_csv("feature_availability.csv")
    if not avail.empty:
        st.dataframe(avail, width="stretch", hide_index=True)

# --------------------------------------------------------------------------
# Live airport activity — informational only, does not feed the model
# --------------------------------------------------------------------------
st.divider()
st.subheader("Live airport activity (OpenSky)")
st.caption("Live ADS-B snapshot near the selected origin, for operational context only. It does not feed the model.")
try:
    snap = _cached_live_snapshot(origin)
except OpenSkyError as exc:
    st.info(f"Live data unavailable right now ({exc})")
except Exception as exc:  # offline / sandbox
    st.info(f"Live data unavailable right now ({type(exc).__name__})")
else:
    lc1, lc2, lc3 = st.columns(3)
    lc1.metric(f"Aircraft near {origin}", snap.total_aircraft)
    lc2.metric("Airborne", snap.airborne)
    lc3.metric("On ground", snap.on_ground)
    as_of = dt.datetime.utcfromtimestamp(snap.generated_unix).strftime("%Y-%m-%d %H:%M UTC")
    st.caption(f"As of {as_of} · source: OpenSky Network")
