"""Data & EDA — Five Vs, exploratory figures, interactive views."""
from __future__ import annotations

import streamlit as st

import lib


st.title("Data & Exploratory Analysis")

five_vs = lib.load_csv("five_vs_summary.csv")
if not five_vs.empty:
    st.subheader("Five Vs")
    st.dataframe(five_vs, width="stretch", hide_index=True)

st.divider()
st.subheader("Exploratory figures")

pairs = [
    ("01_class_balance.png", "Class balance — overall delay rate"),
    ("02_monthly_delay_trend.png", "Monthly delay rate, 2018–2023 (COVID period shaded)"),
    ("03_hour_dow_heatmap.png", "Delay rate by departure hour × day of week"),
    ("04_delay_rate_by_carrier.png", "Delay rate by carrier"),
    ("05_top_airports_volume_delay.png", "Busiest origin airports — volume & delay rate"),
    ("06_delay_duration_violin_by_carrier.png", "Delay-duration distribution by carrier"),
    ("07_distance_vs_delay_hexbin.png", "Route distance vs delay minutes (density)"),
    ("08_correlation_heatmap.png", "Correlation of engineered numeric features"),
]
for i in range(0, len(pairs), 2):
    cols = st.columns(2)
    for col, (fname, cap) in zip(cols, pairs[i:i + 2]):
        with col:
            lib.show_figure(fname, cap)

st.divider()
st.subheader("NOAA weather coverage (Assessment 4)")
c1, c2 = st.columns([1, 1])
with c1:
    lib.show_figure("22_weather_coverage.png", "Share of airport-days with a value, per weather element (2018–2023)")
with c2:
    st.markdown("Each airport is matched to the nearest GHCN-Daily station that reports each element (median distance 0.8 km for precipitation). "
                "Missing values are passed to the model as missing, not filled in.")

st.divider()
st.subheader("Interactive views")
tab1, tab2, tab3 = st.tabs(["Monthly delay by carrier", "US airport map", "Carrier → hub tier"])
with tab1:
    lib.show_figure("09_interactive_monthly_delay_by_carrier.html")
with tab2:
    lib.show_figure("10_interactive_us_airport_map.html")
with tab3:
    lib.show_figure("11_interactive_sunburst_carrier_hub.html")
