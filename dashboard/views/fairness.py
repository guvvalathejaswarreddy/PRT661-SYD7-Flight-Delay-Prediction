"""Fairness — what the carrier / airport gaps do and do not show (Assessment 4 reframing)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import lib

st.title("Fairness and Bias Evaluation")
st.info(
    "**How to read this page.** Carriers differ in schedule structure, hub geography and base delay rates, and a carrier is an "
    "operating group, not a protected attribute of any passenger. A gap in how often a group is flagged is therefore **not, by itself, "
    "evidence of unfair treatment**. We check four things per group: the *base rate* (how often it really is late), *calibration* "
    "(does a 30% score mean about 30% late for this group?), *AUC*, and the error-rate gaps at one global threshold."
)

GROUPS = {"Carrier": "a4_fairness_carrier.csv", "Origin hub tier": "a4_fairness_origin_hub_tier.csv",
          "Geographic quadrant": "a4_fairness_quadrant.csv"}
gaps = lib.a4_csv("a4_fairness_gap_summary.csv")
thr = lib.a4_json("a4_thresholds.json")

tabs = st.tabs(list(GROUPS) + ["Per-carrier thresholds (mitigation)"])
for tab, (label, fname) in zip(tabs, GROUPS.items()):
    df = lib.a4_csv(fname)
    with tab:
        if df.empty:
            st.warning(f"`assets/a4/{fname}` not found.")
            continue
        g = gaps[gaps.iloc[:, 0] == label].iloc[0] if not gaps.empty and (gaps.iloc[:, 0] == label).any() else None
        if g is not None:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Base-rate gap", f"{g['base_rate_gap']:.3f}", help="Largest minus smallest real delay rate between groups")
            m2.metric("Flagged-rate gap", f"{g['demographic_parity_gap']:.3f}")
            m3.metric("Correlation: base rate vs flagged rate", f"{g['corr(base_rate, flagged_rate)']:.2f}")
            m4.metric("Worst calibration gap", f"{g['max_abs_calibration_gap']:.3f}", help="Mean predicted minus observed delay rate")
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(df, x="base_rate", y="flagged_rate", size="n", text="group", template=lib.PLOTLY_TEMPLATE,
                             labels={"base_rate": "Observed delay rate (base rate)", "flagged_rate": "Share flagged at the global threshold"})
            fig.update_traces(textposition="top center")
            st.plotly_chart(fig, width="stretch")
            st.caption("Groups that are late more often are flagged more often: the pattern follows the base rate.")
        with c2:
            fig2 = px.bar(df.sort_values("calibration_gap"), x="calibration_gap", y="group", orientation="h", template=lib.PLOTLY_TEMPLATE,
                          labels={"calibration_gap": "Mean predicted − observed delay rate", "group": ""})
            fig2.add_vline(x=0, line_color="black")
            st.plotly_chart(fig2, width="stretch")
            st.caption("Close to zero means the probabilities mean the same thing for every group.")
        st.dataframe(df.style.format({"n": "{:,.0f}", "base_rate": "{:.3f}", "mean_predicted": "{:.3f}", "calibration_gap": "{:+.3f}",
                                      "ECE": "{:.3f}", "auc": "{:.3f}", "tpr": "{:.3f}", "fpr": "{:.3f}",
                                      "flagged_rate": "{:.3f}", "precision": "{:.3f}"}), width="stretch", hide_index=True)

with tabs[-1]:
    mit = lib.a4_csv("a4_fairness_per_carrier_threshold_mitigation.csv")
    st.subheader("Mitigation tested: per-carrier thresholds (equal opportunity)")
    st.markdown(
        "Each carrier gets the threshold at which it catches the same share of its real delays on validation data. "
        "This **narrows the gaps** (true-positive-rate gap 0.51 → 0.25, false-positive-rate gap 0.57 → 0.28, flagged-rate gap 0.58 → 0.28) "
        "but is a **trade-off**: carriers with a low base rate get a lower threshold, more false alarms and lower precision. "
        "It is a policy choice for the operations team, not an automatic improvement."
    )
    if not mit.empty:
        st.dataframe(mit.style.format({"threshold": "{:.3f}", "tpr": "{:.3f}", "fpr": "{:.3f}", "flagged_rate": "{:.3f}", "precision": "{:.3f}"}),
                     width="stretch", hide_index=True, height=35 * (len(mit) + 1) + 3)

st.divider()
if not gaps.empty:
    st.subheader("Gap summary")
    st.dataframe(gaps.rename(columns={gaps.columns[0]: "grouping"}).style.format({c: "{:.3f}" for c in gaps.columns[1:]}),
                 width="stretch", hide_index=True)
lib.show_figure("27_fairness_base_rate_and_calibration.png", "Flagged rate follows base rate; calibration after the A4 calibrator")
