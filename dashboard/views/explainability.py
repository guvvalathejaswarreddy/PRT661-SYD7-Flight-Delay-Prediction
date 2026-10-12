"""Explainability — what drives the delay probability, in plain language."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import lib

st.title("Explainability")
summary = lib.a4_json("a4_summary.json")
phrases = lib.a4_json("a4_feature_phrases.json")
imp = lib.a4_csv("a4_mean_abs_contribution.csv")

st.markdown(
    "Contributions are exact tree contributions from the deployed LightGBM model (log-odds scale). "
    "A positive contribution pushes a flight towards *delayed*; a negative one pushes it towards *on time*. "
    "The Predict page shows the top reasons for one flight in plain language."
)
if summary:
    c1, c2, c3 = st.columns(3)
    c1.metric("Schedule + recent-delay inputs", f"{(1 - summary['weather_share_of_contribution']):.0%} of total effect")
    c2.metric("Prior-day weather inputs", f"{summary['weather_share_of_contribution']:.0%} of total effect")
    c3.metric("Largest single driver", "Scheduled departure hour")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Beeswarm — deployed model")
    lib.show_figure("28_shap_beeswarm_v2.png")
with col2:
    st.subheader("Mean absolute contribution")
    lib.show_figure("29_shap_importance_v2.png")

if not imp.empty:
    imp = imp.rename(columns={imp.columns[0]: "feature"})
    imp["in plain words"] = imp["feature"].map(lambda f: phrases.get(f, f))
    st.subheader("All inputs, ranked")
    st.dataframe(imp[["feature", "in plain words", "mean_abs_contribution"]].style.format({"mean_abs_contribution": "{:.4f}"}),
                 width="stretch", hide_index=True)

with st.expander("Assessment 2 model (before the corrections): SHAP plots"):
    a, b = st.columns(2)
    with a:
        lib.show_figure("17_shap_summary_beeswarm.png")
    with b:
        lib.show_figure("18_shap_mean_abs_importance.png")
