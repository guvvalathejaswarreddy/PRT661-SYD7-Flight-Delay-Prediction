"""Explainability — SHAP feature attributions for the best model."""
from __future__ import annotations

import streamlit as st

import lib

st.title("Explainability — SHAP")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Beeswarm")
    lib.show_figure("17_shap_summary_beeswarm.png")
with col2:
    st.subheader("Mean |SHAP|")
    lib.show_figure("18_shap_mean_abs_importance.png")
