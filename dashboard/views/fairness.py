"""Fairness & bias evaluation — per-group performance and gap metrics."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

import lib


st.title("Fairness & Bias Evaluation")

GROUPS = {
    "Carrier": "fairness_carrier.csv",
    "Origin hub tier": "fairness_origin_hub_tier.csv",
    "Geographic quadrant": "fairness_geo_quadrant.csv",
}

overall = lib.load_metrics().get("classifier", {}).get("auc_roc")

summary_rows = []
tabs = st.tabs(list(GROUPS))
for tab, (label, fname) in zip(tabs, GROUPS.items()):
    df = lib.load_csv(fname)
    with tab:
        if df.empty:
            st.warning(f"`assets/{fname}` not found.")
            continue
        eo = lib.equalised_odds_gap(df)
        dp = lib.demographic_parity_gap(df)
        summary_rows.append({"Grouping": label, "Equalised-odds gap": eo,
                             "Demographic-parity gap": dp, "Groups": len(df)})
        m1, m2, m3 = st.columns(3)
        m1.metric("Equalised-odds gap", f"{eo:.3f}")
        m2.metric("Demographic-parity gap", f"{dp:.3f}")
        m3.metric("Groups (n ≥ 200)", f"{len(df)}")

        fig = px.bar(
            df.sort_values("auc"), x="auc", y="group", orientation="h",
            template=lib.PLOTLY_TEMPLATE, color="auc", color_continuous_scale="Blues",
            labels={"auc": "AUC-ROC", "group": ""},
        )
        if overall:
            fig.add_vline(x=overall, line_dash="dash", line_color=lib.ACCENT,
                          annotation_text=f"overall {overall:.3f}")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.scatter(
            df, x="fpr", y="tpr", size="n", color="group", template=lib.PLOTLY_TEMPLATE,
            labels={"fpr": "False-positive rate", "tpr": "True-positive rate"},
        )
        fig2.update_layout(showlegend=len(df) <= 8)
        st.plotly_chart(fig2, width="stretch")

        st.dataframe(
            df.style.format({"n": "{:,.0f}", "auc": "{:.3f}", "f1": "{:.3f}",
                             "tpr": "{:.3f}", "fpr": "{:.3f}", "positive_pred_rate": "{:.3f}"}),
            width="stretch", hide_index=True,
        )

st.divider()
st.subheader("Gap summary")
if summary_rows:
    st.dataframe(
        pd.DataFrame(summary_rows).style.format(
            {"Equalised-odds gap": "{:.3f}", "Demographic-parity gap": "{:.3f}"}),
        width="stretch", hide_index=True,
    )
lib.show_figure("20_fairness_group_auc.png", "Per-group AUC across all three groupings")
