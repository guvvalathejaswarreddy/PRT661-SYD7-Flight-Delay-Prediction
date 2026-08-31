"""Overview — operational view of the 2023 predictions from the trained model."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

import lib


df = lib.load_predictions()
metrics = lib.load_metrics()

st.title("Operations Overview")

if not lib.no_predictions_banner():
    st.stop()

# --------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    dmin, dmax = df["flight_date"].min().date(), df["flight_date"].max().date()
    date_range = st.date_input("Flight date range", value=(dmin, dmax),
                               min_value=dmin, max_value=dmax)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
    else:
        d0, d1 = dmin, dmax

    carriers = sorted(df["carrier"].unique())
    carrier_sel = st.multiselect("Carrier", carriers, default=[],
                                 placeholder="All carriers")

    tiers = sorted(df["origin_hub_tier"].unique()) if "origin_hub_tier" in df else []
    tier_sel = st.multiselect("Origin hub tier", tiers, default=[],
                              placeholder="All hub tiers")

    threshold = st.slider("Decision threshold (P[delay] ≥ …)", 0.05, 0.95, 0.50, 0.05)


mask = (df["flight_date"].dt.date >= d0) & (df["flight_date"].dt.date <= d1)
if carrier_sel:
    mask &= df["carrier"].isin(carrier_sel)
if tier_sel:
    mask &= df["origin_hub_tier"].isin(tier_sel)
f = df.loc[mask].copy()

if f.empty:
    st.warning("No flights match the current filters.")
    st.stop()

f["flagged"] = f["predicted_delay_prob"] >= threshold

# --------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------
has_actuals = "actual_delayed" in f.columns
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Flights in view", f"{len(f):,}")
k2.metric("Predicted delay rate", f"{f['flagged'].mean() * 100:.1f}%")
if has_actuals:
    k3.metric("Actual delay rate", f"{f['actual_delayed'].mean() * 100:.1f}%")
else:
    k3.metric("Actual delay rate", "—")
k4.metric(
    "Avg predicted delay (flagged)",
    f"{f.loc[f['flagged'], 'predicted_delay_minutes'].mean():.0f} min" if f["flagged"].any() else "—",
)
if has_actuals:
    tp = int((f["flagged"] & f["actual_delayed"]).sum())
    fp = int((f["flagged"] & ~f["actual_delayed"]).sum())
    fn = int((~f["flagged"] & f["actual_delayed"]).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    k5.metric("Precision / Recall @ threshold", f"{prec * 100:.0f}% / {rec * 100:.0f}%")
else:
    k5.metric("Model AUC", f"{metrics['classifier']['auc_roc']:.3f}" if metrics else "—")

st.divider()

# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Delay rate by carrier — predicted vs actual")
    g = f.groupby("carrier").agg(
        predicted=("flagged", "mean"),
        actual=("actual_delayed", "mean") if has_actuals else ("flagged", "mean"),
        flights=("flagged", "size"),
    ).reset_index()
    g[["predicted", "actual"]] *= 100
    long = g.melt(id_vars=["carrier", "flights"],
                  value_vars=["predicted", "actual"] if has_actuals else ["predicted"],
                  var_name="kind", value_name="delay_rate")
    fig = px.bar(
        long.sort_values("delay_rate"), x="carrier", y="delay_rate", color="kind",
        barmode="group", template=lib.PLOTLY_TEMPLATE,
        color_discrete_map={"predicted": lib.BRAND, "actual": lib.ACCENT},
        labels={"delay_rate": "Delay rate (%)", "kind": ""},
    )
    fig.update_layout(legend=dict(orientation="h", y=1.12, x=0), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Busiest origin airports — flights & predicted delay rate")
    a = f.groupby("origin").agg(
        flights=("flagged", "size"),
        pred_rate=("flagged", "mean"),
    ).reset_index().sort_values("flights", ascending=False).head(15)
    a["pred_rate"] *= 100
    fig = px.bar(
        a.sort_values("flights"), x="flights", y="origin", orientation="h",
        color="pred_rate", color_continuous_scale="OrRd", template=lib.PLOTLY_TEMPLATE,
        labels={"flights": "Flights in view", "origin": "", "pred_rate": "Pred. delay %"},
    )
    fig.update_layout(margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Predicted delay-probability distribution")
    fig = px.histogram(
        f, x="predicted_delay_prob", nbins=40, template=lib.PLOTLY_TEMPLATE,
        color_discrete_sequence=[lib.BRAND],
        labels={"predicted_delay_prob": "P(delay ≥ 15 min)"},
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color=lib.ACCENT,
                  annotation_text=f"threshold {threshold:.2f}")
    fig.update_layout(margin=dict(t=10), yaxis_title="Flights")
    st.plotly_chart(fig, width="stretch")

with c2:
    st.subheader("Weekly delay rate over 2023")
    w = (
        f.set_index("flight_date")
        .resample("W")
        .agg(predicted=("flagged", "mean"),
             actual=("actual_delayed", "mean") if has_actuals else ("flagged", "mean"))
        .reset_index()
    )
    w[["predicted", "actual"]] *= 100
    fig = px.line(
        w.melt(id_vars="flight_date",
               value_vars=["predicted", "actual"] if has_actuals else ["predicted"],
               var_name="kind", value_name="rate"),
        x="flight_date", y="rate", color="kind", template=lib.PLOTLY_TEMPLATE,
        color_discrete_map={"predicted": lib.BRAND, "actual": lib.ACCENT},
        labels={"rate": "Delay rate (%)", "flight_date": "", "kind": ""},
    )
    fig.update_layout(legend=dict(orientation="h", y=1.15, x=0), margin=dict(t=10))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
# Flight-level table
# --------------------------------------------------------------------------
st.subheader("Flight-level predictions")
show_cols = [
    "flight_date", "carrier", "origin", "destination", "distance",
    "predicted_delay_prob", "predicted_delay_minutes", "flagged",
]
if has_actuals:
    show_cols += ["actual_delayed", "actual_delay_minutes"]
table = (
    f[show_cols]
    .sort_values("predicted_delay_prob", ascending=False)
    .rename(columns={"flagged": "predicted_delayed"})
    .head(500)
    .reset_index(drop=True)
)
st.dataframe(
    table,
    width="stretch",
    column_config={
        "predicted_delay_prob": st.column_config.ProgressColumn(
            "P(delay)", min_value=0.0, max_value=1.0, format="%.2f"),
        "predicted_delay_minutes": st.column_config.NumberColumn("Pred. min", format="%.0f"),
        "actual_delay_minutes": st.column_config.NumberColumn("Actual min", format="%.0f"),
        "distance": st.column_config.NumberColumn("Miles", format="%.0f"),
    },
)
st.download_button(
    "Download filtered predictions (CSV)",
    f[show_cols].to_csv(index=False).encode(),
    file_name="flight_delay_predictions_filtered.csv",
    mime="text/csv",
)
