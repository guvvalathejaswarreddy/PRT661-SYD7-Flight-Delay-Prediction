"""Model Performance — Assessment 4 evaluation (ablation, calibration, thresholds and workload, duration)
followed by the Assessment 2 baseline material (walk-forward folds, COVID check)."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

import lib

st.title("Model Performance")

summary = lib.a4_json("a4_summary.json")
abl = lib.a4_csv("a4_ablation_results.csv")
cal = lib.a4_csv("a4_calibration_summary.csv")
ops = lib.a4_csv("a4_operating_points.csv")
dur = lib.a4_csv("a4_duration_quantile_results.csv")
alerts = lib.a4_csv("a4_alert_workload_top_airports.csv")
wf = lib.load_csv("walk_forward_results.csv")
final = lib.load_csv("final_tuned_model_metrics.csv")
covid = lib.load_csv("covid_distribution_shift_check.csv")

if summary and not cal.empty:
    raw, cl = cal.iloc[0], cal.iloc[1]
    cols = st.columns(5)
    cols[0].metric("AUC-ROC (2023 test)", f"{cl['auc_roc']:.3f}", help="Deployed model: V2, fitted on the first 85% of training dates")
    cols[1].metric("Average precision", f"{cl['avg_precision']:.3f}", help="Delay base rate is 0.206")
    cols[2].metric("Calibration error (ECE)", f"{cl['ECE']:.3f}", delta=f"{cl['ECE'] - raw['ECE']:+.3f} vs raw score", delta_color="inverse")
    cols[3].metric("Brier score", f"{cl['brier']:.3f}", delta=f"{cl['brier'] - raw['brier']:+.3f} vs raw score", delta_color="inverse")
    cols[4].metric("Duration skill vs naive", "≈ 0", help="See the Duration tab")
st.caption("Train 2018 + 2019 + 2022 (COVID years excluded) → test 2023. Thresholds and calibration are fitted on held-out late-2022 validation data, never on the test year.")
st.divider()

tab_abl, tab_cal, tab_thr, tab_dur, tab_a2, tab_wf, tab_covid = st.tabs(
    ["Ablation", "Calibration", "Thresholds & workload", "Duration (honest)", "A2 baseline models", "Walk-forward folds", "COVID shift"])

# --------------------------------------------------------------------------
with tab_abl:
    st.subheader("What the corrections and the weather data changed")
    if not abl.empty:
        names = {"A2 (reference, published)": "A2 (as published)", "V1_leakage_safe": "V1 — leakage-safe features",
                 "V2_plus_prior_day_weather": "V2 — + prior-day NOAA weather (deployed)",
                 "V3_same_day_weather_UPPER_BOUND": "V3 — same-day weather (upper bound, NOT deployable)"}
        show = abl.assign(variant=abl["variant"].map(names).fillna(abl["variant"])).set_index("variant")
        st.dataframe(show.style.format("{:.4f}"), width="stretch")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        lib.show_figure("23_ablation_auc.png", "AUC-ROC on the 2023 test set")
    with c2:
        st.markdown(
            """
**Reading the table**
- **V1 vs A2 (+0.0006):** replacing the 7-row window with an exact 7-calendar-day window and computing hub features from training years only changed accuracy almost nothing, so the A2 result was not inflated by those choices. The definitions are now exact and tested.
- **V2 vs V1 (+0.0080 AUC, 95% interval +0.0076 to +0.0085):** prior-day weather gives a small but statistically reliable gain.
- **V3:** same-day weather would add about +0.025 more, but it is only known after the day ends. A weather *forecast* feed would be needed to use it, so V3 is reported as an upper bound only.
"""
        )
    leak = lib.a4_csv("leakage_audit.csv")
    if not leak.empty:
        st.subheader("Automated leakage checks")
        st.dataframe(leak.assign(passed=leak["passed"].map({True: "passed", False: "FAILED"})), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_cal:
    st.subheader("From a ranking score to a probability")
    st.markdown("With balanced class weights the raw score ranks flights well but is not a probability (it averages ≈ 0.48 when 20.6% of flights are late). A Platt calibrator fitted on held-out validation data fixes that without changing the ranking (AUC identical).")
    if not cal.empty:
        st.dataframe(cal.set_index("version").style.format("{:.4f}"), width="stretch")
        st.caption("Precision, recall and F1 in the calibrated row are at a 0.5 probability threshold, which is not a sensible alert level once the score is a true probability. Use the operating points in the next tab.")
    lib.show_figure("24_calibration_before_after.png", "Reliability curve before and after calibration (2023 test)")

# --------------------------------------------------------------------------
with tab_thr:
    st.subheader("Choosing an alert level: precision, recall and workload")
    if not ops.empty:
        t = ops.rename(columns={"flights flagged": "share of flights flagged"})
        st.dataframe(t.style.format({"threshold": "{:.3f}", "share of flights flagged": "{:.1%}", "alerts per day (network)": "{:,.0f}",
                                      "precision": "{:.3f}", "recall": "{:.3f}", "F1": "{:.3f}",
                                      "false alarms per 10 alerts": "{:.1f}", "lift over base rate": "{:.2f}x"}, na_rep="—"),
                     width="stretch", hide_index=True)
    st.markdown("Thresholds are chosen on validation data and evaluated once on 2023. A team that can only act on **10% of departures** gets alerts that are right **44%** of the time (2.1x the base rate). Flagging 42% of flights, as A2's 0.5 rule did, is right only 31% of the time.")
    lib.show_figure("25_threshold_tradeoff_and_workload.png", "Trade-off curves and alert workload at the 10%-budget threshold")
    preds = lib.load_a4_sample()
    if not preds.empty:
        st.subheader("Threshold explorer (2023 sample)")
        thr = st.slider("Alert when calibrated P(delay) ≥", 0.05, 0.80, 0.37, 0.01)
        y = preds["actual_delayed"].to_numpy(); p = preds["predicted_delay_prob"].to_numpy(); yhat = p >= thr
        tp = int((yhat & y).sum()); fp = int((yhat & ~y).sum()); fn = int((~yhat & y).sum()); tn = int((~yhat & ~y).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0; rec = tp / (tp + fn) if (tp + fn) else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.3f}"); m2.metric("Recall", f"{rec:.3f}")
        m3.metric("Flights flagged", f"{yhat.mean() * 100:.1f}%")
        m4.metric("Alerts / day (network)", f"{yhat.mean() * summary.get('n_test', 6_740_139) / 365:,.0f}")
        fig = px.imshow(np.array([[tn, fp], [fn, tp]]), text_auto=",d", color_continuous_scale="Blues",
                        x=["No alert", "Alert"], y=["Actually on time", "Actually delayed"], template=lib.PLOTLY_TEMPLATE, aspect="auto")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")
    if not alerts.empty:
        st.subheader("Alerts per day at the busiest airports (10% budget threshold)")
        st.dataframe(alerts.style.format({"average alerts / day": "{:.1f}", "95th percentile alerts / day": "{:.0f}"}), width="stretch", hide_index=True)

# --------------------------------------------------------------------------
with tab_dur:
    st.subheader("Delay duration: the model cannot predict minutes reliably")
    st.error("Result: the minutes model has no practical skill. Its median estimate is no more accurate than always guessing the typical delay (mean error 43.7 vs 44.4 min; R² −0.07). The dashboard therefore shows a wide range, not a single number.")
    if not dur.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("80% interval coverage", f"{dur['coverage_80pct'].iloc[0]:.1%}", help="Nominal 80%")
        c2.metric("Median interval width", f"{dur['median_interval_width_min'].iloc[0]:.0f} min")
        c3.metric("Skill vs naive quantile (q50)", f"{dur.loc[dur['quantile'] == 0.5, 'skill vs naive'].iloc[0]:.1%}")
        st.dataframe(dur.drop(columns=["coverage_80pct", "median_interval_width_min"]).style.format("{:.3f}"), width="stretch", hide_index=True)
    lib.show_figure("26_duration_intervals.png", "Predicted 10–90% range against actual delay (delayed flights only)")

# --------------------------------------------------------------------------
with tab_a2:
    st.subheader("Assessment 2 tuned models — 2023 test set")
    if not final.empty:
        show = final.rename(columns={final.columns[0]: "model"}).set_index("model")
        st.dataframe(show.style.format("{:.4f}"), width="stretch")
    a, b, c = st.columns(3)
    with a:
        lib.show_figure("14_roc_curves.png", "ROC curves — A2 final models")
    with b:
        lib.show_figure("16_calibration_curves.png", "Calibration — A2 raw scores (before the A4 calibrator)")
    with c:
        lib.show_figure("15_confusion_matrices.png", "Confusion matrices @ 0.5")
    lib.show_figure("13_optuna_search_progress.png", "Optuna search — best validation AUC vs trial")

with tab_wf:
    st.subheader("Walk-forward cross-validation (Assessment 2)")
    if not wf.empty:
        st.dataframe(wf[["fold", "model", "train_rows", "test_rows", "auc_roc", "f1", "precision", "recall", "brier", "seconds"]]
                     .style.format({"auc_roc": "{:.4f}", "f1": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}", "brier": "{:.4f}",
                                    "train_rows": "{:,.0f}", "test_rows": "{:,.0f}", "seconds": "{:.0f}"}),
                     width="stretch", hide_index=True)
        fig = px.line(wf, x="fold", y="auc_roc", color="model", markers=True, template=lib.PLOTLY_TEMPLATE,
                      labels={"auc_roc": "AUC-ROC", "fold": ""})
        fig.update_yaxes(range=[0.5, 0.75])
        st.plotly_chart(fig, width="stretch")
    lib.show_figure("12_walk_forward_model_comparison.png", "Model comparison across walk-forward folds (mean ± std)")

with tab_covid:
    st.subheader("COVID-19 period distribution-shift check")
    if not covid.empty:
        st.dataframe(covid.style.format({"actual_delay_rate": "{:.1%}", "auc": "{:.4f}", "f1": "{:.4f}", "n": "{:,.0f}"}),
                     width="stretch", hide_index=True)
    lib.show_figure("21_covid_distribution_shift.png", "AUC and actual delay rate: in-distribution vs COVID period")
