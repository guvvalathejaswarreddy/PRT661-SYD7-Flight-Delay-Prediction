"""Model Performance — walk-forward CV, tuned final models, threshold explorer,
duration regressor, and the COVID-period distribution-shift check."""
from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st

import lib


st.title("Model Performance")

metrics = lib.load_metrics()
wf = lib.load_csv("walk_forward_results.csv")
final = lib.load_csv("final_tuned_model_metrics.csv")
covid = lib.load_csv("covid_distribution_shift_check.csv")
reg_csv = lib.load_csv("duration_regressor_metrics.csv")

# --------------------------------------------------------------------------
# Headline metrics
# --------------------------------------------------------------------------
if metrics:
    c = metrics["classifier"]
    r = metrics["regressor"]
    cols = st.columns(5)
    cols[0].metric("AUC-ROC (2023)", f"{c['auc_roc']:.3f}")
    cols[1].metric("F1", f"{c['f1']:.3f}")
    cols[2].metric("Precision / Recall", f"{c['precision']:.2f} / {c['recall']:.2f}")
    cols[3].metric("Brier score", f"{c['brier']:.3f}")
    cols[4].metric("Regressor MAE", f"{r['mae']:.0f} min", help=f"RMSE {r['rmse']:.0f} · R² {r['r2']:.3f}")
else:
    st.warning("assets/model_metrics.json not found — run `python dashboard/generate_predictions.py`.")

st.divider()

tab_final, tab_wf, tab_thr, tab_reg, tab_covid = st.tabs(
    ["Final tuned models", "Walk-forward folds", "Threshold explorer",
     "Duration regressor", "COVID shift"]
)

# --------------------------------------------------------------------------
with tab_final:
    st.subheader("Tuned final models — 2023 test set")
    if not final.empty:
        show = final.rename(columns={final.columns[0]: "model"}).set_index("model")
        st.dataframe(show.style.format("{:.4f}").background_gradient(cmap="Blues", axis=0),
                     width="stretch")
    a, b, c = st.columns(3)
    with a:
        lib.show_figure("14_roc_curves.png", "ROC curves — final models, 2023")
    with b:
        lib.show_figure("16_calibration_curves.png", "Calibration — final models, 2023")
    with c:
        lib.show_figure("15_confusion_matrices.png", "Confusion matrices @ 0.5")
    lib.show_figure("12_walk_forward_model_comparison.png",
                    "Model comparison across walk-forward folds (mean ± std)")
    lib.show_figure("13_optuna_search_progress.png", "Optuna search — best validation AUC vs trial")

# --------------------------------------------------------------------------
with tab_wf:
    st.subheader("Walk-forward cross-validation")
    if not wf.empty:
        st.dataframe(
            wf[["fold", "model", "train_rows", "test_rows", "auc_roc", "f1",
                "precision", "recall", "brier", "seconds"]]
            .style.format({"auc_roc": "{:.4f}", "f1": "{:.4f}", "precision": "{:.4f}",
                           "recall": "{:.4f}", "brier": "{:.4f}",
                           "train_rows": "{:,.0f}", "test_rows": "{:,.0f}", "seconds": "{:.0f}"}),
            width="stretch", hide_index=True,
        )
        fig = px.line(
            wf, x="fold", y="auc_roc", color="model", markers=True,
            template=lib.PLOTLY_TEMPLATE, labels={"auc_roc": "AUC-ROC", "fold": ""},
        )
        fig.update_yaxes(range=[0.5, 0.75])
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
with tab_thr:
    st.subheader("Decision-threshold explorer")
    preds = lib.load_predictions()
    if preds.empty or "actual_delayed" not in preds.columns:
        st.warning("Needs predictions.parquet with actuals — run `generate_predictions.py`.")
    else:
        thr = st.slider("Threshold", 0.05, 0.95, 0.50, 0.01)
        y = preds["actual_delayed"].to_numpy()
        p = preds["predicted_delay_prob"].to_numpy()
        yhat = p >= thr
        tp = int((yhat & y).sum()); fp = int((yhat & ~y).sum())
        fn = int((~yhat & y).sum()); tn = int((~yhat & ~y).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Precision", f"{prec:.3f}")
        m2.metric("Recall", f"{rec:.3f}")
        m3.metric("F1", f"{f1:.3f}")
        m4.metric("Flagged", f"{yhat.mean() * 100:.1f}%")
        cm = np.array([[tn, fp], [fn, tp]])
        fig = px.imshow(
            cm, text_auto=",d", color_continuous_scale="Blues",
            x=["Pred on-time", "Pred delayed"], y=["Actual on-time", "Actual delayed"],
            template=lib.PLOTLY_TEMPLATE, aspect="auto",
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------
with tab_reg:
    st.subheader("Delay-duration regression (delayed flights only)")
    if metrics:
        r = metrics["regressor"]
        a, b, c = st.columns(3)
        a.metric("MAE", f"{r['mae']:.1f} min")
        b.metric("RMSE", f"{r['rmse']:.1f} min")
        c.metric("R²", f"{r['r2']:.3f}")
    lib.show_figure("19_duration_regressor_predicted_vs_actual.png",
                    "Predicted vs actual delay minutes; residual distribution")

# --------------------------------------------------------------------------
with tab_covid:
    st.subheader("COVID-19 period distribution-shift check")
    if not covid.empty:
        st.dataframe(
            covid.style.format({"actual_delay_rate": "{:.1%}", "auc": "{:.4f}",
                                "f1": "{:.4f}", "n": "{:,.0f}"}),
            width="stretch", hide_index=True,
        )
    lib.show_figure("21_covid_distribution_shift.png",
                    "AUC and actual delay rate: in-distribution vs COVID period")
