"""About — pipeline, method, model card."""
from __future__ import annotations

import streamlit as st

import lib

st.title("About")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Pipeline")
    st.markdown(
        """
| Layer | Component |
|---|---|
| Ingestion | BTS on-time performance + OpenFlights metadata |
| Storage | Local raw / curated / feature zones |
| Processing | DuckDB clean + join, feature engineering |
| Modelling | LogReg, Random Forest, XGBoost, LightGBM + duration regressor |
| Dashboard | Streamlit |
"""
    )
with col2:
    st.subheader("Method")
    st.markdown(
        """
| Item | Value |
|---|---|
| Cross-validation | Walk-forward (expanding window) |
| Excluded from training | 2020–2021 |
| Production fold | train 2018 + 2019 + 2022 → test 2023 |
| Tuning | Optuna |
| Interpretability | SHAP |
| Fairness | carrier, hub tier, geography |
"""
    )

st.subheader("Model card")
m = lib.load_metrics()
if m:
    c, r = m["classifier"], m["regressor"]
    st.markdown(
        f"""
| | |
|---|---|
| Classifier | {c['model']} |
| Training | {m['training']} of {m['train_years']} |
| Test | {m['test_rows']:,} flights, {m['test_year']} |
| AUC-ROC / F1 | {c['auc_roc']:.3f} / {c['f1']:.3f} |
| Precision / Recall @0.5 | {c['precision']:.3f} / {c['recall']:.3f} |
| Brier | {c['brier']:.3f} |
| Duration regressor | MAE {r['mae']:.1f} min, RMSE {r['rmse']:.1f}, R² {r['r2']:.3f} |
"""
    )
