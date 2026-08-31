"""Navigation entry point. Page bodies live in dashboard/views/."""
from __future__ import annotations

import streamlit as st

import lib

lib.page_config()

nav = st.navigation(
    {
        "Operations": [
            st.Page("views/overview.py", title="Overview", default=True),
            st.Page("views/predict.py", title="Predict a Flight"),
        ],
        "Model": [
            st.Page("views/performance.py", title="Model Performance"),
            st.Page("views/explainability.py", title="Explainability"),
            st.Page("views/fairness.py", title="Fairness"),
        ],
        "Reference": [
            st.Page("views/data_eda.py", title="Data & EDA"),
            st.Page("views/about.py", title="About"),
        ],
    }
)

nav.run()
