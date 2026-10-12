# Flight Delay Prediction: Streamlit Dashboard

Multi-page dashboard for the final (Assessment 4) model: a calibrated LightGBM classifier with 26 inputs (including
prior-day NOAA weather) and quantile duration models. It reads only files exported by the analysis notebook
(`notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb`, sections 15-24), so it starts without the 37.9M-row dataset.

## Pages

| Page | Contents |
|---|---|
| **Overview** | 2023 sample: filters, KPIs, predicted vs actual delay rate, probability distribution, trend |
| **Predict a Flight** | Carrier / route / date / time -> calibrated probability, alert level with 2023 precision, recall and workload, 80% duration range with reliability warning, plain-language reasons, live NOAA prior-day lookup, OpenSky context panel |
| **Model Performance** | Ablation, calibration, thresholds and alert workload, duration range |
| **Explainability** | SHAP importance and contribution of the weather inputs |
| **Fairness** | Base-rate and calibration framing, equalised-odds gaps, per-carrier thresholds |
| **Data & EDA** | Five Vs, weather coverage, exploratory figures |
| **About** | Model card, governance, limitations |

## Running the dashboard

```bash
python -m pip install -r dashboard/requirements.txt   # pinned versions
streamlit run dashboard/streamlit_app.py              # or ./run_dashboard.sh
```

Served at http://localhost:8501.

## Data it reads

```
dashboard/
├── streamlit_app.py            # entry point (st.navigation)
├── views/                      # the 7 page bodies
├── lib.py                      # paths, loaders, calibration, styling
├── noaa_live.py                # previous-day NOAA lookup (nearest mapped station)
├── data/
│   └── a4_predictions_sample.parquet   # 300k stratified 2023 sample used by the pages
└── assets/
    ├── a4/                     # FINAL exports from the notebook: model + duration bundles, calibrated thresholds,
    │                           #   operating points, fairness, ablation, monitoring, station map, phrases
    ├── figures/                # analysis figures (PNG + interactive HTML)
    └── *.csv, best_hyperparameters.json   # Assessment 2 walk-forward / COVID / Five Vs tables still shown
```

To refresh the dashboard, re-run the notebook and copy its `a4_outputs` folder into `assets/a4/`.

## Checks

```bash
python3 -m pytest tests -q       # 11 tests incl. prior-day weather and calibration
python3 monitoring/monitor.py    # monthly AUC, calibration gap, alert share, PSI
```

Limitations: figures here come from a 300,000-flight sample, so they match the full 2023 test set closely but not
exactly; the dashboard has no authentication and is intended for a local or trusted network.
