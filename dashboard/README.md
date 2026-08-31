# Flight Delay Prediction — Streamlit Dashboard

Multi-page dashboard driven by the **actual trained model** (tuned LightGBM
classifier + duration regressor; walk-forward production fold: train
2018 + 2019 + 2022 → test 2023).

## Pages

| Page | Contents |
|---|---|
| **Overview** (home) | 2023 predictions: filters, KPIs, predicted-vs-actual delay rate by carrier/airport, probability distribution, weekly trend, flight-level table + CSV export |
| **Predict a Flight** | Enter carrier / route / date / time → the trained classifier + duration regressor return P(delay), a verdict at the chosen threshold, estimated minutes late, and a plain-language "what's driving this" |
| **Model Performance** | Final tuned-model metrics, walk-forward folds, interactive decision-threshold explorer (live confusion matrix), duration regressor, COVID distribution-shift check |
| **Explainability** | SHAP beeswarm + mean-\|SHAP\| importance, with a written read-out |
| **Fairness** | Per-group AUC / TPR / FPR and equalised-odds & demographic-parity gaps by carrier, hub tier, geography |
| **Data & EDA** | Five Vs table, exploratory figures, interactive Plotly views |
| **About** | Project summary, method, model card, known limitations |

## Running the dashboard

From the repository's `software/` folder:

```bash
python -m pip install -r dashboard/requirements.txt
streamlit run dashboard/streamlit_app.py      # or ./run_dashboard.sh
```

The app is served at http://localhost:8501.

## Data it reads

Everything the app needs is committed under `dashboard/`:

```
dashboard/
├── streamlit_app.py            # entry point — wires st.navigation only
├── views/                      # the 6 page bodies (overview, performance, …)
├── lib.py                      # shared paths / loaders / styling
├── generate_predictions.py     # (re)builds predictions.parquet from the model
├── .streamlit/config.toml      # theme + viewer toolbar
├── data/
│   └── predictions.parquet     # 300k stratified sample of 2023 predictions (committed)
└── assets/
    ├── model_classifier.joblib # fitted tuned-LightGBM classifier pipeline (Predict page)
    ├── model_regressor.joblib  # fitted duration-regressor pipeline (Predict page)
    ├── predict_reference.json  # per-airport / per-carrier / per-route lookups for the form
    ├── model_metrics.json      # metrics from the last generate_predictions.py run
    ├── figures/                # all 21 analysis figures (PNG + interactive HTML)
    └── *.csv                   # walk-forward, fairness, COVID, Five Vs tables
```

`generate_predictions.py` writes the two `.joblib` models and
`predict_reference.json` alongside the metrics — the **Predict a Flight** page
loads those directly, so it uses exactly the model that produced the dashboard
numbers.

## Regenerate predictions from the model

```bash
python dashboard/generate_predictions.py            # ~2 min, 5M-row training sample
python dashboard/generate_predictions.py --full     # every training row (more RAM/time)
python dashboard/generate_predictions.py --write-full  # also emit predictions_full.parquet
```

It re-fits the tuned LightGBM classifier + duration regressor from
`assets/best_hyperparameters.json` on 2018 + 2019 + 2022, scores 2023, writes a
stratified sample to `data/predictions.parquet`, and refreshes
`assets/model_metrics.json`. Requires `storage/feature-zone/features.parquet`
(built by `../run_pipeline.sh`) or `--features PATH`.

On the full training set the classifier reaches **AUC-ROC ≈ 0.675**, **F1 ≈ 0.41**
(`assets/final_tuned_model_metrics.csv`). The duration regressor's R² ≈ 0 is a
reported negative finding — delay magnitude depends on cause-level and weather
data, which are not yet joined.
