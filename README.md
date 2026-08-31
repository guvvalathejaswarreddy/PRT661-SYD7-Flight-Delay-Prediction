# PRT661 Group SYD7 — Flight Delay Prediction

Batch predictive-analytics pipeline: classifies whether a US domestic flight
will arrive 15+ minutes late, and if so, predicts how many minutes late.

## Repository layout

| Stage | Folder | What it does |
|---|---|---|
| Ingestion | `ingestion/` | Downloads BTS on-time performance + OpenFlights airport metadata |
| Storage | `storage/` | Local raw / curated / feature zones (stand-in for MinIO — see `docker-compose.yml`) |
| Processing | `processing/` | DuckDB clean + join, then feature engineering |
| Modelling — local | `models/` | Quick baseline scripts: single time-based holdout, LogReg / RF / XGBoost / LightGBM + duration regressor |
| Modelling — full | `notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb` | **The canonical analysis** (see below) |
| Dashboard | `dashboard/` | Multi-page Streamlit app driven by the trained model |

### Analysis notebook

`notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb` runs the full 2018–2023
dataset (~37.9M flights) and implements everything the proposal specifies:

- **walk-forward (expanding-window) cross-validation**, folds on complete years;
- **COVID period (2020–2021) excluded** from training + a separate distribution-shift check;
- **graph-derived hub features** — out-degree centrality + volume-based hub tier;
- **Optuna** hyper-parameter tuning (XGBoost, LightGBM, duration regressor);
- **SHAP** interpretability and a **fairness** breakdown by carrier / hub tier / geography;
- checkpoint/resume so "Run all" continues after any interruption.

Running it (re)creates `PRT661_outputs/` locally (figures, per-model metrics,
trained `.pkl`s, tuned hyper-parameters). That folder is **git-ignored** — the
committed copies that the dashboard and `generate_predictions.py` use live in
`dashboard/assets/` (`figures/`, `*.csv`, `model_metrics.json`,
`best_hyperparameters.json`, `run_config.json`).
Production fold: **train 2018 + 2019 + 2022 → test 2023**; best model
**tuned LightGBM, AUC-ROC ≈ 0.675, F1 ≈ 0.41**
(`dashboard/assets/final_tuned_model_metrics.csv`).

The `models/*.py` scripts are a lighter, faster local baseline (single 80/20
time holdout, fewer features) kept for `run_pipeline.sh`; their numbers are not
the ones reported.

## Quick start

```bash
python3 -m pip install -r requirements.txt

# Option A — full analysis: open the notebook in Colab / Jupyter and Run All.
# Option B — local baseline pipeline:
./run_pipeline.sh        # ingest -> clean -> features -> train -> predict

# Dashboard (needs storage/feature-zone/features.parquet from the pipeline
# or the notebook's feature build):
python dashboard/generate_predictions.py    # build predictions from the tuned model
./run_dashboard.sh                          # streamlit run dashboard/streamlit_app.py
```

`dashboard/data/predictions.parquet` (a 300k-row 2023 sample) is committed, so
the dashboard runs straight after `pip install` without rebuilding anything.

## Dataset

Primary: **BTS Airline On-Time Performance Dataset** (2018–2023), the source
named in the Assessment 1 proposal.
Official page: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
Direct monthly file (used by `ingestion/download_bts.py`):
```
https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip
```
Supplementary: **OpenFlights** airport metadata (`ingestion/download_openflights.py`,
https://openflights.org/data.php).

`dataset/bts_parquet/` (72 monthly files, Jan 2018 – Dec 2023) and
`storage/` are **git-ignored** — rebuild them with `ingestion/download_bts.py`
+ `ingestion/import_from_downloads.py` + `processing/`, or `./run_pipeline.sh`.
`dataset/openflights/airports.dat` is committed (small, public ODbL data).

## Known gaps / deferred work

- **NOAA weather is not joined yet.** The proposal's Five Vs and feature set
  reference weather; this build joins BTS + OpenFlights only. This caps the
  classifier near AUC ≈ 0.68 (weather-enhanced models in the literature reach
  0.80–0.92) and is the top priority for Assessment 3: add
  `ingestion/download_noaa.py` + a nearest-station join in
  `processing/clean_and_join.py`.
- **Delay-*duration* is essentially unpredictable** from the current features
  (R² ≈ 0) — magnitude depends on the specific delay cause. Reported as a
  negative finding.
- **Airflow DAG and real MinIO/S3 wiring are stubbed** — `docker-compose.yml`
  starts MinIO locally, but the scripts read/write local folders under
  `storage/`, not the S3 API (`run_pipeline.sh` stands in for the DAG).
- The `models/*.py` local pipeline still uses a single time-based holdout; the
  notebook is the walk-forward version.

## Dashboard

See `dashboard/README.md`. Multi-page Streamlit app: Overview · **Predict a
Flight** · Model Performance · Explainability · Fairness · Data & EDA · About.
The Predict page loads the fitted models (`dashboard/assets/model_*.joblib`,
written by `generate_predictions.py`) and returns a live P(delay) + estimated
minutes for user-entered flight details. Everything is driven by
`generate_predictions.py` output and the committed `dashboard/assets/`
artifacts. Replaces the Plotly Dash app named in the Assessment 1 proposal.
