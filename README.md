# PRT661 Group SYD7: Flight Delay Prediction

Batch predictive-analytics system: estimates the probability that a US domestic flight arrives 15+ minutes late,
and, for flights that are delayed, an 80% range for how many minutes late.

## The notebook is the source of truth

`notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb` is the canonical analysis. It runs on the full 2018–2023 extract
(~37.9M flights) and produces every number in the final report. All results, the model, the calibrator, the thresholds
and the figures come from it.

| Notebook part | Content |
|---|---|
| Cells 1–75 (Assessment 2) | EDA, walk-forward CV, COVID exclusion, graph features, Optuna tuning, SHAP, fairness |
| Sections 15–24 (Assessment 4) | Feature audit, training-year hub features, NOAA prior-day weather, leakage tests, ablation, calibration and thresholds, duration range, fairness re-framing, explainability, export |

It checkpoints each step (A4 checkpoints are prefixed `a4_`), so "Run all" resumes after an interruption. Run it in
Colab or Jupyter. It exports the deployable files to `dashboard/assets/a4/`.

**Final model:** LightGBM, 26 inputs (11 schedule/carrier/network inputs + 15 prior-day NOAA weather inputs), Platt
calibrated. Trained on 2018 + 2019 + 2022, tested once on 2023 (6,740,139 flights).

| Result (2023 test) | Value |
|---|---|
| AUC-ROC, Assessment 2 model → final ablation variant V2 | 0.6755 → 0.6841 (deployed 85%-fit model: 0.6824) |
| Calibration error, raw → calibrated | 0.270 → 0.006 |
| Same-day weather upper bound (not deployable) | 0.7088 |
| Duration 80% range | covers 78.8%, 112 min wide, no skill over a constant |

Source files: `dashboard/assets/a4/a4_ablation_results.csv`, `a4_calibration_summary.csv`, `a4_summary.json`.

## Repository layout

| Folder | Role |
|---|---|
| `notebooks/` | **Canonical analysis** (above) |
| `dashboard/` | Streamlit app. Reads only the exported files in `dashboard/assets/a4/` and `dashboard/data/` |
| `processing/calendar_features.py` | Leakage-safe rolling delay rates and training-year hub features. Same definitions as notebook §16–17; imported by the tests |
| `tests/` | 11 pytest tests (rolling window, no look-ahead, training-only hub features, NOAA quality flag and prior day, calibration, thresholds, data schema) |
| `monitoring/monitor.py` | Monthly AUC, calibration gap, alert share, PSI and retraining rules on the 2023 sample |
| `ingestion/` | BTS, OpenFlights and NOAA GHCN-Daily downloads; OpenSky live module |
| `processing/clean_and_join.py`, `build_features.py` | DuckDB clean + join and feature build; `run_pipeline.sh` runs ingestion → features → tests → monitoring |
| `experiments/` | One-month inbound-aircraft pilot (not used in the model; see report limitations) |
| `docs/` | Jira ticket register and dashboard screenshots |
| `Dockerfile`, `docker-compose.yml` | Dashboard container and local MinIO (container image not built in development) |

### Scope of the scripts

The scripts cover data preparation, tests, monitoring and the dashboard. Model training, the ablation, calibration,
thresholds and the duration quantiles live only in the notebook; it exports the files the dashboard reads.
The Assessment 2 local baseline scripts were retired because the notebook supersedes them (they scored AUC ≈ 0.675).

## Quick start

```bash
python3 -m pip install -r dashboard/requirements.txt     # pinned versions used for the results
./run_dashboard.sh                                        # streamlit run dashboard/streamlit_app.py
python3 -m pytest tests -q                                # 11 tests
python3 monitoring/monitor.py                             # monitoring report and figure
```

The dashboard starts straight after install: the model bundles, thresholds, reference tables and a 300,000-flight 2023
sample are committed under `dashboard/`. To regenerate them, run the notebook and copy its `a4_outputs` folder into
`dashboard/assets/a4/`.

## Dashboard

Seven pages: Overview · Predict a Flight · Model Performance · Explainability · Fairness · Data & EDA · About.

- Predict a Flight returns a calibrated probability, the alert level with its 2023 precision, recall and workload, an
  80% duration range with a reliability warning, and plain-language reasons.
- It looks up the previous day's weather at the nearest NOAA station (`dashboard/noaa_live.py`). If NOAA cannot be
  reached, the weather is treated as unknown or a manual scenario can be entered.
- An OpenSky panel shows live aircraft near the origin for context only (not a model input).

See `dashboard/README.md`.

## Data

- **BTS Airline On-Time Performance**, 2018–2023: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
- **NOAA GHCN-Daily** (`ingestion/download_noaa.py`): US stations, 13 elements, quality-flag-passed rows, 127.3M observations.
  Only the previous day (d−1) is used as a deployable input; same-day weather appears only in the labelled upper-bound experiment.
- **OpenFlights** airports: https://openflights.org/data.php (`dataset/openflights/airports.dat`, ODbL).

Raw BTS/NOAA data, `storage/` and `PRT661_outputs/` are git-ignored and rebuilt by the ingestion scripts or the notebook.

## Known limitations

- Modest discrimination (AUC ≈ 0.68): prior-day weather adds +0.008; same-day weather would add about +0.025 but is not available before departure.
- One test year (2023) and one split; the bootstrap interval covers sampling noise only.
- Delay duration has no skill over a constant guess (R² ≈ −0.07), so only a wide range is shown.
- Hyper-parameters are those tuned in Assessment 2; re-tuning on the 26 inputs is future work.
- Monitoring is a prototype: no scheduler, no automatic retraining, no model registry, and R1 fires seasonally without a same-season reference.
- The dashboard has no authentication (local or trusted network only). Airflow and S3/MinIO wiring are not built; the local `storage/` folders stand in.
