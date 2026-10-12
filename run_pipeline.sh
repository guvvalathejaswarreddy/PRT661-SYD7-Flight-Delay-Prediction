#!/usr/bin/env bash
# Data pipeline: ingestion -> clean + join -> leakage-safe features, then tests and monitoring.
# Model training, calibration, thresholds and evaluation are done in
# notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb (the canonical analysis); it exports the dashboard files.
# Stands in for the Airflow DAG (PRT661-5), which is not built.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 Download BTS flight data =="
python3 ingestion/download_bts.py

echo "== 2/6 Download OpenFlights airport data =="
python3 ingestion/download_openflights.py

echo "== 3/6 Download NOAA GHCN-Daily weather (US stations, quality-flag-passed rows) =="
python3 ingestion/download_noaa.py

echo "== 4/6 Clean + join (DuckDB) =="
python3 processing/clean_and_join.py

echo "== 5/6 Build features (leakage-safe rolling rates + training-year hub features) =="
python3 processing/build_features.py

echo "== 6/6 Checks: automated tests and monitoring report =="
python3 -m pytest tests -q
python3 monitoring/monitor.py

echo "Done. Open the notebook for modelling, then ./run_dashboard.sh to view the results."
