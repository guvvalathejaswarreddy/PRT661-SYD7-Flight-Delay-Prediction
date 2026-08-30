#!/usr/bin/env bash
# Runs the full local pipeline end to end: ingestion -> processing ->
# feature engineering -> models -> predictions -> (then launch the dashboard
# separately with run_dashboard.sh). Stands in for the Airflow DAG (PRT661-5)
# until that's wired up.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/6 Download BTS flight data =="
python3 ingestion/download_bts.py

echo "== 2/6 Download OpenFlights airport data =="
python3 ingestion/download_openflights.py

echo "== 3/6 Clean + join (DuckDB) =="
python3 processing/clean_and_join.py

echo "== 4/6 Build features =="
python3 processing/build_features.py

echo "== 5/6 Train models =="
python3 models/train_baseline_models.py
python3 models/train_advanced_models.py
python3 models/train_duration_regressor.py

echo "== 6/6 Generate predictions for dashboard (tuned model) =="
python3 dashboard/generate_predictions.py

echo "Done. Run ./run_dashboard.sh to view the results."
