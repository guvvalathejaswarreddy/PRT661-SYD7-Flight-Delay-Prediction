"""
PRT661-14: Predict delay time model.

Regresses arr_delay_minutes for flights that were actually delayed, using
LightGBM. Reports MAE / RMSE / R^2 on a time-based holdout.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "storage" / "feature-zone" / "features.parquet"
METRICS_PATH = ROOT / "models" / "regressor_metrics.json"

NUMERIC_FEATURES = [
    "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
    "distance", "carrier_delay_rate_7d", "origin_delay_rate_7d",
]
CATEGORICAL_FEATURES = ["carrier"]
TARGET = "arr_delay_minutes"


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("flight_date")
    cutoff = df["flight_date"].quantile(1 - test_frac)
    return df[df["flight_date"] < cutoff], df[df["flight_date"] >= cutoff]


def main():
    df = pd.read_parquet(FEATURE_PATH)
    delayed = df[df["arr_del15"] == 1].copy()
    print(f"[filter] {len(delayed):,} delayed flights used for duration regression")

    train, test = time_based_split(delayed)
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[cols], train[TARGET]
    X_test, y_test = test[cols], test[TARGET]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    pipe = Pipeline([
        ("pre", pre),
        ("model", LGBMRegressor(
            n_estimators=300, max_depth=-1, learning_rate=0.08,
            n_jobs=-1, random_state=42, verbosity=-1,
        )),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    metrics = {
        "mae": round(mean_absolute_error(y_test, pred), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "r2": round(r2_score(y_test, pred), 4),
    }
    print(f"[LightGBM regressor] {metrics}")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"[write] metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
