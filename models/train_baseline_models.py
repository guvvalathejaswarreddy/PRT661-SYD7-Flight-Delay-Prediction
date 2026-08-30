"""
PRT661-11 / PRT661-12: Plan model testing method + build basic prediction models.

Time-based holdout (train on earliest 80% of days, test on the most recent 20%
-- a simple stand-in for the walk-forward CV described in the proposal; see
README for how to extend this to full walk-forward folds) and two baseline
classifiers: Logistic Regression and Random Forest, predicting arr_del15
(1 = arrival delayed 15+ minutes).
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "storage" / "feature-zone" / "features.parquet"
METRICS_PATH = ROOT / "models" / "baseline_metrics.json"

NUMERIC_FEATURES = [
    "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
    "distance", "carrier_delay_rate_7d", "origin_delay_rate_7d",
]
CATEGORICAL_FEATURES = ["carrier"]
TARGET = "arr_del15"


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("flight_date")
    cutoff = df["flight_date"].quantile(1 - test_frac)
    train = df[df["flight_date"] < cutoff]
    test = df[df["flight_date"] >= cutoff]
    return train, test


def build_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ], remainder="passthrough")
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(name, pipe, X_test, y_test):
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": round(roc_auc_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred), 4),
    }
    print(f"[{name}] {metrics}")
    return metrics


def main():
    df = pd.read_parquet(FEATURE_PATH)
    train, test = time_based_split(df)
    print(f"[split] train={len(train):,} test={len(test):,} (time-based holdout)")

    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[cols], train[TARGET]
    X_test, y_test = test[cols], test[TARGET]

    results = {}

    logreg = build_pipeline(LogisticRegression(max_iter=1000, class_weight="balanced"))
    logreg.fit(X_train, y_train)
    results["logistic_regression"] = evaluate("Logistic Regression", logreg, X_test, y_test)

    rf = build_pipeline(RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1, random_state=42,
    ))
    rf.fit(X_train, y_train)
    results["random_forest"] = evaluate("Random Forest", rf, X_test, y_test)

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[write] metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
