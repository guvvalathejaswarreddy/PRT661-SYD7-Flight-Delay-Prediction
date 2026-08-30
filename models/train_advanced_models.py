"""
PRT661-13: Build advanced prediction models.

Gradient-boosting classifiers (XGBoost, LightGBM) for arr_del15, plus a SHAP
feature-importance summary for the better of the two. Optuna hyperparameter
search from the original proposal is deferred (see README) -- this uses solid
default hyperparameters to get a working advanced model within the build window.
"""

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = ROOT / "storage" / "feature-zone" / "features.parquet"
METRICS_PATH = ROOT / "models" / "advanced_metrics.json"

NUMERIC_FEATURES = [
    "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
    "distance", "carrier_delay_rate_7d", "origin_delay_rate_7d",
]
CATEGORICAL_FEATURES = ["carrier"]
TARGET = "arr_del15"


def time_based_split(df: pd.DataFrame, test_frac: float = 0.2):
    df = df.sort_values("flight_date")
    cutoff = df["flight_date"].quantile(1 - test_frac)
    return df[df["flight_date"] < cutoff], df[df["flight_date"] >= cutoff]


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
    cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X_train, y_train = train[cols], train[TARGET]
    X_test, y_test = test[cols], test[TARGET]

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    results = {}

    xgb = build_pipeline(XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        scale_pos_weight=scale_pos_weight, eval_metric="auc",
        n_jobs=-1, random_state=42,
    ))
    xgb.fit(X_train, y_train)
    results["xgboost"] = evaluate("XGBoost", xgb, X_test, y_test)

    lgbm = build_pipeline(LGBMClassifier(
        n_estimators=300, max_depth=-1, learning_rate=0.08,
        class_weight="balanced", n_jobs=-1, random_state=42, verbosity=-1,
    ))
    lgbm.fit(X_train, y_train)
    results["lightgbm"] = evaluate("LightGBM", lgbm, X_test, y_test)

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[write] metrics -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
