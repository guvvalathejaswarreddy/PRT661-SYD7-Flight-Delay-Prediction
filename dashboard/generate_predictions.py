"""
Generate the prediction table that powers the Streamlit dashboard.

This reproduces the *production fold* of the main analysis notebook
(`notebooks/PRT661_Flight_Delay_Full_Analysis.ipynb`):

  * train years : 2018 + 2019 + 2022   (COVID years 2020-2021 excluded)
  * test  year  : 2023
  * classifier  : tuned LightGBM  (hyper-params from
                  dashboard/assets/best_hyperparameters.json)
  * regressor   : tuned LightGBM, delayed flights only

The Colab-trained pipelines do not unpickle cleanly across scikit-learn
versions, so this script *re-fits* the tuned models locally from the saved
hyper-parameters. On the canonical full training set the classifier reaches
AUC-ROC ~= 0.675 (see dashboard/assets/final_tuned_model_metrics.csv); by
default this script trains on a stratified 5M-row sample so it runs in a couple
of minutes on a laptop -- pass --full to use every training row.

Outputs
-------
dashboard/data/predictions.parquet        stratified sample (committed, ~300k rows)
dashboard/data/predictions_full.parquet   every 2023 flight (git-ignored)   [--write-full]
dashboard/assets/model_metrics.json       metrics computed on this run

Usage
-----
    python dashboard/generate_predictions.py
    python dashboard/generate_predictions.py --full --write-full
    python dashboard/generate_predictions.py --features /path/to/features.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FEATURES = ROOT / "storage" / "feature-zone" / "features.parquet"
DATA_DIR = ROOT / "dashboard" / "data"
ASSETS_DIR = ROOT / "dashboard" / "assets"
# best_hyperparameters.json ships in the repo under dashboard/assets/; the
# PRT661_outputs/ copy (git-ignored) is used only when this runs next to a
# fresh notebook export.
HYPERPARAMS_PATH = next(
    (p for p in (ASSETS_DIR / "best_hyperparameters.json",
                 ROOT / "PRT661_outputs" / "models" / "best_hyperparameters.json")
     if p.exists()),
    ASSETS_DIR / "best_hyperparameters.json",
)

RANDOM_STATE = 42
TRAIN_YEARS = [2018, 2019, 2022]
TEST_YEAR = 2023

NUMERIC_FEATURES = [
    "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
    "distance", "carrier_delay_rate_7d", "origin_delay_rate_7d",
    "origin_degree_centrality",
]
CATEGORICAL_FEATURES = ["carrier", "origin_hub_tier"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "arr_del15"
DURATION_TARGET = "arr_delay_minutes"

FALLBACK_CLF_PARAMS = dict(
    n_estimators=359, num_leaves=119, learning_rate=0.013,
    feature_fraction=0.68, bagging_fraction=0.62, min_child_samples=39,
)
FALLBACK_REG_PARAMS = dict(
    n_estimators=316, num_leaves=102, learning_rate=0.069,
    feature_fraction=0.66, bagging_fraction=0.75,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_hyperparams() -> tuple[dict, dict]:
    if HYPERPARAMS_PATH.exists():
        hp = json.loads(HYPERPARAMS_PATH.read_text())
        clf = hp.get("lightgbm_classifier", FALLBACK_CLF_PARAMS)
        reg = hp.get("lightgbm_regressor", FALLBACK_REG_PARAMS)
        log(f"Loaded tuned hyper-parameters from {HYPERPARAMS_PATH.name}")
        return clf, reg
    log("best_hyperparameters.json not found -- using built-in fallback params")
    return dict(FALLBACK_CLF_PARAMS), dict(FALLBACK_REG_PARAMS)


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"\nFeature table not found: {path}\n"
            "Build it first with the pipeline (see software/README.md):\n"
            "    ./run_pipeline.sh            # full local pipeline\n"
            "or point this script at an existing file with --features PATH\n"
        )
    log(f"Reading {path} ...")
    cols = [
        "flight_date", "carrier", "origin", "destination", "distance",
        "crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday",
        "carrier_delay_rate_7d", "origin_delay_rate_7d",
        "arr_del15", "arr_delay_minutes",
    ]
    df = pd.read_parquet(path, columns=cols)
    # Low-memory dtypes: the object carrier/origin columns dominate RAM otherwise.
    for c in ("carrier", "origin", "destination"):
        df[c] = df[c].astype("category")
    for c in ("crs_dep_hour", "day_of_week", "is_weekend", "month", "is_holiday", "arr_del15"):
        df[c] = df[c].astype("int16")
    for c in ("distance", "carrier_delay_rate_7d", "origin_delay_rate_7d", "arr_delay_minutes"):
        df[c] = df[c].astype("float32")
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["year"] = df["flight_date"].dt.year.astype("int16")
    log(f"  {len(df):,} rows x {df.shape[1]} cols  "
        f"({df.memory_usage(deep=True).sum() / 1e9:.2f} GB)")
    return df


def add_graph_features(df: pd.DataFrame) -> pd.DataFrame:
    """Out-degree centrality + volume-based hub tier, computed over the whole
    dataset -- identical to section 4 of the analysis notebook."""
    log("Deriving graph-hub features (origin_degree_centrality, origin_hub_tier) ...")
    degree = df.groupby("origin", observed=True)["destination"].nunique()
    degree_norm = (degree - degree.min()) / (degree.max() - degree.min())
    df["origin_degree_centrality"] = df["origin"].map(degree_norm).astype("float32")

    volume = df["origin"].value_counts()
    top_decile = volume.quantile(0.90)
    top_4deciles = volume.quantile(0.60)

    def tier_for(v: float) -> str:
        if v >= top_decile:
            return "major"
        if v >= top_4deciles:
            return "medium"
        return "small"

    hub_tier_map = volume.apply(tier_for)
    df["origin_hub_tier"] = df["origin"].map(hub_tier_map).astype("category")
    log(f"  hub tiers: {df['origin_hub_tier'].value_counts().to_dict()}")
    return df


def make_pipeline(model) -> Pipeline:
    # Sparse one-hot (2 ones per row) keeps memory tiny; LightGBM consumes the
    # resulting CSR matrix directly. Trees are scale-invariant so the numeric
    # block is passed through unscaled.
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", dtype=np.float32),
          CATEGORICAL_FEATURES)],
        remainder="passthrough",
        sparse_threshold=1.0,
    )
    return Pipeline([("pre", pre), ("model", model)])


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df
    frac = n / len(df)
    out = df.groupby([TARGET], observed=True, group_keys=False).sample(
        frac=frac, random_state=seed
    )
    return out


def predict_in_chunks(pipe: Pipeline, X: pd.DataFrame, kind: str,
                      chunk: int = 1_000_000) -> np.ndarray:
    parts = []
    for start in range(0, len(X), chunk):
        block = X.iloc[start:start + chunk]
        if kind == "proba":
            parts.append(pipe.predict_proba(block)[:, 1].astype("float32"))
        else:
            parts.append(pipe.predict(block).astype("float32"))
    return np.concatenate(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES,
                    help=f"feature parquet (default: {DEFAULT_FEATURES})")
    ap.add_argument("--full", action="store_true",
                    help="train on every training-year row (needs more RAM/time)")
    ap.add_argument("--train-sample-n", type=int, default=5_000_000,
                    help="rows to train on when not --full (default 5,000,000)")
    ap.add_argument("--dashboard-sample-n", type=int, default=300_000,
                    help="rows written to the committed predictions.parquet")
    ap.add_argument("--write-full", action="store_true",
                    help="also write predictions_full.parquet (every 2023 flight)")
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    clf_params, reg_params = load_hyperparams()
    df = load_features(args.features)
    df = add_graph_features(df)

    train = df[df["year"].isin(TRAIN_YEARS)]
    test = df[df["year"] == TEST_YEAR].copy()
    log(f"Split: train={len(train):,} rows ({TRAIN_YEARS})  test={len(test):,} rows ({TEST_YEAR})")
    if test.empty:
        raise SystemExit(f"No {TEST_YEAR} rows in the feature table -- nothing to score.")

    if args.full:
        train_fit = train
        log("Training on the FULL training set")
    else:
        train_fit = stratified_sample(train, args.train_sample_n, RANDOM_STATE)
        log(f"Training on a stratified {len(train_fit):,}-row sample "
            f"(pass --full for all {len(train):,})")

    # ---- Classifier -------------------------------------------------------
    scale_pos_weight = (train_fit[TARGET] == 0).sum() / max((train_fit[TARGET] == 1).sum(), 1)
    clf = make_pipeline(LGBMClassifier(
        **clf_params, class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
    ))
    log("Fitting LightGBM classifier ...")
    ts = time.time()
    clf.fit(train_fit[ALL_FEATURES], train_fit[TARGET])
    log(f"  done in {time.time() - ts:.0f}s")

    log("Scoring 2023 (delay probability) ...")
    test["predicted_delay_prob"] = predict_in_chunks(clf, test[ALL_FEATURES], "proba")
    test["predicted_delayed"] = test["predicted_delay_prob"] >= 0.5

    # ---- Duration regressor (delayed flights only) ----------------------
    reg_train = train_fit[train_fit[TARGET] == 1]
    reg = make_pipeline(LGBMRegressor(
        **reg_params, n_jobs=-1, random_state=RANDOM_STATE, verbosity=-1,
    ))
    log(f"Fitting LightGBM duration regressor on {len(reg_train):,} delayed flights ...")
    ts = time.time()
    reg.fit(reg_train[ALL_FEATURES], reg_train[DURATION_TARGET])
    log(f"  done in {time.time() - ts:.0f}s")

    log("Scoring 2023 (delay minutes) ...")
    test["predicted_delay_minutes"] = predict_in_chunks(
        reg, test[ALL_FEATURES], "value").clip(min=0.0)
    # A flight we don't expect to be delayed shouldn't carry a big ETA;
    # damp its predicted minutes by its delay probability.
    not_delayed = ~test["predicted_delayed"]
    test.loc[not_delayed, "predicted_delay_minutes"] *= test.loc[not_delayed, "predicted_delay_prob"]

    # ---- Persist the fitted models + a reference file for the Predict page --
    joblib.dump(clf, ASSETS_DIR / "model_classifier.joblib", compress=3)
    joblib.dump(reg, ASSETS_DIR / "model_regressor.joblib", compress=3)
    log(f"Wrote {ASSETS_DIR / 'model_classifier.joblib'} + model_regressor.joblib")

    # Per-origin / per-carrier lookups so the Predict page can turn a handful of
    # human inputs into the full feature row the model expects.
    origin_grp = df.groupby("origin", observed=True)
    origins = {}
    for o, g in origin_grp:
        origins[str(o)] = {
            "hub_tier": str(g["origin_hub_tier"].iloc[0]),
            "degree_centrality": float(g["origin_degree_centrality"].iloc[0]),
            "delay_rate_7d": float(g["origin_delay_rate_7d"].mean()),
            "flights": int(len(g)),
            "destinations": sorted(map(str, g["destination"].unique())),
        }
    carriers_ref = {
        str(c): {"delay_rate_7d": float(g["carrier_delay_rate_7d"].mean()),
                 "flights": int(len(g))}
        for c, g in df.groupby("carrier", observed=True)
    }
    # Typical route distance (median) keyed "ORIGIN-DEST".
    route_dist = (
        df.groupby(["origin", "destination"], observed=True)["distance"].median().round().astype("int32")
    )
    routes_ref = {f"{o}-{d}": int(v) for (o, d), v in route_dist.items()}

    reference = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "features": {"numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES,
                     "order": ALL_FEATURES},
        "defaults": {
            "carrier_delay_rate_7d": float(df["carrier_delay_rate_7d"].mean()),
            "origin_delay_rate_7d": float(df["origin_delay_rate_7d"].mean()),
            "distance": float(df["distance"].median()),
            "overall_delay_rate": float(df[TARGET].mean()),
        },
        "carriers": carriers_ref,
        "origins": origins,
        "routes": routes_ref,
    }
    (ASSETS_DIR / "predict_reference.json").write_text(json.dumps(reference))
    log(f"Wrote {ASSETS_DIR / 'predict_reference.json'}  "
        f"({len(origins)} airports, {len(carriers_ref)} carriers, {len(routes_ref)} routes)")

    # ---- Metrics on the full 2023 test set -----------------------------
    y_true = test[TARGET].to_numpy()
    proba = test["predicted_delay_prob"].to_numpy()
    pred = (proba >= 0.5).astype(int)
    delayed_mask = y_true == 1
    metrics = {
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "training": "full" if args.full else f"stratified {len(train_fit):,}-row sample",
        "train_years": TRAIN_YEARS,
        "test_year": TEST_YEAR,
        "test_rows": int(len(test)),
        "classifier": {
            "model": "LightGBM (tuned)",
            "auc_roc": float(roc_auc_score(y_true, proba)),
            "average_precision": float(average_precision_score(y_true, proba)),
            "f1": float(f1_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred)),
            "brier": float(brier_score_loss(y_true, proba)),
            "actual_delay_rate": float(y_true.mean()),
            "predicted_delay_rate": float(pred.mean()),
        },
        "regressor": {
            "model": "LightGBM (tuned), delayed flights only",
            "mae": float(mean_absolute_error(test.loc[delayed_mask, DURATION_TARGET],
                                             test.loc[delayed_mask, "predicted_delay_minutes"])),
            "rmse": float(np.sqrt(mean_squared_error(
                test.loc[delayed_mask, DURATION_TARGET],
                test.loc[delayed_mask, "predicted_delay_minutes"]))),
            "r2": float(r2_score(test.loc[delayed_mask, DURATION_TARGET],
                                 test.loc[delayed_mask, "predicted_delay_minutes"])),
            "n_delayed": int(delayed_mask.sum()),
        },
    }
    (ASSETS_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"Wrote {ASSETS_DIR / 'model_metrics.json'}")
    log(f"  classifier AUC-ROC = {metrics['classifier']['auc_roc']:.4f}  "
        f"F1 = {metrics['classifier']['f1']:.4f}")
    log(f"  regressor  MAE = {metrics['regressor']['mae']:.1f} min  "
        f"R2 = {metrics['regressor']['r2']:.3f}")

    # ---- Write prediction tables --------------------------------------
    keep = [
        "flight_date", "carrier", "origin", "destination", "distance",
        "crs_dep_hour", "origin_hub_tier", "year",
        "arr_del15", "arr_delay_minutes",
        "predicted_delay_prob", "predicted_delayed", "predicted_delay_minutes",
    ]
    out = test[keep].rename(columns={
        "arr_del15": "actual_delayed",
        "arr_delay_minutes": "actual_delay_minutes",
    })
    out["carrier"] = out["carrier"].astype(str)
    out["origin"] = out["origin"].astype(str)
    out["destination"] = out["destination"].astype(str)
    out["origin_hub_tier"] = out["origin_hub_tier"].astype(str)
    out["actual_delayed"] = out["actual_delayed"].astype(bool)

    if args.write_full:
        full_path = DATA_DIR / "predictions_full.parquet"
        out.to_parquet(full_path, index=False)
        log(f"Wrote {full_path}  ({len(out):,} rows)")

    sample = stratified_sample(
        out.assign(**{TARGET: out["actual_delayed"].astype(int)}),
        args.dashboard_sample_n, RANDOM_STATE,
    ).drop(columns=[TARGET])
    sample = sample.sort_values("flight_date").reset_index(drop=True)
    sample_path = DATA_DIR / "predictions.parquet"
    sample.to_parquet(sample_path, index=False)
    log(f"Wrote {sample_path}  ({len(sample):,} rows, "
        f"{sample_path.stat().st_size / 1e6:.1f} MB)")

    log(f"All done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
