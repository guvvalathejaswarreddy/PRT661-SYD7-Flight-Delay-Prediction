"""
PRT661-9 / PRT661-10 (revised in Assessment 4, PRT661-28): temporal + leakage-safe delay-rate + graph features.

Reads the curated flights table, engineers temporal features, flight-weighted delay rates over the seven calendar
days before each flight ([d-7, d-1]) and hub features computed from the TRAINING years only, then writes the
model-ready feature table to the feature zone. Definitions are shared with the analysis notebook
(processing/calendar_features.py).

Usage: python3 processing/build_features.py [--train-years 2018 2019 2022]
"""
import argparse
from pathlib import Path

import holidays
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calendar_features import add_graph_features, add_rolling_rates  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CURATED_PATH = ROOT / "storage" / "curated-zone" / "flights_curated.parquet"
FEATURE_DIR = ROOT / "storage" / "feature-zone"
FEATURE_PATH = FEATURE_DIR / "features.parquet"
DEFAULT_TRAIN_YEARS = (2018, 2019, 2022)          # COVID years 2020-2021 are excluded from training


def build(df: pd.DataFrame, train_years=DEFAULT_TRAIN_YEARS) -> pd.DataFrame:
    df = df.copy()
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df = df.sort_values("flight_date").reset_index(drop=True)
    if "year" not in df:
        df["year"] = df["flight_date"].dt.year

    df["crs_dep_hour"] = (df["crs_dep_time"].fillna(0) // 100).clip(0, 23).astype(int)
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["flight_date"].dt.month
    us_holidays = holidays.UnitedStates(years=df["flight_date"].dt.year.unique().tolist())
    df["is_holiday"] = df["flight_date"].dt.date.isin(us_holidays).astype(int)

    df = add_rolling_rates(df, train_years)
    df = add_graph_features(df, train_years)
    return df.dropna(subset=["distance", "origin_lat", "dest_lat"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-years", type=int, nargs="+", default=list(DEFAULT_TRAIN_YEARS))
    args = ap.parse_args()
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CURATED_PATH)
    print(f"[load] {len(df):,} curated records")
    df = build(df, tuple(args.train_years))
    df.to_parquet(FEATURE_PATH, index=False)
    print(f"[write] {len(df):,} feature rows -> {FEATURE_PATH}")
    print(f"[summary] overall delay rate: {df['arr_del15'].mean() * 100:.1f}%; hub features from years {args.train_years}")


if __name__ == "__main__":
    main()
