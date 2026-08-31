"""
PRT661-9 / PRT661-10: Analyse delay patterns + create prediction features.

Reads the curated flights table, engineers temporal and rolling delay-rate
features, and writes the model-ready feature table to the feature-zone.
"""

from pathlib import Path

import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CURATED_PATH = ROOT / "storage" / "curated-zone" / "flights_curated.parquet"
FEATURE_DIR = ROOT / "storage" / "feature-zone"
FEATURE_PATH = FEATURE_DIR / "features.parquet"


def main():
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CURATED_PATH)
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df = df.sort_values("flight_date").reset_index(drop=True)
    print(f"[load] {len(df):,} curated records")

    # Temporal features
    df["crs_dep_hour"] = (df["crs_dep_time"].fillna(0) // 100).clip(0, 23).astype(int)
    df["day_of_week"] = df["flight_date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["month"] = df["flight_date"].dt.month

    us_holidays = holidays.UnitedStates(years=df["flight_date"].dt.year.unique().tolist())
    df["is_holiday"] = df["flight_date"].dt.date.isin(us_holidays).astype(int)

    # Rolling 7-day delay rate per carrier and per origin airport, computed
    # only from *prior* days so we don't leak the current day's outcome.
    def rolling_rate(frame: pd.DataFrame, key: str, out_col: str) -> pd.Series:
        daily = (
            frame.groupby([key, frame["flight_date"].dt.date])["arr_del15"]
            .mean()
            .rename("daily_rate")
            .reset_index()
            .rename(columns={"flight_date": "day"})
        )
        daily["day"] = pd.to_datetime(daily["day"])
        daily = daily.sort_values([key, "day"])
        daily[out_col] = (
            daily.groupby(key)["daily_rate"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        )
        daily[out_col] = daily[out_col].fillna(daily["daily_rate"].mean())
        return daily[[key, "day", out_col]]

    carrier_rates = rolling_rate(df, "carrier", "carrier_delay_rate_7d")
    origin_rates = rolling_rate(df, "origin", "origin_delay_rate_7d")

    df["day"] = df["flight_date"].dt.normalize()
    df = df.merge(carrier_rates, on=["carrier", "day"], how="left")
    df = df.merge(origin_rates, on=["origin", "day"], how="left")
    df = df.drop(columns=["day"])

    df["carrier_delay_rate_7d"] = df["carrier_delay_rate_7d"].fillna(df["arr_del15"].mean())
    df["origin_delay_rate_7d"] = df["origin_delay_rate_7d"].fillna(df["arr_del15"].mean())

    df = df.dropna(subset=["distance", "origin_lat", "dest_lat"])

    df.to_parquet(FEATURE_PATH, index=False)
    print(f"[write] {len(df):,} feature rows -> {FEATURE_PATH}")
    print(f"[summary] overall delay rate: {df['arr_del15'].mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
