"""
Download NOAA GHCN-Daily weather (the supplementary source named in the
Assessment 1 proposal) for the 2018-2023 BTS window.

Each yearly file (~170 MB gzipped, every station worldwide) is streamed and
filtered on the fly, so the raw file never touches disk (this machine has very
little free space). Kept:
  * US stations only (ID prefix US / territories RQ, VQ, GQ, AQ)
  * the elements useful for delay prediction (see ELEMENTS)
  * rows that passed NOAA quality control (blank quality flag)

Outputs (dataset/noaa/):
  ghcnd-stations.txt         station metadata (id, lat, lon, elevation, name)
  ghcnd-inventory.txt        which station reports which element, and when
  noaa_daily_{year}.parquet  filtered long-format observations for that year

Usage:
  python3 ingestion/download_noaa.py              # 2018-2023
  python3 ingestion/download_noaa.py 2022 2023    # specific years
"""
import gzip
import sys
from pathlib import Path

import pandas as pd
import requests

NOAA_DIR = Path(__file__).resolve().parent.parent / "dataset" / "noaa"
BASE = "https://www.ncei.noaa.gov/pub/data/ghcn/daily"
YEARS = list(range(2018, 2024))

# GHCN-Daily elements: precipitation, snowfall, snow depth, max/min temp
# (tenths of deg C), average wind (tenths of m/s), peak gust, plus weather-type
# flags for fog, thunder, ice pellets, hail, glaze/rime and blowing snow.
ELEMENTS = {"PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND", "WSF2",
            "WT01", "WT03", "WT04", "WT05", "WT06", "WT09"}
US_PREFIXES = ("US", "RQ", "VQ", "GQ", "AQ")
CHUNK = 2_000_000
COLS = ["station", "date", "element", "value", "m_flag", "q_flag", "s_flag", "obs_time"]


def fetch_small(name: str) -> Path:
    dest = NOAA_DIR / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {name} already present")
        return dest
    print(f"[download] {name}")
    with requests.get(f"{BASE}/{name}", stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
        tmp.replace(dest)
    return dest


def download_year(year: int) -> None:
    out = NOAA_DIR / f"noaa_daily_{year}.parquet"
    if out.exists():
        print(f"[skip] {out.name} already built")
        return
    url = f"{BASE}/by_year/{year}.csv.gz"
    print(f"[stream] {url}")
    parts, n_in = [], 0
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        r.raw.decode_content = False
        with gzip.GzipFile(fileobj=r.raw) as gz:
            reader = pd.read_csv(gz, header=None, names=COLS, chunksize=CHUNK,
                                 dtype=str, keep_default_na=False)
            for chunk in reader:
                n_in += len(chunk)
                keep = (
                    chunk["station"].str.startswith(US_PREFIXES)
                    & chunk["element"].isin(ELEMENTS)
                    & (chunk["q_flag"] == "")
                )
                sub = chunk.loc[keep, ["station", "date", "element", "value"]]
                if len(sub):
                    sub = sub.assign(
                        date=pd.to_datetime(sub["date"], format="%Y%m%d"),
                        value=pd.to_numeric(sub["value"]).astype("float32"),
                    )
                    parts.append(sub)
                print(f"\r  {n_in:,} rows read", end="", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df["station"] = df["station"].astype("category")
    df["element"] = df["element"].astype("category")
    tmp = out.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(out)
    print(f"\n[ok] {out.name}: kept {len(df):,} of {n_in:,} rows "
          f"({out.stat().st_size / 1e6:.0f} MB)")


def main():
    NOAA_DIR.mkdir(parents=True, exist_ok=True)
    years = [int(y) for y in sys.argv[1:]] or YEARS
    fetch_small("ghcnd-stations.txt")
    fetch_small("ghcnd-inventory.txt")
    for y in years:
        download_year(y)
    print("[summary] NOAA download complete")


if __name__ == "__main__":
    main()
