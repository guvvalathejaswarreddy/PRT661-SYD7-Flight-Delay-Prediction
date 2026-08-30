"""
Processes BTS monthly zip files that were manually downloaded into ~/Downloads
(the official transtats.bts.gov links, downloaded by hand due to slow sandbox
connectivity). For each zip: extract the CSV, convert it straight to a
compressed Parquet file in dataset/bts_parquet/, then delete both the zip and
the intermediate CSV.

Why Parquet instead of keeping the raw CSV: the local disk here only has
~16GB free. All 72 months as raw CSV would need ~17GB (measured from Jan 2023:
a 27MB zip extracts to a 243MB CSV, roughly a 9x expansion). Converting each
month to Parquet immediately drops it to Assessment 1's own estimate of
~2.5GB for the full 2018-2023 range, and processing/clean_and_join.py can read
Parquet just as easily as CSV via DuckDB.

Usage:
  python3 ingestion/import_from_downloads.py
"""

import re
import shutil
import zipfile
from pathlib import Path

import duckdb

DOWNLOADS_DIR = Path.home() / "Downloads"
PARQUET_DIR = Path(__file__).resolve().parent.parent / "dataset" / "bts_parquet"
ZIP_PATTERN = re.compile(
    r"^On_Time_Reporting_Carrier_On_Time_Performance_1987_present_(\d{4})_(\d{1,2})\.zip$"
)


def process_zip(zip_path: Path, con: duckdb.DuckDBPyConnection):
    m = ZIP_PATTERN.match(zip_path.name)
    if not m:
        return None
    year, month = int(m.group(1)), int(m.group(2))
    out_path = PARQUET_DIR / f"bts_{year}_{month:02d}.parquet"

    if out_path.exists():
        print(f"[skip] bts_{year}_{month:02d}.parquet already exists -- deleting leftover zip")
        zip_path.unlink()
        return (year, month)

    if not zipfile.is_zipfile(zip_path):
        print(f"[bad-zip] {zip_path.name} is not a valid zip, leaving it for you to re-download")
        return None

    tmp_csv_dir = PARQUET_DIR / ".tmp_extract"
    tmp_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_csv_dir / f"bts_{year}_{month:02d}.csv"

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            print(f"[bad-zip] {zip_path.name} has no CSV inside, skipping")
            return None
        with zf.open(csv_names[0]) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    con.execute(f"""
        COPY (SELECT * FROM read_csv_auto('{csv_path.as_posix()}', ignore_errors=true))
        TO '{out_path.as_posix()}' (FORMAT PARQUET)
    """)
    n_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]

    csv_path.unlink()
    zip_path.unlink()
    print(f"[ok] bts_{year}_{month:02d}: {n_rows:,} rows -> {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return (year, month)


def main():
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    zips = sorted(DOWNLOADS_DIR.glob("On_Time_Reporting_Carrier_On_Time_Performance_1987_present_*.zip"))
    print(f"[found] {len(zips)} BTS zip file(s) in {DOWNLOADS_DIR}")

    done = []
    for zip_path in zips:
        result = process_zip(zip_path, con)
        if result:
            done.append(result)

    tmp_dir = PARQUET_DIR / ".tmp_extract"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    all_parquets = sorted(PARQUET_DIR.glob("bts_*.parquet"))
    print(f"\n[summary] {len(done)} newly processed this run, {len(all_parquets)} total months in {PARQUET_DIR}")

    expected = {(y, m) for y in range(2018, 2024) for m in range(1, 13)}
    have = set()
    for p in all_parquets:
        mm = re.match(r"bts_(\d{4})_(\d{2})\.parquet$", p.name)
        if mm:
            have.add((int(mm.group(1)), int(mm.group(2))))
    missing = sorted(expected - have)
    if missing:
        print(f"[missing] {len(missing)} month(s) still not downloaded: {missing}")


if __name__ == "__main__":
    main()
