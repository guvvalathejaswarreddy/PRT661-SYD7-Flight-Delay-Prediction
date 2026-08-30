"""
PRT661-2: Download flight delay data.

Pulls monthly BTS "Airline On-Time Performance" zip files straight from the
official Bureau of Transportation Statistics endpoint (the same dataset named
in the Assessment 1 proposal) and lands the raw CSVs in the local raw-zone.

Official source: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ
Direct file pattern (verified working):
  https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip

Usage:
  python3 ingestion/download_bts.py                # downloads YEAR_MONTHS below
  python3 ingestion/download_bts.py 2023 1 2 3      # downloads 2023 Jan-Mar
"""

import subprocess
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "bts"
URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

# Full Assessment 1 scope: 2018-2023 (~36M rows / ~12GB across 72 months).
DEFAULT_YEAR_MONTHS = [(year, month) for year in range(2018, 2024) for month in range(1, 13)]


def zip_is_intact(zip_path: Path) -> bool:
    """zipfile.is_zipfile() only checks the end-of-archive marker -- a
    connection drop mid-transfer can still leave that marker intact while the
    actual entry data is truncated/corrupt. testzip() reads every entry's
    CRC and catches that."""
    if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download_month(year: int, month: int) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = URL_TEMPLATE.format(year=year, month=month)
    zip_path = RAW_DIR / f"bts_{year}_{month:02d}.zip"
    csv_path = RAW_DIR / f"bts_{year}_{month:02d}.csv"

    if csv_path.exists():
        print(f"[skip] {csv_path.name} already extracted")
        return RAW_DIR

    if zip_is_intact(zip_path):
        print(f"[skip] {zip_path.name} already downloaded")
    else:
        if zip_path.exists():
            zip_path.unlink()  # drop any leftover corrupt/partial file before retrying
        print(f"[download] {url}")
        # -4 forces IPv4 (IPv6/NAT64 route hangs on this network); -C - resumes
        # a partial download; -c/-b keep the F5 load-balancer session cookie
        # across retries, without which the server resets mid-transfer.
        cookie_jar = RAW_DIR / ".bts_cookies.txt"
        for attempt in range(1, 11):
            result = subprocess.run(
                ["curl", "-4", "-sS", "-L", "-c", str(cookie_jar), "-b", str(cookie_jar),
                 "-C", "-", "--retry-all-errors", "--retry", "8", "--retry-delay", "3",
                 "--max-time", "900", "-o", str(zip_path), url],
            )
            if result.returncode == 0 and zip_is_intact(zip_path):
                break
            print(f"[retry {attempt}] download incomplete or corrupt (curl exit {result.returncode}), retrying...")
            zip_path.unlink(missing_ok=True)  # force a clean re-download, not a resume onto bad bytes
        else:
            raise RuntimeError(f"Failed to fully download {url} after retries")
        print(f"[ok] saved {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")

    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        for name in csv_names:
            if not csv_path.exists():
                with zf.open(name) as src, open(csv_path, "wb") as dst:
                    dst.write(src.read())
                print(f"[extract] {csv_path.name}")

    # Delete the zip once the CSV is safely extracted -- keeping both roughly
    # doubles disk usage for no benefit once the month is processed.
    zip_path.unlink(missing_ok=True)
    return RAW_DIR


def main():
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
        months = [int(m) for m in sys.argv[2:]] or [1]
        year_months = [(year, m) for m in months]
    else:
        year_months = DEFAULT_YEAR_MONTHS

    failed = []
    for year, month in year_months:
        try:
            download_month(year, month)
        except Exception as exc:
            print(f"[FAIL] {year}-{month:02d}: {exc}")
            failed.append((year, month))

    if failed:
        failed_path = RAW_DIR / "FAILED_MONTHS.txt"
        failed_path.write_text("\n".join(f"{y} {m}" for y, m in failed) + "\n")
        print(f"[summary] {len(failed)} month(s) failed after retries: {failed} -> see {failed_path}")
        sys.exit(2)

    (RAW_DIR / "FAILED_MONTHS.txt").unlink(missing_ok=True)
    print(f"[summary] all {len(year_months)} months downloaded successfully")


if __name__ == "__main__":
    main()
