"""
PRT661-4: Set up data storage (airport metadata leg).

Downloads the OpenFlights airport database and lands it in the raw-zone.
Source: https://openflights.org/data.php
Direct file (verified working):
  https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat
"""

import subprocess
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "dataset" / "openflights"
URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"

# OpenFlights airports.dat has no header row; these are the documented columns.
COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "latitude", "longitude", "altitude", "timezone", "dst",
    "tz_database_timezone", "type", "source",
]


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "airports.dat"

    if target.exists():
        print(f"[skip] {target.name} already downloaded")
    else:
        print(f"[download] {URL}")
        subprocess.run(
            ["curl", "-4", "-sS", "-L", "--max-time", "60", "-o", str(target), URL],
            check=True,
        )
        print(f"[ok] saved {target.name} ({target.stat().st_size / 1e3:.0f} KB)")

    # Write the column header alongside it so downstream DuckDB reads are simple.
    header_path = RAW_DIR / "airports_columns.txt"
    header_path.write_text(",".join(COLUMNS))


if __name__ == "__main__":
    main()
