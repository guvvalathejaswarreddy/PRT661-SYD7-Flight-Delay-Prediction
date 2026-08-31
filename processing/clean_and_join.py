"""
PRT661-7 / PRT661-8: Clean and fix data + combine flight and weather/airport data.

Reads raw BTS monthly data (Parquet, converted from the downloaded CSVs by
ingestion/import_from_downloads.py to save disk space) and the OpenFlights
airport table with DuckDB, cleans them, joins flights to origin/destination
airport metadata, and writes a single curated Parquet table.

NOTE: NOAA weather ingestion is not wired in yet (deferred -- see README);
this stage joins BTS + OpenFlights only for now.
"""

import shutil
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
BTS_RAW_GLOB = str(ROOT / "dataset" / "bts_parquet" / "*.parquet")
AIRPORTS_RAW = ROOT / "dataset" / "openflights" / "airports.dat"
CURATED_DIR = ROOT / "storage" / "curated-zone"
CURATED_PATH = CURATED_DIR / "flights_curated.parquet"

AIRPORT_COLUMNS = [
    "airport_id", "name", "city", "country", "iata", "icao",
    "latitude", "longitude", "altitude", "timezone", "dst",
    "tz_database_timezone", "type", "source",
]


def main():
    CURATED_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # This machine has very little free disk (~17GB) and the full 35M-row
    # dataset overflowed DuckDB's default temp-spill space. Cap memory so it
    # spills to disk in smaller increments, turn off insertion-order
    # preservation (not needed here, and it forces extra buffering), and
    # point the temp directory at the same volume with an explicit cap.
    con.execute("PRAGMA memory_limit='4GB'")
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA preserve_insertion_order=false")
    temp_dir = ROOT / "dataset" / ".duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}'")

    # Airports: no header in the raw file, so name the columns explicitly.
    con.execute(f"""
        CREATE VIEW airports AS
        SELECT * FROM read_csv_auto(
            '{AIRPORTS_RAW.as_posix()}',
            header=false,
            names={AIRPORT_COLUMNS}
        )
    """)
    con.execute("""
        CREATE TABLE airports_clean AS
        SELECT DISTINCT ON (iata) iata, latitude, longitude, city, country
        FROM airports
        WHERE iata IS NOT NULL AND iata != '\\N' AND iata != ''
    """)

    # Raw BTS flights, all months -- a view (lazy), not a materialized table,
    # so the ~35M rows are never fully duplicated in memory/temp at once.
    con.execute(f"""
        CREATE VIEW flights_raw AS
        SELECT * FROM read_parquet('{BTS_RAW_GLOB}', union_by_name=true)
    """)

    n_raw = con.execute("SELECT COUNT(*) FROM flights_raw").fetchone()[0]
    print(f"[load] {n_raw:,} raw flight records")

    # Clean: drop cancelled/diverted flights (no arrival delay to learn from),
    # dedupe on the flight-leg key, join to airport metadata, and write the
    # curated Parquet directly from one query -- avoids materializing a
    # separate flights_clean / flights_curated table on top of flights_raw.
    con.execute(f"""
        COPY (
            SELECT
                f.flight_date, f.carrier, f.origin, f.destination,
                f.crs_dep_time, f.distance, f.arr_del15, f.arr_delay_minutes,
                o.latitude  AS origin_lat,
                o.longitude AS origin_lon,
                d.latitude  AS dest_lat,
                d.longitude AS dest_lon
            FROM (
                SELECT DISTINCT ON (FlightDate, Reporting_Airline, Flight_Number_Reporting_Airline, Origin, Dest)
                    CAST(FlightDate AS DATE)                AS flight_date,
                    Reporting_Airline                        AS carrier,
                    Origin                                   AS origin,
                    Dest                                      AS destination,
                    CAST(CRSDepTime AS INTEGER)              AS crs_dep_time,
                    CAST(Distance AS DOUBLE)                 AS distance,
                    CAST(COALESCE(ArrDel15, 0) AS INTEGER)   AS arr_del15,
                    CAST(COALESCE(ArrDelayMinutes, 0) AS DOUBLE) AS arr_delay_minutes
                FROM flights_raw
                WHERE Cancelled = 0 AND Diverted = 0
                  AND FlightDate IS NOT NULL
                  AND Origin IS NOT NULL AND Dest IS NOT NULL
            ) f
            LEFT JOIN airports_clean o ON f.origin = o.iata
            LEFT JOIN airports_clean d ON f.destination = d.iata
        ) TO '{CURATED_PATH.as_posix()}' (FORMAT PARQUET)
    """)

    n_curated = con.execute(f"SELECT COUNT(*) FROM read_parquet('{CURATED_PATH.as_posix()}')").fetchone()[0]
    print(f"[write] {n_curated:,} curated records -> {CURATED_PATH}")

    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
