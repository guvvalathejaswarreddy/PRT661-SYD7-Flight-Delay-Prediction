"""Prior-day NOAA GHCN-Daily observations for an airport, fetched live from NCEI's Access Data Service.

Uses the same airport -> nearest-station map the model was built with (assets/a4/airport_weather_station_map.csv),
the same element set, the same units (mm, deg C, m/s) and the same quality rule (rows with a quality flag are dropped).
For a flight on day d the observations are those of day d-1 (precipitation also d-3..d-1), exactly as in training.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import requests

API = "https://www.ncei.noaa.gov/access/services/data/v1"
ELEMENTS = ["PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND", "WSF2", "WT01", "WT03"]
NAN = float("nan")


class NoaaUnavailable(RuntimeError):
    pass


def _clean(value, attrs) -> float:
    """Value as float; NaN when missing or when the quality flag (2nd attribute field) is set."""
    if value in (None, ""):
        return NAN
    flags = (attrs or "").split(",")
    if len(flags) > 1 and flags[1].strip():
        return NAN
    try:
        return float(value)
    except ValueError:
        return NAN


def fetch_prior_day(iata: str, flight_date: dt.date, station_map: pd.DataFrame, timeout: int = 20) -> dict:
    """Returns {"values": {element: value on d-1}, "prcp_3d": float, "stations": DataFrame} for one airport.

    Raises NoaaUnavailable when the airport has no mapped station or the service cannot be reached.
    """
    rows = station_map[station_map["iata"] == iata]
    if rows.empty:
        raise NoaaUnavailable(f"no weather station mapped to {iata}")
    stations = sorted(rows["station"].unique())
    d1 = flight_date - dt.timedelta(days=1)
    try:
        r = requests.get(API, params={
            "dataset": "daily-summaries", "stations": ",".join(stations), "dataTypes": ",".join(ELEMENTS),
            "startDate": (flight_date - dt.timedelta(days=3)).isoformat(), "endDate": d1.isoformat(),
            "format": "json", "units": "metric", "includeAttributes": "true"}, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise NoaaUnavailable(f"NOAA service not reachable ({type(exc).__name__})") from exc

    obs = {}                                              # (station, date) -> record
    for rec in data:
        obs[(rec.get("STATION"), rec.get("DATE"))] = rec
    values, used = {}, []
    for _, m in rows.sort_values("element").iterrows():
        el, st = m["element"], m["station"]
        rec = obs.get((st, d1.isoformat()))
        v = _clean(rec.get(el), rec.get(f"{el}_ATTRIBUTES")) if rec else NAN
        if el in ("WT01", "WT03"):                        # weather-type flags are only reported when present
            v = 0.0 if math.isnan(v) else v
        values[el] = v
        used.append({"element": el, "station": st, "station_name": m["name"], "distance_km": m["dist_km"], "value": v})
    # precipitation over d-3 .. d-1 from the PRCP station
    prcp_st = rows.loc[rows["element"] == "PRCP", "station"]
    prcp_3d = NAN
    if not prcp_st.empty:
        vals = []
        for k in (1, 2, 3):
            rec = obs.get((prcp_st.iloc[0], (flight_date - dt.timedelta(days=k)).isoformat()))
            v = _clean(rec.get("PRCP"), rec.get("PRCP_ATTRIBUTES")) if rec else NAN
            if not math.isnan(v):
                vals.append(v)
        prcp_3d = float(sum(vals)) if vals else NAN
    return {"values": values, "prcp_3d": prcp_3d, "stations": pd.DataFrame(used), "date": d1}


def to_model_inputs(res: dict) -> dict:
    """NOAA observation dict -> the dict of names the Predict page uses (prcp, snow, ..., prcp3)."""
    v = res["values"]
    if all(math.isnan(v.get(k, NAN)) for k in ("PRCP", "TMAX", "TMIN", "AWND")):      # no observations at all (e.g. future date)
        return dict(prcp=NAN, snow=NAN, snwd=NAN, tmax=NAN, tmin=NAN, awnd=NAN, wsf2=NAN, thunder=NAN, fog=NAN, prcp3=NAN)
    return dict(prcp=v.get("PRCP", NAN), snow=v.get("SNOW", NAN), snwd=v.get("SNWD", NAN), tmax=v.get("TMAX", NAN),
                tmin=v.get("TMIN", NAN), awnd=v.get("AWND", NAN), wsf2=v.get("WSF2", NAN),
                thunder=v.get("WT03", NAN), fog=v.get("WT01", NAN), prcp3=res["prcp_3d"])
