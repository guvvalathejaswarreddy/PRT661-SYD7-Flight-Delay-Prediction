"""Live aircraft data from the OpenSky Network (PRT661-16).

Free, research-licensed ADS-B feed (Schafer et al., 2014). Used here to add a
live "what's happening near this airport right now" context panel to the
Predict a Flight dashboard page (PRT661-17) alongside the existing model,
which is trained on the static 2018-2023 BTS extract.

This module is intentionally independent of the trained model: it does not
add, remove, or change any of the 11 features in dashboard/lib.py::ALL_FEATURES,
so nothing here requires retraining the classifier or regressor. It only
supplies a live snapshot for display.

Auth: OpenSky uses OAuth2 client-credentials (client_id + client_secret),
generated from an "API client" in your OpenSky account settings. Set these
as environment variables (e.g. in a local, git-ignored .env) -- never commit
them:

    OPENSKY_CLIENT_ID=...
    OPENSKY_CLIENT_SECRET=...

Anonymous (unauthenticated) calls to /states/all also work at a lower rate
limit, so the panel still degrades gracefully with no credentials at all.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import NamedTuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AIRPORTS_DAT = PROJECT_ROOT / "dataset" / "openflights" / "airports.dat"


def _load_dotenv_if_present() -> None:
    """Minimal .env loader (no extra dependency) so running the dashboard or
    this script directly picks up OPENSKY_CLIENT_ID/SECRET automatically."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv_if_present()

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/opensky-network"
    "/protocol/openid-connect/token"
)
STATES_URL = "https://opensky-network.org/api/states/all"

_STATE_FIELDS = [
    "icao24", "callsign", "origin_country", "time_position", "last_contact",
    "longitude", "latitude", "baro_altitude", "on_ground", "velocity",
    "true_track", "vertical_rate", "sensors", "geo_altitude", "squawk",
    "spi", "position_source",
]

_token_cache: dict = {"access_token": None, "expires_at": 0.0}


class OpenSkyError(RuntimeError):
    """Raised for auth/network failures; callers should degrade gracefully."""


def _get_access_token(client_id: str, client_secret: str) -> str:
    """Client-credentials OAuth2 flow. Tokens are cached for their lifetime
    (minus a safety margin) so we don't re-authenticate on every call."""
    now = time.time()
    if _token_cache["access_token"] and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    try:
        resp = requests.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": client_id,
                "client_secret": client_secret,
            },
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OpenSkyError(f"OpenSky auth request failed: {exc}") from exc

    payload = resp.json()
    _token_cache["access_token"] = payload["access_token"]
    _token_cache["expires_at"] = now + float(payload.get("expires_in", 1800)) - 30
    return _token_cache["access_token"]


def _airport_lookup() -> dict[str, tuple[float, float]]:
    """IATA -> (lat, lon) from the already-committed OpenFlights extract
    (dataset/openflights/airports.dat), so no new data source is needed."""
    lookup: dict[str, tuple[float, float]] = {}
    if not AIRPORTS_DAT.exists():
        return lookup
    with open(AIRPORTS_DAT, encoding="utf-8") as f:
        for row in csv.reader(f):
            # id, name, city, country, iata, icao, lat, lon, ...
            if len(row) < 8:
                continue
            iata = row[4].strip('"')
            if not iata or iata == "\\N":
                continue
            try:
                lookup[iata] = (float(row[6]), float(row[7]))
            except ValueError:
                continue
    return lookup


_AIRPORTS = _airport_lookup()


class LiveSnapshot(NamedTuple):
    airport: str
    generated_unix: int
    total_aircraft: int
    airborne: int
    on_ground: int
    aircraft: list[dict]


def _bbox_around(lat: float, lon: float, radius_deg: float = 0.75) -> tuple[float, float, float, float]:
    return (lat - radius_deg, lon - radius_deg, lat + radius_deg, lon + radius_deg)


def get_live_snapshot(iata: str, radius_deg: float = 0.75) -> LiveSnapshot:
    """Aircraft currently visible on ADS-B within `radius_deg` (~50-80km) of
    `iata`. Raises OpenSkyError on any failure -- callers (the dashboard) are
    expected to catch this and show a friendly fallback rather than crash."""
    iata = iata.upper()
    coords = _AIRPORTS.get(iata)
    if coords is None:
        raise OpenSkyError(f"No coordinates for airport '{iata}' in OpenFlights data.")
    lamin, lomin, lamax, lomax = _bbox_around(*coords, radius_deg)

    headers = {}
    client_id = os.environ.get("OPENSKY_CLIENT_ID")
    client_secret = os.environ.get("OPENSKY_CLIENT_SECRET")
    if client_id and client_secret:
        headers["Authorization"] = f"Bearer {_get_access_token(client_id, client_secret)}"

    try:
        resp = requests.get(
            STATES_URL,
            params={"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OpenSkyError(f"OpenSky states request failed: {exc}") from exc

    payload = resp.json()
    raw_states = payload.get("states") or []
    aircraft = [dict(zip(_STATE_FIELDS, s)) for s in raw_states]
    on_ground = sum(1 for a in aircraft if a.get("on_ground"))

    return LiveSnapshot(
        airport=iata,
        generated_unix=int(payload.get("time") or time.time()),
        total_aircraft=len(aircraft),
        airborne=len(aircraft) - on_ground,
        on_ground=on_ground,
        aircraft=aircraft,
    )


if __name__ == "__main__":
    import sys

    code = sys.argv[1] if len(sys.argv) > 1 else "ATL"
    snap = get_live_snapshot(code)
    print(f"{snap.airport}: {snap.total_aircraft} aircraft nearby "
          f"({snap.airborne} airborne, {snap.on_ground} on ground)")
    for a in snap.aircraft[:5]:
        print(f"  {a['callsign']!s:<10} alt={a['baro_altitude']} "
              f"vel={a['velocity']} on_ground={a['on_ground']}")
