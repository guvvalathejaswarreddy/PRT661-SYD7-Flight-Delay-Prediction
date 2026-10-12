"""Automated checks for the Assessment 4 pipeline and dashboard assets. Run: python3 -m pytest tests -q"""
import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "processing"), str(ROOT / "dashboard")]
warnings.filterwarnings("ignore")

import calendar_features as cf  # noqa: E402
import noaa_live  # noqa: E402


def _flights(n_days=60, seed=0):
    rng = np.random.RandomState(seed)
    days = pd.date_range("2022-11-01", periods=n_days)
    rows = []
    for d in days:
        for c in ("AA", "DL", "WN"):
            for _ in range(rng.randint(3, 9)):
                rows.append((d, c, rng.choice(["ATL", "ORD", "DEN"]), rng.choice(["LAX", "JFK", "SEA", "MIA"]), int(rng.rand() < 0.2)))
    df = pd.DataFrame(rows, columns=["flight_date", "carrier", "origin", "destination", "arr_del15"])
    df["year"] = df["flight_date"].dt.year
    return df


# ---- rolling rates ---------------------------------------------------------------------------------------------
def test_rolling_rate_equals_explicit_calendar_loop():
    df = _flights()
    out = cf.add_rolling_rates(df, train_years=(2022,))
    fill = df.loc[df["year"] == 2022, "arr_del15"].mean()
    for i in np.random.RandomState(1).choice(len(df), 200, replace=False):
        r = df.iloc[i]
        w = df[(df["carrier"] == r["carrier"]) & (df["flight_date"] >= r["flight_date"] - pd.Timedelta(days=7))
               & (df["flight_date"] < r["flight_date"])]
        expected = w["arr_del15"].mean() if len(w) else fill
        assert out["carrier_delay_rate_7d"].iloc[i] == pytest.approx(expected, abs=1e-5)


def test_same_day_outcomes_never_enter_their_own_feature():
    df = _flights()
    day = pd.Timestamp("2022-12-05")
    base = cf.add_rolling_rates(df, (2022,))
    flipped = df.copy()
    m = flipped["flight_date"] == day
    flipped.loc[m, "arr_del15"] = 1 - flipped.loc[m, "arr_del15"]
    new = cf.add_rolling_rates(flipped, (2022,))
    same = df["flight_date"] == day
    nxt = df["flight_date"] == day + pd.Timedelta(days=1)
    assert np.allclose(base.loc[same, "carrier_delay_rate_7d"], new.loc[same, "carrier_delay_rate_7d"])     # unchanged on day d
    assert not np.allclose(base.loc[nxt, "carrier_delay_rate_7d"], new.loc[nxt, "carrier_delay_rate_7d"])   # reaches day d+1


# ---- graph features --------------------------------------------------------------------------------------------
def test_graph_features_use_training_years_only():
    df = _flights()
    late = pd.DataFrame({"flight_date": [pd.Timestamp("2023-02-01")] * 3, "carrier": "AA", "origin": "ZZZ",
                         "destination": ["LAX", "JFK", "SEA"], "arr_del15": 0, "year": 2023})
    df = pd.concat([df, late], ignore_index=True)
    out = cf.add_graph_features(df, train_years=(2022,))
    z = out[out["origin"] == "ZZZ"]
    assert (z["origin_hub_tier"].astype(str) == "small").all() and (z["origin_degree_centrality"] == 0).all()
    assert out["origin_degree_centrality"].between(0, 1).all()


# ---- NOAA live lookup ------------------------------------------------------------------------------------------
def test_noaa_quality_flag_drops_value():
    assert np.isnan(noaa_live._clean("5.0", ",D,W,2400"))       # failed a quality check
    assert noaa_live._clean("5.0", ",,W,2400") == 5.0
    assert np.isnan(noaa_live._clean("", ",,W"))


def test_noaa_uses_prior_day_only(monkeypatch):
    flight_day = dt.date(2023, 7, 14)

    class R:
        def raise_for_status(self): pass
        def json(self):
            rec = lambda d, p: {"STATION": "S1", "DATE": d, "PRCP": str(p), "PRCP_ATTRIBUTES": ",,W", "TMAX": "30", "TMAX_ATTRIBUTES": ",,W",
                                "TMIN": "20", "TMIN_ATTRIBUTES": ",,W", "AWND": "3", "AWND_ATTRIBUTES": ",,W"}
            return [rec("2023-07-13", 4.0), rec("2023-07-12", 1.0), rec("2023-07-11", 2.0), rec("2023-07-14", 999.0)]   # d0 must be ignored
    monkeypatch.setattr(noaa_live.requests, "get", lambda *a, **k: R())
    smap = pd.DataFrame({"iata": "ATL", "element": ["PRCP", "TMAX", "TMIN", "AWND", "SNOW", "SNWD", "WSF2", "WT01", "WT03"],
                         "station": "S1", "dist_km": 1.0, "rnk": 1, "name": "TEST", "state": "GA"})
    res = noaa_live.fetch_prior_day("ATL", flight_day, smap)
    assert res["values"]["PRCP"] == 4.0 and res["prcp_3d"] == 7.0 and res["date"] == dt.date(2023, 7, 13)


def test_noaa_unmapped_airport_raises():
    with pytest.raises(noaa_live.NoaaUnavailable):
        noaa_live.fetch_prior_day("ZZZ", dt.date(2023, 1, 2), pd.DataFrame(columns=["iata", "element", "station", "dist_km", "name"]))


# ---- model and dashboard assets --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def models():
    import lib
    clf, dur = lib.load_a4_models()
    assert clf is not None and dur is not None
    return lib, clf, dur


def _row(clf, **over):
    feats = clf["features"]
    r = {f: np.nan for f in feats}
    r.update(crs_dep_hour=17.0, day_of_week=4.0, is_weekend=0.0, month=7.0, is_holiday=0.0, distance=760.0,
             carrier_delay_rate_7d=0.2, origin_delay_rate_7d=0.2, origin_degree_centrality=0.9, carrier="AA", origin_hub_tier="major")
    r.update(over)
    X = pd.DataFrame([r])[feats]
    X["carrier"] = pd.Categorical(X["carrier"], categories=clf["carrier_categories"])
    X["origin_hub_tier"] = pd.Categorical(X["origin_hub_tier"], categories=clf["tiers"])
    return X


def test_calibrated_probability_is_valid_and_monotone(models):
    lib, clf, _ = models
    grid = np.linspace(0.05, 0.95, 50)
    p = lib.calibrate(grid, clf)
    assert p.min() >= 0 and p.max() <= 1 and np.all(np.diff(p) >= -1e-12)


def test_one_flight_scores_with_missing_weather(models):
    lib, clf, dur = models
    X = _row(clf)                                              # every weather input missing
    raw = clf["model"].predict_proba(X)[0, 1]
    prob = float(lib.calibrate([raw], clf)[0])
    q = np.sort([dur["models"][a].predict(X)[0] for a in dur["quantiles"]])
    assert 0 < prob < 1 and q[0] > 0 and q[0] <= q[1] <= q[2]


def test_rain_and_thunder_change_the_prediction(models):
    lib, clf, _ = models
    dry = clf["model"].predict_proba(_row(clf, o_prcp_d1=0.0, o_thunder_d1=0.0))[0, 1]
    wet = clf["model"].predict_proba(_row(clf, o_prcp_d1=25.0, o_thunder_d1=1.0))[0, 1]
    assert dry != wet


def test_thresholds_are_ordered(models):
    lib, _, _ = models
    t = lib.a4_json("a4_thresholds.json")["thresholds"]
    assert t["budget_5pct"] > t["budget_10pct"] > t["budget_20pct"]
    assert t["cost_theory"] == pytest.approx(0.25)
    assert len(lib.a4_operating_levels()) == 4


def test_prediction_sample_schema_and_calibration():
    s = pd.read_parquet(ROOT / "dashboard" / "data" / "a4_predictions_sample.parquet")
    need = {"flight_date", "carrier", "origin", "destination", "actual_delayed", "predicted_delay_prob",
            "predicted_delay_minutes_q10", "predicted_delay_minutes_q50", "predicted_delay_minutes_q90"}
    assert need <= set(s.columns)
    assert s["predicted_delay_prob"].between(0, 1).all()
    assert abs(s["predicted_delay_prob"].mean() - s["actual_delayed"].mean()) < 0.02        # calibrated on average
    assert (s["predicted_delay_minutes_q10"] <= s["predicted_delay_minutes_q90"]).all()
