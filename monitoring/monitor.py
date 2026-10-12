"""Monthly monitoring of the deployed delay model (Assessment 4, PRT661-23).

Reads scored flights with outcomes (default: the 2023 hold-out sample used by the dashboard) and reports, per month,
discrimination (AUC), calibration (mean gap, ECE, Brier), alert behaviour at the chosen alert budget and population drift
of the predicted probability (PSI against a reference window). A retraining trigger fires when any rule below is broken.

Usage: python3 monitoring/monitor.py [--sample PATH] [--reference-months 2]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "dashboard" / "data" / "a4_predictions_sample.parquet"
A4 = ROOT / "dashboard" / "assets" / "a4"
OUT_FIG = ROOT / "dashboard" / "assets" / "figures" / "30_monitoring_2023.png"

# Rules (set before looking at the 2023 results; documented in the report, section 3.9).
#  Retraining triggers use 3-month pooled windows so that seasonality does not trigger them:
#    R1  3-month AUC below the 2023 test AUC of the deployed model (0.6824) minus 0.03
#    R2  3-month |mean predicted - observed delay rate| above 0.02   (a calibrator refit may be enough)
#  Advisory signals (reported, not triggers, because delay rates are strongly seasonal):
#    single-month PSI of the predicted probability above 0.25, alert share outside 10% +/- 5 points
VALIDATED_AUC = 0.6824                        # full-year 2023 test AUC of the deployed model (notebook §21)
RULES = {"auc_drop": 0.03, "calibration_gap": 0.02, "psi": 0.25, "alert_share_tolerance": 0.05, "window_months": 3}


def ece(y, p, bins=15):
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    cnt = np.bincount(b)
    obs = np.bincount(b, weights=y) / np.maximum(cnt, 1)
    pred = np.bincount(b, weights=p) / np.maximum(cnt, 1)
    return float(np.sum(cnt * np.abs(obs - pred)) / len(y))


def psi(ref, cur, bins=10):
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-6, None), np.clip(c, 1e-6, None)
    return float(np.sum((c - r) * np.log(c / r)))


def run(sample=SAMPLE, ref_months=2):
    s = pd.read_parquet(sample)
    s["month"] = pd.to_datetime(s["flight_date"]).dt.to_period("M")
    thr = json.loads((A4 / "a4_thresholds.json").read_text())["thresholds"]["budget_10pct"]
    months = sorted(s["month"].unique())
    ref = s[s["month"].isin(months[:ref_months])]
    rows = []
    for i, m in enumerate(months):
        g = s[s["month"] == m]
        y, p = g["actual_delayed"].to_numpy().astype(int), g["predicted_delay_prob"].to_numpy()
        flag = p >= thr
        w = s[s["month"].isin(months[max(0, i - RULES["window_months"] + 1): i + 1])] if i + 1 >= RULES["window_months"] else None
        rows.append({"month": str(m), "flights_in_sample": len(g), "delay_rate": y.mean(), "mean_predicted": p.mean(),
                     "calibration_gap": p.mean() - y.mean(), "ECE": ece(y, p), "brier": brier_score_loss(y, p),
                     "auc": roc_auc_score(y, p), "alert_share": flag.mean(),
                     "alert_precision": y[flag].mean() if flag.any() else np.nan,
                     "psi_vs_reference": psi(ref["predicted_delay_prob"].to_numpy(), p),
                     "auc_3m": roc_auc_score(w["actual_delayed"], w["predicted_delay_prob"]) if w is not None else np.nan,
                     "gap_3m": (w["predicted_delay_prob"].mean() - w["actual_delayed"].mean()) if w is not None else np.nan})
    t = pd.DataFrame(rows)
    t["R1_auc_trigger"] = t["auc_3m"] < VALIDATED_AUC - RULES["auc_drop"]
    t["R2_calibration_trigger"] = t["gap_3m"].abs() > RULES["calibration_gap"]
    t["advisory_psi"] = t["psi_vs_reference"] > RULES["psi"]
    t["advisory_alert_share"] = (t["alert_share"] - 0.10).abs() > RULES["alert_share_tolerance"]
    t["retrain_trigger"] = t["R1_auc_trigger"] | t["R2_calibration_trigger"]
    return t, VALIDATED_AUC, thr


def plot(t, ref_auc):
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.4))
    x = range(len(t)); lab = [m[5:] for m in t["month"]]
    ax[0, 0].plot(x, t["auc"], marker="o", label="monthly AUC"); ax[0, 0].plot(x, t["auc_3m"], marker="s", color="#4C72B0", label="3-month AUC"); ax[0, 0].axhline(ref_auc - RULES["auc_drop"], color="#C44E52", ls="--", label=f"R1 trigger ({ref_auc - RULES['auc_drop']:.3f})")
    ax[0, 0].set_title("Discrimination (AUC)"); ax[0, 0].legend(fontsize=8)
    ax[0, 1].bar(x, t["calibration_gap"], color="#55A868"); ax[0, 1].plot(x, t["gap_3m"], marker="s", color="#4C72B0", label="3-month gap"); ax[0, 1].axhline(RULES["calibration_gap"], color="#C44E52", ls="--", label="R2 trigger (+/-0.02)"); ax[0, 1].axhline(-RULES["calibration_gap"], color="#C44E52", ls="--")
    ax[0, 1].set_title("Calibration gap (mean predicted - observed)"); ax[0, 1].legend(fontsize=8)
    ax[1, 0].plot(x, t["alert_share"] * 100, marker="o", color="#DD8452"); ax[1, 0].axhline(10, color="grey", ls=":")
    ax[1, 0].set_title("Share of flights alerted (budget 10%)"); ax[1, 0].set_ylabel("%")
    ax[1, 1].plot(x, t["psi_vs_reference"], marker="o", color="#8172B2"); ax[1, 1].axhline(RULES["psi"], color="#999999", ls="--")
    ax[1, 1].set_title("Drift of predicted probability (PSI, advisory)")
    for a in ax.ravel():
        a.set_xticks(list(x)); a.set_xticklabels(lab); a.set_xlabel("2023 month")
    fig.tight_layout(); OUT_FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(OUT_FIG, dpi=170); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", default=str(SAMPLE)); ap.add_argument("--reference-months", type=int, default=2)
    a = ap.parse_args()
    table, ref_auc, thr = run(Path(a.sample), a.reference_months)
    table.to_csv(A4 / "monitoring_2023.csv", index=False)
    plot(table, ref_auc)
    pd.set_option("display.width", 250)
    print(f"reference AUC {ref_auc:.4f}; alert threshold {thr:.4f}")
    print(table.round(4).to_string(index=False))
    print("retraining trigger (R1/R2) fired in:", list(table.loc[table["retrain_trigger"], "month"]) or "no month")
    print("advisory PSI months:", list(table.loc[table["advisory_psi"], "month"]), "| advisory alert-share months:", list(table.loc[table["advisory_alert_share"], "month"]))
