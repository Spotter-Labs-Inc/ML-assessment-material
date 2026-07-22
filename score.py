"""
Scorer for the freight-rate take-home.

Computes MAE, MAPE (%), RMSE and R^2 for a submission, both overall and
segmented by lane_density_bucket. Ships to candidates too, so they can score
their own time-based holdout (it works against any labeled file that has
`load_id` and `posted_rate`).

Usage:
    python score.py --predictions preds.csv --labels data/test_labels.csv

`preds.csv` must contain columns: load_id, predicted_rate
`labels`    must contain columns: load_id, posted_rate
            (optionally lane_density_bucket for the segmented breakdown)
"""

import argparse
import sys

import numpy as np
import pandas as pd


def _die(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = y_true != 0
    mape = float(np.mean(np.abs(err[nz] / y_true[nz])) * 100) if nz.any() else float("nan")
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n": len(y_true), "MAE": mae, "MAPE%": mape, "RMSE": rmse, "R2": r2}


def _row(label, m):
    return (f"{label:<10} {m['n']:>7,}  {m['MAE']:>10.2f}  {m['MAPE%']:>8.2f}  "
            f"{m['RMSE']:>10.2f}  {m['R2']:>7.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels", required=True)
    args = ap.parse_args()

    try:
        preds = pd.read_csv(args.predictions)
        labels = pd.read_csv(args.labels)
    except Exception as e:  # noqa: BLE001
        _die(f"could not read input: {e}")

    for col in ("load_id", "predicted_rate"):
        if col not in preds.columns:
            _die(f"predictions file missing required column '{col}'")
    for col in ("load_id", "posted_rate"):
        if col not in labels.columns:
            _die(f"labels file missing required column '{col}'")

    if preds["load_id"].duplicated().any():
        _die("predictions contain duplicate load_id values")

    merged = labels.merge(preds[["load_id", "predicted_rate"]], on="load_id", how="left")
    missing = int(merged["predicted_rate"].isna().sum())
    if missing:
        _die(f"{missing} label load_id(s) have no prediction "
             f"(expected {len(labels)} predictions)")

    if not np.isfinite(merged["predicted_rate"]).all():
        _die("predictions contain non-finite values (NaN/inf)")

    overall = metrics(merged["posted_rate"], merged["predicted_rate"])

    print("=" * 62)
    print(f"{'segment':<10} {'n':>7}  {'MAE':>10}  {'MAPE%':>8}  {'RMSE':>10}  {'R2':>7}")
    print("-" * 62)
    print(_row("OVERALL", overall))

    if "lane_density_bucket" in merged.columns:
        print("-" * 62)
        order = ["dense", "medium", "sparse", "unseen"]
        buckets = [b for b in order if b in set(merged["lane_density_bucket"])]
        buckets += [b for b in merged["lane_density_bucket"].unique()
                    if b not in order]
        for b in buckets:
            sub = merged[merged["lane_density_bucket"] == b]
            print(_row(b, metrics(sub["posted_rate"], sub["predicted_rate"])))
    print("=" * 62)


if __name__ == "__main__":
    main()
