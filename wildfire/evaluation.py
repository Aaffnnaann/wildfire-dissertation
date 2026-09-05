"""
Evaluation analysis from saved predictions: precision-recall curves, calibration
(reliability) curves, Brier scores, confusion matrices at a validation-selected
threshold, and error breakdowns by region, season and burned-area category.

Everything is computed from the `preds.npz` files written during training, so
the analysis is reproducible without retraining. Decision thresholds are chosen
on the validation split only; the test split is never used for tuning.

Run:
  python -m wildfire.evaluation --runs runs/seeds \\
      --models hist_grad_boost gru fusion_cross ensemble --out_dir figures
"""
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, precision_recall_curve)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire.classical import DATA
from wildfire.per_region import assign_region

PALETTE = ["#2a6fb0", "#e0761a", "#1b9e8a", "#c0392b", "#7b5ea7", "#8a8c90"]

# readable labels for figure legends
DISPLAY = {
    "ensemble": "Ensemble", "hist_grad_boost": "HistGradientBoosting",
    "random_forest": "Random Forest", "gru": "GRU", "lstm": "LSTM",
    "cnn1d": "CNN-1D", "bilstm": "BiLSTM", "rnn": "RNN",
    "temporal_transformer": "Transformer", "gtn": "GTN",
    "vit_v2": "ViT (masked)", "vit": "ViT",
    "fusion_concat": "Concat fusion", "fusion_cross_v2": "Gated fusion",
    "fusion_cross": "Cross-attention fusion",
}


def nice(name):
    return DISPLAY.get(name, name)


def load_preds(runs, name):
    """Load (y, p, y_val, p_val) for a model, averaging seeds when present."""
    direct = Path(runs) / name / "preds.npz"
    paths = [direct] if direct.exists() else sorted(
        Path(runs).glob(f"{name}_seed*/preds.npz"))
    if not paths:
        return None
    ds = [np.load(p) for p in paths]
    y = ds[0]["y"]
    p = np.mean([d["p"] for d in ds], axis=0)
    if all("p_val" in d.files for d in ds):
        return y, p, ds[0]["y_val"], np.mean([d["p_val"] for d in ds], axis=0)
    return y, p, None, None


def threshold_from_validation(y_val, p_val):
    """Threshold maximising F1 on validation data (test set never used)."""
    if y_val is None:
        return 0.5, False
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_val, (p_val >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]), True


def reliability(y, p, bins=10):
    """Equal-width reliability curve: mean predicted vs observed frequency."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(float(p[m].mean()))
        ys.append(float(y[m].mean()))
        ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def ece(y, p, bins=10):
    """Expected calibration error (weighted mean |confidence - accuracy|)."""
    xs, ys, ns = reliability(y, p, bins)
    if len(xs) == 0:
        return float("nan")
    return float(np.sum(ns * np.abs(xs - ys)) / np.sum(ns))


def burned_area_band(ba):
    if ba <= 0:
        return "no fire"
    if ba < 100:
        return "30-100 ha"
    if ba < 500:
        return "100-500 ha"
    if ba < 2000:
        return "500-2000 ha"
    return ">2000 ha"


SEASON = {12: "1 winter (DJF)", 1: "1 winter (DJF)", 2: "1 winter (DJF)",
          3: "2 spring (MAM)", 4: "2 spring (MAM)", 5: "2 spring (MAM)",
          6: "3 summer (JJA)", 7: "3 summer (JJA)", 8: "3 summer (JJA)",
          9: "4 autumn (SON)", 10: "4 autumn (SON)", 11: "4 autumn (SON)"}


def months_from_doy(years, doys):
    """Calendar month for each (year, day-of-year) pair."""
    return np.array([(date(int(yy), 1, 1) + timedelta(days=int(dd) - 1)).month
                     for yy, dd in zip(years, doys)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/seeds")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--error_model", default=None,
                    help="model used for the error breakdown (default: first model)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    loaded = {}
    for m in args.models:
        r = load_preds(args.runs, m)
        if r is None:
            print(f"[skip] no predictions for '{m}' under {args.runs}")
            continue
        loaded[m] = r
    if not loaded:
        print("no predictions found — run training first")
        return

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200})

    # ---------- precision-recall curves ----------
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for i, (m, (y, p, _, _)) in enumerate(loaded.items()):
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=PALETTE[i % len(PALETTE)], linewidth=1.8,
                label=f"{nice(m)} (AP={average_precision_score(y, p):.3f})")
    base = float(np.mean(list(loaded.values())[0][0]))
    ax.axhline(base, color="#444", linestyle="--", linewidth=1,
               label=f"chance ({base:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves (test set)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "pr_curves.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- calibration ----------
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.plot([0, 1], [0, 1], "--", color="#666", linewidth=1, label="perfect calibration")
    cal_rows = []
    for i, (m, (y, p, yv, pv)) in enumerate(loaded.items()):
        xs, ys, ns = reliability(y, p)
        ax.plot(xs, ys, "o-", color=PALETTE[i % len(PALETTE)], markersize=5,
                linewidth=1.6, label=nice(m))
        cal_rows.append({"model": m,
                         "brier": round(float(brier_score_loss(y, p)), 4),
                         "ece_10bin": round(ece(y, p), 4),
                         "auprc": round(float(average_precision_score(y, p)), 4)})
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fire frequency")
    ax.set_title("Reliability curves (test set)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "calibration.png", bbox_inches="tight")
    plt.close(fig)

    with (out / "calibration_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "auprc", "brier", "ece_10bin"])
        w.writeheader(); w.writerows(cal_rows)

    # ---------- confusion matrices at validation-selected thresholds ----------
    cm_rows = []
    for m, (y, p, yv, pv) in loaded.items():
        thr, from_val = threshold_from_validation(yv, pv)
        yhat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
        cm_rows.append({
            "model": m, "threshold": round(thr, 3),
            "threshold_source": "validation F1-max" if from_val else "default 0.5",
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "precision": round(float(tp / (tp + fp)) if tp + fp else 0.0, 4),
            "recall": round(float(tp / (tp + fn)) if tp + fn else 0.0, 4),
            "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        })
    with (out / "confusion_matrices.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cm_rows[0].keys()))
        w.writeheader(); w.writerows(cm_rows)

    # ---------- error analysis by region / season / burned-area band ----------
    em = args.error_model or list(loaded.keys())[0]
    y, p, yv, pv = loaded[em]
    meta = np.load(DATA / "test.npz")
    if len(meta["y"]) != len(y):
        print(f"[warn] test.npz has {len(meta['y'])} rows but predictions have {len(y)}; "
              "skipping error breakdown")
        return
    thr, _ = threshold_from_validation(yv, pv)
    yhat = (p >= thr).astype(int)
    lon, lat, ba = meta["lon"], meta["lat"], meta["ba"]
    month = months_from_doy(meta["year"], meta["doy"])

    def group_table(labels, title, fname):
        rows = []
        for g in sorted(set(labels)):
            m = np.array([lab == g for lab in labels])
            n = int(m.sum()); npos = int(y[m].sum())
            row = {"group": g, "n": n, "fires": npos,
                   "fire_rate": round(npos / n, 4) if n else 0.0}
            if 0 < npos < n:
                row["auprc"] = round(float(average_precision_score(y[m], p[m])), 4)
                row["lift_over_chance"] = round(row["auprc"] / (npos / n), 2)
                row["recall_at_thr"] = round(float(yhat[m][y[m] == 1].mean()), 4)
                fp_ = int(((yhat[m] == 1) & (y[m] == 0)).sum())
                row["false_alarm_rate"] = round(fp_ / int((y[m] == 0).sum()), 4)
            else:
                row.update({"auprc": None, "lift_over_chance": None,
                            "recall_at_thr": None, "false_alarm_rate": None})
            rows.append(row)
        with (out / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n{title} ({em}, threshold={thr:.2f})")
        for r in rows:
            print(f"  {str(r['group']):<24} n={r['n']:<6} fires={r['fires']:<5} "
                  f"AUPRC={r['auprc']} lift={r['lift_over_chance']}")
        return rows

    regions = [assign_region(a, b) for a, b in zip(lon, lat)]
    group_table(regions, "Error analysis by region", "error_by_region.csv")
    group_table([f"month {mm:02d}" for mm in month], "Error analysis by month",
                "error_by_month.csv")
    # Individual months carry as few as 36 test samples, so the monthly table is
    # too noisy to draw conclusions from. Meteorological seasons give four groups
    # large enough to compare, and match how fire risk is discussed operationally.
    group_table([SEASON[mm] for mm in month], "Error analysis by season",
                "error_by_season.csv")

    # Burned-area bands contain only fires, so AUPRC is undefined within a band.
    # The meaningful quantity is the detection rate (recall) for fires of each
    # size, reported against the shared false-alarm rate on the negatives.
    bands = np.array([burned_area_band(b) for b in ba])
    fire = y == 1
    # false-alarm rate is per negative sample, not per test sample
    neg_fp = (float(((yhat == 1) & (~fire)).sum() / (~fire).sum())
              if (~fire).any() else float("nan"))
    rows = []
    for g in ["30-100 ha", "100-500 ha", "500-2000 ha", ">2000 ha"]:
        m = (bands == g) & fire
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({"burned_area_band": g, "n_fires": n,
                     "recall_at_threshold": round(float(yhat[m].mean()), 4),
                     "mean_predicted_score": round(float(p[m].mean()), 4)})
    rows.append({"burned_area_band": "no fire (negatives)",
                 "n_fires": int((~fire).sum()),
                 "recall_at_threshold": None,
                 "mean_predicted_score": round(float(p[~fire].mean()), 4)})
    with (out / "error_by_burned_area.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nError analysis by burned area ({em}, threshold={thr:.2f}, "
          f"false-alarm rate on negatives={neg_fp:.4f})")
    for r in rows:
        print(f"  {r['burned_area_band']:<22} n={r['n_fires']:<6} "
              f"recall={r['recall_at_threshold']} mean_score={r['mean_predicted_score']}")

    print(f"\nwritten figures + CSVs to {out}")


if __name__ == "__main__":
    main()
