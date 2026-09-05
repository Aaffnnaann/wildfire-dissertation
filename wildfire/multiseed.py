"""
Repeated-runs harness: train each model under several random seeds and
summarise the spread of results.

Single-seed numbers cannot distinguish a real difference between models from
run-to-run variation. This script repeats each configuration across seeds,
records every individual run, and reports mean, standard deviation and a 95%
confidence interval for the mean (Student t, n-1 degrees of freedom).

The ensemble is recomputed per seed from that seed's member predictions, so its
variability is measured on the same footing as the individual models.

Examples
--------
  # neural + classical, 5 seeds (full run; use a GPU for the neural models)
  python -m wildfire.multiseed --models hist_grad_boost gru lstm temporal_transformer \\
      --seeds 0 1 2 3 4 --out_dir runs/seeds

  # tiny local smoke test
  python -m wildfire.multiseed --models gru --seeds 0 1 --epochs 2 --limit 1500 \\
      --out_dir runs/smoke
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

# models produced by wildfire.classical rather than wildfire.train
CLASSICAL = {"hist_grad_boost", "random_forest"}
# default ensemble membership (must all be present for a seed to be ensembled)
ENSEMBLE_MEMBERS = ["hist_grad_boost", "gru", "cnn1d", "fusion_cross"]

T_CRIT = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
          8: 2.365, 9: 2.306, 10: 2.262}


def ci95(values):
    """Mean and 95% CI half-width for the mean. Returns (mean, sd, half_width)."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    mean = float(v.mean())
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = float(v.std(ddof=1))
    t = T_CRIT.get(n, 1.96)
    return mean, sd, t * sd / math.sqrt(n)


def run_one(model, seed, out_dir, epochs, limit, python_exe):
    tag = f"seed{seed}"
    if model in CLASSICAL:
        cmd = [python_exe, "-m", "wildfire.classical",
               "--seed", str(seed), "--tag", tag, "--out_dir", out_dir]
    else:
        cfg = Path("configs") / f"{model}.json"
        if not cfg.exists():
            print(f"[skip] no config for '{model}' at {cfg}")
            return
        cmd = [python_exe, "-m", "wildfire.train", "--config", str(cfg),
               "--seed", str(seed), "--tag", tag, "--out_dir", out_dir]
        if epochs:
            cmd += ["--epochs", str(epochs)]
        if limit:
            cmd += ["--limit", str(limit)]
    print(f"\n=== {model} seed {seed} ===")
    subprocess.run(cmd, check=False)


def collect(out_dir, models, seeds):
    """Read results.json from every completed run directory."""
    rows = []
    for m in models:
        for s in seeds:
            rj = Path(out_dir) / f"{m}_seed{s}" / "results.json"
            if not rj.exists():
                continue
            r = json.loads(rj.read_text())
            rows.append({"model": m, "seed": s,
                         "val_auprc": r.get("val_auprc"),
                         "test_auprc": r["test"]["auprc"],
                         "test_f1": r["test"]["f1"],
                         "test_precision": r["test"].get("precision"),
                         "test_recall": r["test"].get("recall")})
    return rows


def ensemble_rows(out_dir, members, seeds):
    """Recompute the probability-averaging ensemble separately for each seed."""
    rows = []
    for s in seeds:
        paths = [Path(out_dir) / f"{m}_seed{s}" / "preds.npz" for m in members]
        if not all(p.exists() for p in paths):
            continue
        ds = [np.load(p) for p in paths]
        y = ds[0]["y"]
        if not all(np.array_equal(y, d["y"]) for d in ds):
            print(f"[warn] label mismatch across members at seed {s}; skipped")
            continue
        p = np.mean([d["p"] for d in ds], axis=0)
        row = {"model": "ensemble", "seed": s,
               "test_auprc": float(average_precision_score(y, p)),
               "test_f1": float(f1_score(y, (p >= 0.5).astype(int))),
               "test_precision": None, "test_recall": None}
        dest = Path(out_dir) / f"ensemble_seed{s}"
        dest.mkdir(parents=True, exist_ok=True)
        if all("p_val" in d.files for d in ds):
            yv = ds[0]["y_val"]
            pv = np.mean([d["p_val"] for d in ds], axis=0)
            row["val_auprc"] = float(average_precision_score(yv, pv))
            np.savez(dest / "preds.npz", y=y, p=p, y_val=yv, p_val=pv)
        else:
            row["val_auprc"] = None
            np.savez(dest / "preds.npz", y=y, p=p)
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out_dir", default="runs/seeds")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--limit", type=int, default=None, help="subsample train (smoke tests)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--collect_only", action="store_true",
                    help="skip training and only aggregate existing runs")
    ap.add_argument("--ensemble", nargs="*", default=None,
                    help=f"ensemble members (default: {' '.join(ENSEMBLE_MEMBERS)})")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    if not args.collect_only:
        for m in args.models:
            for s in args.seeds:
                run_one(m, s, args.out_dir, args.epochs, args.limit, args.python)

    rows = collect(args.out_dir, args.models, args.seeds)
    members = args.ensemble if args.ensemble is not None else ENSEMBLE_MEMBERS
    if members:
        rows += ensemble_rows(args.out_dir, members, args.seeds)

    if not rows:
        print("no completed runs found — nothing to aggregate")
        return

    per_run = Path(args.out_dir) / "per_seed_metrics.csv"
    with per_run.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "seed", "val_auprc", "test_auprc",
                                          "test_f1", "test_precision", "test_recall"])
        w.writeheader()
        w.writerows(rows)

    summary = []
    for m in dict.fromkeys(r["model"] for r in rows):
        vals = [r["test_auprc"] for r in rows if r["model"] == m]
        f1s = [r["test_f1"] for r in rows if r["model"] == m]
        mean, sd, hw = ci95(vals)
        f_mean, f_sd, _ = ci95(f1s)
        summary.append({
            "model": m, "n_seeds": len(vals),
            "auprc_mean": round(mean, 4),
            "auprc_sd": None if math.isnan(sd) else round(sd, 4),
            "auprc_ci95_low": None if math.isnan(hw) else round(mean - hw, 4),
            "auprc_ci95_high": None if math.isnan(hw) else round(mean + hw, 4),
            "f1_mean": round(f_mean, 4),
            "f1_sd": None if math.isnan(f_sd) else round(f_sd, 4),
        })
    summary.sort(key=lambda r: -r["auprc_mean"])

    sm = Path(args.out_dir) / "seed_summary.csv"
    with sm.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\n{'model':<24}{'n':>3}{'AUPRC mean':>12}{'sd':>8}{'95% CI':>20}")
    for r in summary:
        ci = ("--" if r["auprc_ci95_low"] is None
              else f"[{r['auprc_ci95_low']:.4f}, {r['auprc_ci95_high']:.4f}]")
        sd = "--" if r["auprc_sd"] is None else f"{r['auprc_sd']:.4f}"
        print(f"{r['model']:<24}{r['n_seeds']:>3}{r['auprc_mean']:>12.4f}{sd:>8}{ci:>20}")
    print(f"\nwritten: {per_run}\n         {sm}")


if __name__ == "__main__":
    main()
