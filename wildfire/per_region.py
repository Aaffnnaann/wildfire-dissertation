"""
Per-region performance breakdown: does the (basin-wide) model do better in some
parts of the Mediterranean? Answers "could we focus on a smaller area?".

Trains the best single model (gradient boosting) on the full basin, then reports
test AUPRC per geographic sub-region using each sample's lon/lat. No retraining
per region — this evaluates the one full model, region by region.

Run:  python -m wildfire.per_region --out_dir figures
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire.classical import load, DATA

# (name, lon_min, lon_max, lat_min, lat_max) — approximate Mediterranean sub-regions
REGIONS = [
    ("Iberia",              -10,  3, 36, 44),
    ("Maghreb (NW Africa)", -10, 12, 28, 37),
    ("Italy & Central",       3, 19, 37, 47),
    ("Balkans & Greece",     19, 28, 36, 47),
    ("E. Med (Turkey/Levant)",28, 45, 28, 43),
]


def assign_region(lon, lat):
    for name, x0, x1, y0, y1 in REGIONS:
        if x0 <= lon < x1 and y0 <= lat < y1:
            return name
    return "Other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    Xtr, ytr, wtr = load("train")
    Xte, yte, _ = load("test")
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)
    hgb.fit(Xtr, ytr, sample_weight=wtr)
    p = hgb.predict_proba(Xte)[:, 1]

    d = np.load(DATA / "test.npz")
    lon, lat = d["lon"], d["lat"]
    region = np.array([assign_region(a, b) for a, b in zip(lon, lat)])

    print(f"overall test AUPRC {average_precision_score(yte, p):.4f}\n")
    # 'base' = fire prevalence = AUPRC of a random classifier; 'lift' = AUPRC/base
    # (skill above chance). Comparing raw AUPRC across regions is confounded by base.
    print(f"{'region':<26}{'samples':>8}{'fires':>7}{'fire%':>7}{'AUPRC':>9}{'base':>7}{'lift':>7}")
    rows = []
    for name, *_ in REGIONS + [("Other", 0, 0, 0, 0)]:
        m = region == name
        n = int(m.sum())
        if n == 0:
            continue
        nf = int(yte[m].sum())
        base = nf / n
        if nf >= 10 and nf < n:                       # enough positives to score
            a = float(average_precision_score(yte[m], p[m]))
            f = float(f1_score(yte[m], (p[m] >= 0.5).astype(int)))
            lift = a / base
        else:
            a = f = lift = float("nan")
        rows.append((name, n, nf, a, f, base, lift))
        print(f"{name:<26}{n:>8}{nf:>7}{100*base:>6.1f}%{a:>9.4f}{base:>7.3f}{lift:>7.2f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_region.json").write_text(json.dumps(rows, indent=1))

    # ---- figure: map of test points (left) + per-region AUPRC bars (right) ----
    plt.rcParams.update({"font.family": "Segoe UI", "figure.dpi": 200})
    fig, (axm, axb) = plt.subplots(1, 2, figsize=(13, 4.6),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    palette = ["#2a6fb0", "#e0761a", "#1b9e8a", "#c0392b", "#7b5ea7", "#8a8c90"]
    names = [r[0] for r in rows]
    cmap = {n: palette[i % len(palette)] for i, n in enumerate(names)}
    for n in names:
        m = region == n
        axm.scatter(lon[m], lat[m], s=4, c=cmap[n], alpha=0.5, linewidths=0, label=n)
    axm.set_title("Test samples by region", fontweight="bold", fontsize=11)
    axm.set_xlabel("Longitude"); axm.set_ylabel("Latitude")
    axm.legend(frameon=False, markerscale=3, fontsize=8, loc="lower left")
    axm.grid(alpha=0.25)

    valid = [r for r in rows if not np.isnan(r[3])]
    valid.sort(key=lambda r: r[3])
    axb.barh([r[0] for r in valid], [r[3] for r in valid],
             color=[cmap[r[0]] for r in valid], label="Model AUPRC")
    for i, r in enumerate(valid):
        axb.text(r[3] - 0.01, i, f"{r[3]:.3f}", va="center", ha="right",
                 color="white", fontsize=8.5, fontweight="bold")
    # baseline (fire prevalence) = AUPRC of a random classifier — shows the confound
    axb.scatter([r[5] for r in valid], range(len(valid)), color="#222", s=32, zorder=3,
                label="Baseline (fire rate)")
    axb.set_title("Test AUPRC by region\n(bar = model, dot = chance baseline)",
                  fontweight="bold", fontsize=11)
    axb.set_xlabel("AUPRC"); axb.set_xlim(0.1, 1.0)
    axb.legend(frameon=False, fontsize=8, loc="lower right")
    axb.spines["top"].set_visible(False); axb.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "per_region.png", bbox_inches="tight")
    print(f"\nsaved {out/'per_region.png'}")


if __name__ == "__main__":
    main()
