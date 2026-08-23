"""
Feature ablation: how few variables do we actually need?

Ranks the 27 input variables by Random-Forest importance (aggregated across the
30 day-columns for each dynamic variable), then trains the best model (histogram
gradient boosting) on the top-k variables for k = 3, 5, 8, 13, 20, 27, and plots
test AUPRC vs. number of variables.

Run:  python -m wildfire.feature_ablation --out_dir figures
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wildfire.classical import load
from wildfire.data.convert import DYNAMIC, STATIC

N_DYN = len(DYNAMIC)   # 13
T = 30


def column_map():
    """variable name -> list of flattened-feature column indices."""
    m = {}
    for v, name in enumerate(DYNAMIC):
        m[name] = [d * N_DYN + v for d in range(T)]   # 30 day-columns per dynamic var
    base = N_DYN * T
    for s, name in enumerate(STATIC):
        m[name] = [base + s]                          # 1 column per static var
    return m


def hgb():
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="figures")
    ap.add_argument("--ks", nargs="+", type=int, default=[3, 5, 8, 13, 20, 27])
    args = ap.parse_args()

    Xtr, ytr, wtr = load("train")
    Xte, yte, _ = load("test")
    cmap = column_map()

    # --- rank variables by Random Forest importance (needs NaN imputation) ---
    col_mean = np.nanmean(Xtr, axis=0)
    Xtr_imp = np.where(np.isnan(Xtr), col_mean, Xtr)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                class_weight="balanced", random_state=0)
    rf.fit(Xtr_imp, ytr, sample_weight=wtr)
    imp = rf.feature_importances_
    var_imp = {name: float(imp[cols].sum()) for name, cols in cmap.items()}
    total = sum(var_imp.values())
    ranked = sorted(var_imp, key=var_imp.get, reverse=True)

    print("Variable importance (Random Forest, aggregated, normalised):")
    for i, name in enumerate(ranked, 1):
        print(f"{i:>2}. {name:<22}{100*var_imp[name]/total:>6.1f}%")

    # --- ablation: train HistGB on top-k variables ---
    rows = []
    for k in args.ks:
        chosen = ranked[:k]
        cols = sorted(c for name in chosen for c in cmap[name])
        m = hgb()
        m.fit(Xtr[:, cols], ytr, sample_weight=wtr)     # HistGB handles NaN natively
        p = m.predict_proba(Xte[:, cols])[:, 1]
        a = float(average_precision_score(yte, p))
        f = float(f1_score(yte, (p >= 0.5).astype(int)))
        rows.append((k, a, f))
        print(f"top-{k:<2} vars ({len(cols)} cols):  AUPRC {a:.4f}  F1 {f:.3f}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature_ablation.json").write_text(json.dumps(
        {"ranking": [(n, var_imp[n]) for n in ranked], "ablation": rows}, indent=1))

    ks = [r[0] for r in rows]
    aucs = [r[1] for r in rows]
    plt.rcParams.update({"font.family": "Segoe UI", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot(ks, aucs, "o-", color="#1b9e8a", linewidth=2, markersize=7)
    for x, y in zip(ks, aucs):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8, color="#146b5e")
    ax.set_xlabel("Number of variables (top-k by importance)", fontsize=11)
    ax.set_ylabel("Test AUPRC", fontsize=11)
    ax.set_title("Feature ablation: how few variables are needed?\n(gradient boosting)",
                 fontsize=11.5, fontweight="bold")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "feature_ablation.png", bbox_inches="tight")
    print(f"\nsaved {out/'feature_ablation.png'}")


if __name__ == "__main__":
    main()
