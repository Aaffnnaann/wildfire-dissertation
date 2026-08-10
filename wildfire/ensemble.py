"""
Ensemble the saved test predictions of several models by averaging probabilities.

Each model writes runs/<name>/preds.npz (keys: y, p) during its run. This loads
the listed models, checks their labels align, averages their probabilities, and
reports the ensemble AUPRC/F1 alongside each member.

Run:  python -m wildfire.ensemble --runs runs --models gru fusion_cross hist_grad_boost
"""
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score


def load(runs, name):
    d = np.load(Path(runs) / name / "preds.npz")
    return d["y"], d["p"]


def score(y, p):
    return average_precision_score(y, p), f1_score(y, (p >= 0.5).astype(int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, default=None,
                    help="optional per-model weights (default: equal)")
    args = ap.parse_args()

    y0, preds, present = None, [], []
    for m in args.models:
        try:
            y, p = load(args.runs, m)
        except FileNotFoundError:
            print(f"[skip] no preds for '{m}' (did it run with --out_dir {args.runs}?)")
            continue
        if y0 is None:
            y0 = y
        elif not np.array_equal(y0, y):
            raise SystemExit(f"label mismatch for '{m}' — different test set/order")
        preds.append(p)
        present.append(m)

    if len(preds) < 2:
        raise SystemExit("need at least 2 models with saved preds to ensemble")

    w = np.array(args.weights, float) if args.weights else np.ones(len(preds))
    w = w[:len(preds)] / w[:len(preds)].sum()
    ens = np.average(np.stack(preds), axis=0, weights=w)

    print(f"{'model':<22}{'AUPRC':>9}{'F1':>8}")
    for m, p in zip(present, preds):
        a, f = score(y0, p)
        print(f"{m:<22}{a:>9.4f}{f:>8.4f}")
    a, f = score(y0, ens)
    print("-" * 39)
    print(f"{'ENSEMBLE (' + str(len(preds)) + ')':<22}{a:>9.4f}{f:>8.4f}")
    print("\nbaselines: best single 0.8739 (HistGB) · published GTN 0.858")


if __name__ == "__main__":
    main()
