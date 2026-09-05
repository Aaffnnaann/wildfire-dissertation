"""
Paired bootstrap confidence intervals for differences in test AUPRC.

Two models evaluated on the same test set produce paired predictions, so the
difference in their scores can be assessed by resampling test samples (with
replacement) and recomputing both scores on each resample. This keeps the
pairing intact and accounts for the fact that both models see identical data.

Reported for each comparison:
  * observed difference in AUPRC (model A - model B)
  * percentile 95% confidence interval for that difference
  * a two-sided bootstrap p-value (proportion of resamples whose difference
    falls on the opposite side of zero, doubled and clipped at 1)

An interval excluding zero indicates the ordering is stable under resampling of
the test set. This quantifies test-set sampling uncertainty only; it does not
capture training-seed variability, which is measured separately by
`wildfire.multiseed`.

Run:
  python -m wildfire.stats_tests --runs runs/seeds \\
      --pairs hist_grad_boost:gru gru:fusion_cross hist_grad_boost:ensemble
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score


def load_preds(runs, name):
    """Load y/p for a run directory, tolerating a seed suffix."""
    direct = Path(runs) / name / "preds.npz"
    if direct.exists():
        d = np.load(direct)
        return d["y"], d["p"]
    cands = sorted(Path(runs).glob(f"{name}_seed*/preds.npz"))
    if not cands:
        raise FileNotFoundError(f"no predictions for '{name}' under {runs}")
    # average the per-seed probability vectors so the comparison uses the
    # seed-averaged model rather than one arbitrary run
    ds = [np.load(c) for c in cands]
    y = ds[0]["y"]
    if not all(np.array_equal(y, d["y"]) for d in ds):
        raise ValueError(f"label mismatch across seeds for '{name}'")
    return y, np.mean([d["p"] for d in ds], axis=0)


def paired_bootstrap(y, pa, pb, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    obs = average_precision_score(y, pa) - average_precision_score(y, pb)
    diffs = np.empty(n_boot)
    ok = 0
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ys = y[idx]
        if ys.min() == ys.max():        # degenerate resample, no positives
            continue
        diffs[ok] = (average_precision_score(ys, pa[idx])
                     - average_precision_score(ys, pb[idx]))
        ok += 1
    diffs = diffs[:ok]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value
    prop = min((diffs <= 0).mean(), (diffs >= 0).mean())
    p_value = min(1.0, 2 * prop)
    return {
        "observed_diff": float(obs),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_value": float(p_value),
        "n_bootstrap": int(ok),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/seeds")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="comparisons as A:B (difference computed as A - B)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = []
    for pair in args.pairs:
        a, b = pair.split(":")
        try:
            ya, pa = load_preds(args.runs, a)
            yb, pb = load_preds(args.runs, b)
        except (FileNotFoundError, ValueError) as e:
            print(f"[skip] {pair}: {e}")
            continue
        if not np.array_equal(ya, yb):
            print(f"[skip] {pair}: test labels differ between the two runs")
            continue
        r = paired_bootstrap(ya, pa, pb, args.n_boot, args.seed)
        r.update({"model_a": a, "model_b": b,
                  "auprc_a": float(average_precision_score(ya, pa)),
                  "auprc_b": float(average_precision_score(yb, pb))})
        results.append(r)
        print(f"\n{a} vs {b}")
        print(f"  AUPRC {r['auprc_a']:.4f} vs {r['auprc_b']:.4f}")
        print(f"  diff {r['observed_diff']:+.4f}  "
              f"95% CI [{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}]  "
              f"p={r['p_value']:.4f}  "
              f"{'excludes zero' if r['excludes_zero'] else 'includes zero'}")

    if results:
        out = Path(args.out) if args.out else Path(args.runs) / "bootstrap_tests.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["model_a", "model_b", "auprc_a", "auprc_b", "observed_diff",
                "ci95_low", "ci95_high", "p_value", "excludes_zero", "n_bootstrap"]
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in cols})
        (out.with_suffix(".json")).write_text(json.dumps(results, indent=2))
        print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
