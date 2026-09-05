"""
Count-matched sensitivity dataset.

The project CSV contains 194 more negative samples than the count reported in the
Mesogeos paper (17,342 vs 17,148); positives are identical (8,574). The audit in
`wildfire.audit_dataset` shows the extra rows are distinct (cell, date) samples,
not duplicates, so the published subset cannot be recovered exactly: the paper's
negative draw is stochastic and neither its seed nor its sample identifiers are
published.

What this script CAN do is build a *count-matched* variant: a deterministic,
seeded subsample of negatives that reproduces the paper's per-split totals
exactly (train 12,902 / val 1,508 / test 2,738 negatives). Training on it tests
whether the 194 extra negatives materially affect any conclusion. It is a
sensitivity check, not a reproduction of the published samples, and results on it
are not directly comparable to the main results because the test set changes.

Run:  python -m wildfire.paper_matched --seed 0
"""
import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path(r"C:\Users\Afnan\Desktop\Dissertation\data\processed")

# negative counts per split as reported in the Mesogeos paper
PAPER_NEG = {"train": 12902, "val": 1508, "test": 2738}
PAPER_POS = {"train": 6451, "val": 754, "test": 1369}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA))
    ap.add_argument("--dst", default=str(DATA.parent / "processed_paper_matched"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {"seed": args.seed, "source": str(src), "splits": {}}

    for split in ("train", "val", "test"):
        d = dict(np.load(src / f"{split}.npz"))
        y = d["y"]
        pos_idx = np.flatnonzero(y == 1)
        neg_idx = np.flatnonzero(y == 0)
        want = PAPER_NEG[split]

        if len(pos_idx) != PAPER_POS[split]:
            print(f"[warn] {split}: {len(pos_idx)} positives, paper reports "
                  f"{PAPER_POS[split]} — positives left untouched")
        if len(neg_idx) < want:
            raise SystemExit(f"{split}: only {len(neg_idx)} negatives, need {want}")

        keep_neg = rng.choice(neg_idx, size=want, replace=False)
        keep = np.sort(np.concatenate([pos_idx, keep_neg]))
        out = {k: v[keep] for k, v in d.items()}
        np.savez_compressed(dst / f"{split}.npz", **out)

        report["splits"][split] = {
            "positives": int(len(pos_idx)),
            "negatives_before": int(len(neg_idx)),
            "negatives_after": int(want),
            "negatives_dropped": int(len(neg_idx) - want),
            "total_after": int(len(keep)),
        }
        print(f"{split}: kept {len(keep)} = {len(pos_idx)} pos + {want} neg "
              f"(dropped {len(neg_idx) - want} neg)")

    # normalisation stats must be recomputed from this variant's training split
    meta = json.loads((src / "meta.json").read_text())
    tr = np.load(dst / "train.npz")
    meta["dyn_mean"] = np.nanmean(tr["dynamic"], axis=(0, 1)).tolist()
    meta["dyn_std"] = np.nanstd(tr["dynamic"], axis=(0, 1)).tolist()
    meta["sta_mean"] = np.nanmean(tr["static"], axis=0).tolist()
    meta["sta_std"] = np.nanstd(tr["static"], axis=0).tolist()
    meta["provenance"] = ("count-matched sensitivity variant: negatives subsampled "
                          f"to the paper's per-split totals with seed {args.seed}; "
                          "NOT the published sample set")
    (dst / "meta.json").write_text(json.dumps(meta, indent=1))
    (dst / "count_match_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {dst}")


if __name__ == "__main__":
    main()
