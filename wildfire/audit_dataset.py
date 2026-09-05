"""
Dataset provenance audit: reconcile the project CSV sample counts against the
counts reported in the Mesogeos paper (Kondylatos et al., 2023).

Paper reports: 25,722 total = 8,574 positives + 17,148 negatives, split as
train 19,353 / val 2,262 / test 4,107.

This script recomputes the counts from the raw CSVs, checks the per-sample row
structure, searches for duplicate (cell, target-date) samples, and reports the
difference by split and class. It writes a JSON report so the dissertation can
cite exact numbers rather than estimates.

Run:  python -m wildfire.audit_dataset
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_RAW = Path(r"C:\Users\Afnan\Desktop\Dissertation\data")
T = 30

# Counts as reported in the Mesogeos paper, Section "Experimental Setup".
PAPER = {
    "total": 25722, "positives": 8574, "negatives": 17148,
    "train": 19353, "val": 2262, "test": 4107,
    "train_pos": 6451, "train_neg": 12902,
    "val_pos": 754, "val_neg": 1508,
    "test_pos": 1369, "test_neg": 2738,
}


def load(csv_path: Path):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["sample", "time_idx"])
    return df


def summarise(df, name):
    n_samples = df["sample"].nunique()
    rows = len(df)
    per = df.groupby("sample").size()
    last = df[df.time_idx == T - 1].copy()
    return {
        "file": name,
        "rows": int(rows),
        "unique_samples": int(n_samples),
        "rows_per_sample_min": int(per.min()),
        "rows_per_sample_max": int(per.max()),
        "rows_per_sample_all_30": bool((per == T).all()),
        "last_day_rows": int(len(last)),
        "date_min": str(last["time"].min().date()),
        "date_max": str(last["time"].max().date()),
    }, last


def dup_analysis(last, name):
    """Duplicate (lon, lat, target-date) keys among the last-day rows."""
    key = last[["x", "y", "time"]].copy()
    key["x"] = key["x"].round(6)
    key["y"] = key["y"].round(6)
    dup_mask = key.duplicated(keep="first")
    n_dup = int(dup_mask.sum())
    # size of duplicate groups
    grp = key.groupby(["x", "y", "time"]).size()
    return {
        "file": name,
        "unique_cell_date_keys": int(len(grp)),
        "duplicate_rows_beyond_first": n_dup,
        "max_group_size": int(grp.max()),
        "groups_with_more_than_one": int((grp > 1).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA_RAW / "processed" / "dataset_audit.json"))
    args = ap.parse_args()

    report = {"paper_reported": PAPER}

    pos = load(DATA_RAW / "positives.csv")
    neg = load(DATA_RAW / "negatives.csv")
    s_pos, last_pos = summarise(pos, "positives.csv")
    s_neg, last_neg = summarise(neg, "negatives.csv")
    report["files"] = [s_pos, s_neg]

    n_pos, n_neg = s_pos["unique_samples"], s_neg["unique_samples"]
    report["project_counts"] = {
        "positives": n_pos, "negatives": n_neg, "total": n_pos + n_neg,
        "neg_per_pos_ratio": round(n_neg / n_pos, 4),
        "negatives_if_exactly_2x": 2 * n_pos,
        "negatives_excess_over_2x": n_neg - 2 * n_pos,
    }
    report["difference_vs_paper"] = {
        "total": (n_pos + n_neg) - PAPER["total"],
        "positives": n_pos - PAPER["positives"],
        "negatives": n_neg - PAPER["negatives"],
    }

    # duplicates
    report["duplicates"] = [dup_analysis(last_pos, "positives.csv"),
                            dup_analysis(last_neg, "negatives.csv")]

    # cross-file overlap: does any negative share a (cell, date) with a positive?
    kp = set(map(tuple, last_pos[["x", "y", "time"]].round(6).astype(str).to_numpy()))
    kn = list(map(tuple, last_neg[["x", "y", "time"]].round(6).astype(str).to_numpy()))
    report["pos_neg_key_overlap"] = int(sum(1 for k in kn if k in kp))

    # split-by-year counts on the project data
    def split_counts(last, label):
        yr = last["time"].dt.year
        return {
            "train_2006_2019": int((yr <= 2019).sum()),
            "val_2020": int((yr == 2020).sum()),
            "test_2021_2022": int((yr >= 2021).sum()),
            "label": label,
        }
    report["project_splits"] = [split_counts(last_pos, "positive"),
                                split_counts(last_neg, "negative")]

    # per-split difference vs paper
    p, n = report["project_splits"][0], report["project_splits"][1]
    report["split_difference_vs_paper"] = {
        "train": {"project": p["train_2006_2019"] + n["train_2006_2019"],
                  "paper": PAPER["train"],
                  "diff": p["train_2006_2019"] + n["train_2006_2019"] - PAPER["train"],
                  "pos_diff": p["train_2006_2019"] - PAPER["train_pos"],
                  "neg_diff": n["train_2006_2019"] - PAPER["train_neg"]},
        "val": {"project": p["val_2020"] + n["val_2020"], "paper": PAPER["val"],
                "diff": p["val_2020"] + n["val_2020"] - PAPER["val"],
                "pos_diff": p["val_2020"] - PAPER["val_pos"],
                "neg_diff": n["val_2020"] - PAPER["val_neg"]},
        "test": {"project": p["test_2021_2022"] + n["test_2021_2022"], "paper": PAPER["test"],
                 "diff": p["test_2021_2022"] + n["test_2021_2022"] - PAPER["test"],
                 "pos_diff": p["test_2021_2022"] - PAPER["test_pos"],
                 "neg_diff": n["test_2021_2022"] - PAPER["test_neg"]},
    }

    # burned-area / label sanity on positives
    ba = last_pos["burned_area_has"]
    report["positives_burned_area"] = {
        "min": float(ba.min()), "median": float(ba.median()), "max": float(ba.max()),
        "below_30_ha": int((ba < 30).sum()),
    }

    # column inventory
    report["columns"] = {
        "positives": list(pos.columns),
        "negatives_equal": bool(list(pos.columns) == list(neg.columns)),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nwritten:", out)


if __name__ == "__main__":
    main()
