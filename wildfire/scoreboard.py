"""
Model-comparison scoreboard (the headline bar chart of the dissertation).

The figure used to have no generator in the repository, so it could not be
regenerated when a model was added. It is now built from a single CSV of the
results table, which keeps the figure and the table in step by construction.

The published GTN score is drawn as a reference line only. The project files
contain 194 more negatives than the count reported in the benchmark paper (see
wildfire.audit_dataset), so it is not a like-for-like comparison.

Run:  python -m wildfire.scoreboard
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGS = Path(r"C:\Users\Afnan\Desktop\Dissertation\figures")

FAMILY_COLOUR = {
    "ensemble": "#1b9e5a",
    "classical": "#d99b25",
    "weather": "#2a6fb0",
    "fusion": "#c0392b",
    "satellite": "#1b9e8a",
}
PUBLISHED_GTN = 0.858


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=str(FIGS / "model_comparison.csv"))
    ap.add_argument("--out", default=str(FIGS / "results_scoreboard.png"))
    ap.add_argument("--xmin", type=float, default=0.70)
    args = ap.parse_args()

    with open(args.table, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["test_auprc"]), reverse=True)

    labels = [r["label"] for r in rows]
    scores = [float(r["test_auprc"]) for r in rows]
    colours = [FAMILY_COLOUR[r["family"]] for r in rows]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    y = range(len(rows))
    ax.barh(list(y), scores, color=colours, height=0.72)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()

    for i, s in enumerate(scores):
        ax.text(s + 0.002, i, f"{s:.3f}", va="center", fontsize=6.5)

    ax.axvline(PUBLISHED_GTN, color="#666", linestyle="--", linewidth=1)
    # label above the first bar; the lower right corner belongs to the legend
    ax.text(PUBLISHED_GTN - 0.002, -0.75,
            f"published GTN ({PUBLISHED_GTN:.3f})", fontsize=6,
            color="#555", ha="right", va="center")

    ax.set_xlim(args.xmin, 0.90)
    ax.set_xlabel(f"Test AUPRC (axis starts at {args.xmin:.2f})", fontsize=8)
    ax.set_title("Model comparison on Mesogeos Track A", fontweight="bold", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOUR.values()]
    ax.legend(handles, list(FAMILY_COLOUR), fontsize=6, frameon=False,
              loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)
    print(f"written {args.out} ({len(rows)} models)")


if __name__ == "__main__":
    main()
