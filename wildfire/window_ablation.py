"""
Window-length ablation: train the same model at several input history lengths and
plot AUPRC vs. window. Answers "why 30 days and not 1/2/3?" empirically.

Run:  python -m wildfire.window_ablation --config configs/gru.json \
          --windows 1 3 7 14 30 --out_dir runs/window_ablation
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gru.json")
    ap.add_argument("--windows", nargs="+", type=int, default=[1, 3, 7, 14, 30])
    ap.add_argument("--out_dir", default="runs/window_ablation")
    ap.add_argument("--epochs", type=int, default=None, help="override for a quick test")
    args = ap.parse_args()

    rows = []
    for w in args.windows:
        od = f"{args.out_dir}/w{w:02d}"
        cmd = [sys.executable, "-m", "wildfire.train", "--config", args.config,
               "--window", str(w), "--out_dir", od]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print(f"\n{'=' * 55}\nwindow = {w} days\n{'=' * 55}")
        subprocess.run(cmd, check=False)
        res = list(Path(od).rglob("results.json"))
        if res:
            r = json.loads(res[0].read_text())
            rows.append((w, r["val_auprc"], r["test"]["auprc"], r["test"]["f1"]))

    rows.sort()
    print(f"\n{'window (days)':>14}{'val AUPRC':>11}{'test AUPRC':>12}{'test F1':>9}")
    for w, v, ta, tf in rows:
        print(f"{w:>14}{v:>11.4f}{ta:>12.4f}{tf:>9.4f}")

    ws = [r[0] for r in rows]
    va = [r[1] for r in rows]
    te = [r[2] for r in rows]
    plt.figure(figsize=(6.2, 4.2))
    plt.plot(ws, va, "o-", color="#2a6fb0", label="Validation")
    plt.plot(ws, te, "s-", color="#e0761a", label="Test")
    plt.xlabel("Input history window (days)")
    plt.ylabel("AUPRC")
    plt.title("Effect of input window length on fire-danger AUPRC")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "window_ablation.png", dpi=200)
    (out / "window_ablation.json").write_text(json.dumps(rows, indent=1))
    print(f"\nsaved {out/'window_ablation.png'}")


if __name__ == "__main__":
    main()
