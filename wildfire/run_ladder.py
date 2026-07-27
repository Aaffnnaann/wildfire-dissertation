"""
Run the full weather/time-series model ladder in sequence and print one
comparison table — the empirical justification for the transformer choice.

Run:  python -m wildfire.run_ladder
      python -m wildfire.run_ladder --out_dir /kaggle/working/runs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

LADDER = ["cnn1d", "rnn", "gru", "lstm", "bilstm", "temporal_transformer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    for name in LADDER:
        cmd = [sys.executable, "-m", "wildfire.train",
               "--config", f"configs/{name}.json", "--out_dir", args.out_dir]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
        subprocess.run(cmd, check=False)

    rows = []
    for name in LADDER:
        p = Path(args.out_dir) / name / "results.json"
        if p.exists():
            r = json.loads(p.read_text())
            rows.append((name, r["val_auprc"], r["test"]["auprc"], r["test"]["f1"]))
    rows.sort(key=lambda x: -x[2])

    print(f"\n{'=' * 60}\nWEATHER-BRANCH LADDER — ranked by test AUPRC\n{'=' * 60}")
    print(f"{'model':<22}{'val_auprc':>10}{'test_auprc':>12}{'test_f1':>10}")
    for name, v, ta, tf in rows:
        print(f"{name:<22}{v:>10.4f}{ta:>12.4f}{tf:>10.4f}")
    print("\nMesogeos paper baselines:  LSTM 0.853  Transformer 0.856  GTN 0.858")


if __name__ == "__main__":
    main()
