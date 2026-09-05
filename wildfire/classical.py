"""
Classical ML baseline — the bottom rung of the ladder (shows why deep learning
is needed). Flattens the sequence into a feature vector and fits tree ensembles.

Features per sample: 30 days x 13 dynamic  = 390, plus 14 static = 404.
Uses the same log-burned-area sample weighting as the deep models, and the same
chronological split, so numbers are directly comparable.

Run:  python -m wildfire.classical
"""
import argparse
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from wildfire.train import resolve_data_root

DEFAULT_DATA = r"C:\Users\Afnan\Desktop\Dissertation\data\processed"
DATA = Path(resolve_data_root(DEFAULT_DATA))


def load(split):
    d = np.load(DATA / f"{split}.npz")
    n = len(d["y"])
    X = np.concatenate([d["dynamic"].reshape(n, -1), d["static"]], axis=1)
    w = np.log1p(d["ba"].astype(np.float64))
    pos = d["y"] == 1
    w[~pos] = w[pos].min() if pos.any() else 1.0
    w = w / w.mean()
    return X.astype(np.float32), d["y"].astype(int), w.astype(np.float32)


def report(name, y, p):
    yhat = (p >= 0.5).astype(int)
    return {"model": name,
            "auprc": round(float(average_precision_score(y, p)), 4),
            "f1": round(float(f1_score(y, yhat)), 4),
            "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
            "recall": round(float(recall_score(y, yhat)), 4)}


def save_run(out_root, name, seed, y, p, y_val, p_val):
    """Save predictions and a results.json matching the format written by train.py.

    Validation predictions are stored alongside the test predictions so that any
    decision threshold can be chosen on validation data only.
    """
    d = Path(out_root) / name
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "preds.npz", y=y.astype(np.float32), p=p.astype(np.float32),
             y_val=y_val.astype(np.float32), p_val=p_val.astype(np.float32))
    test = report(name, y, p)
    (d / "results.json").write_text(json.dumps(
        {"config": {"model": name}, "seed": seed,
         "val_auprc": float(average_precision_score(y_val, p_val)),
         "test": {k: v for k, v in test.items() if k != "model"}}, indent=1))
    return test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None,
                    help="suffix for run directories, e.g. 'seed1' (keeps repeats separate)")
    ap.add_argument("--data", default=None,
                    help="override the data root, e.g. the count-matched sensitivity "
                         "variant produced by wildfire.paper_matched")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    if args.data:
        global DATA
        DATA = Path(resolve_data_root(args.data))
        print(f"[data] {DATA}")

    Xtr, ytr, wtr = load("train")
    Xva, yva, _ = load("val")
    Xte, yte, _ = load("test")
    print(f"features={Xtr.shape[1]} train={len(ytr)} val={len(yva)} test={len(yte)} seed={args.seed}")

    results = []

    # RandomForest can't take NaN -> mean-impute from train columns.
    col_mean = np.nanmean(Xtr, axis=0)
    imp = lambda X: np.where(np.isnan(X), col_mean, X)
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1,
                                class_weight="balanced", random_state=args.seed)
    rf.fit(imp(Xtr), ytr, sample_weight=wtr)
    results.append(save_run(args.out_dir, "random_forest" + suffix, args.seed,
                            yte, rf.predict_proba(imp(Xte))[:, 1],
                            yva, rf.predict_proba(imp(Xva))[:, 1]))

    # HistGradientBoosting handles NaN natively.
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=args.seed)
    hgb.fit(Xtr, ytr, sample_weight=wtr)
    results.append(save_run(args.out_dir, "hist_grad_boost" + suffix, args.seed,
                            yte, hgb.predict_proba(Xte)[:, 1],
                            yva, hgb.predict_proba(Xva)[:, 1]))

    print(f"\n{'model':<24}{'test_auprc':>12}{'test_f1':>10}")
    for r in sorted(results, key=lambda x: -x["auprc"]):
        print(f"{r['model']:<24}{r['auprc']:>12}{r['f1']:>10}")


if __name__ == "__main__":
    main()
