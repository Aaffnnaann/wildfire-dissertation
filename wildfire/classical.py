"""
Classical ML baseline — the bottom rung of the ladder (shows why deep learning
is needed). Flattens the sequence into a feature vector and fits tree ensembles.

Features per sample: 30 days x 13 dynamic  = 390, plus 14 static = 404.
Uses the same log-burned-area sample weighting as the deep models, and the same
chronological split, so numbers are directly comparable.

Run:  python -m wildfire.classical
"""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

DATA = Path(r"C:\Users\Afnan\Desktop\Dissertation\data\processed")


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


def main():
    Xtr, ytr, wtr = load("train")
    Xte, yte, _ = load("test")
    print(f"features={Xtr.shape[1]} train={len(ytr)} test={len(yte)}")

    results = []

    # RandomForest can't take NaN -> mean-impute from train columns.
    col_mean = np.nanmean(Xtr, axis=0)
    Xtr_imp = np.where(np.isnan(Xtr), col_mean, Xtr)
    Xte_imp = np.where(np.isnan(Xte), col_mean, Xte)
    rf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1,
                                class_weight="balanced", random_state=0)
    rf.fit(Xtr_imp, ytr, sample_weight=wtr)
    results.append(report("random_forest", yte, rf.predict_proba(Xte_imp)[:, 1]))

    # HistGradientBoosting handles NaN natively.
    hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                         l2_regularization=1.0, random_state=0)
    hgb.fit(Xtr, ytr, sample_weight=wtr)
    results.append(report("hist_grad_boost", yte, hgb.predict_proba(Xte)[:, 1]))

    print(f"\n{'model':<20}{'test_auprc':>12}{'test_f1':>10}")
    for r in sorted(results, key=lambda x: -x["auprc"]):
        print(f"{r['model']:<20}{r['auprc']:>12}{r['f1']:>10}")
    print("\nDL comparison: GRU 0.864  LSTM 0.860  Transformer 0.843  (paper GTN 0.858)")

    out = Path(__file__).resolve().parents[1] / "runs" / "classical"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
