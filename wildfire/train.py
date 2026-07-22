"""
Training loop for Mesogeos Track A models.

Run:  python -m wildfire.train --config configs/lstm.json
      python -m wildfire.train --config configs/lstm.json --epochs 2 --limit 2000   (smoke test)
"""
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from wildfire.data.dataset import MesogeosDataset
from wildfire.models.lstm import LSTMBaseline
from wildfire.models.temporal_transformer import TemporalTransformer

MODELS = {"lstm": LSTMBaseline, "temporal_transformer": TemporalTransformer}


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            p = torch.sigmoid(model(batch))
            ys.append(batch["y"].cpu().numpy())
            ps.append(p.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    yhat = (p >= 0.5).astype(int)
    return {
        "auprc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="subsample train set (smoke test)")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.epochs:
        cfg["epochs"] = args.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.get("seed", 0))

    root = cfg["data_root"]
    window = cfg.get("window", 30)
    train_ds = MesogeosDataset(root, "train", window)
    val_ds = MesogeosDataset(root, "val", window)
    test_ds = MesogeosDataset(root, "test", window)
    if args.limit:
        idx = np.random.RandomState(0).permutation(len(train_ds))[:args.limit]
        train_ds = Subset(train_ds, idx.tolist())

    bs = cfg.get("batch_size", 256)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=512)
    test_dl = DataLoader(test_ds, batch_size=512)

    model = MODELS[cfg["model"]](**cfg.get("model_args", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={cfg['model']} params={n_params:,} device={device} "
          f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 1e-3),
                            weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])

    out_dir = Path(cfg.get("out_dir", "runs")) / cfg["model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    best_auprc, best_state, patience_left = -1.0, None, cfg.get("patience", 8)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logit = model(batch)
            loss = (F.binary_cross_entropy_with_logits(
                logit, batch["y"], reduction="none") * batch["w"]).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()

        val = evaluate(model, val_dl, device)
        marker = ""
        if val["auprc"] > best_auprc:
            best_auprc = val["auprc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.get("patience", 8)
            marker = " *"
        else:
            patience_left -= 1
        print(f"epoch {epoch:02d} loss={np.mean(losses):.4f} "
              f"val_auprc={val['auprc']:.4f} val_f1={val['f1']:.4f} "
              f"({time.time() - t0:.0f}s){marker}")
        if patience_left == 0:
            print("early stop")
            break

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    test = evaluate(model, test_dl, device)
    print(f"TEST  auprc={test['auprc']:.4f} f1={test['f1']:.4f} "
          f"precision={test['precision']:.4f} recall={test['recall']:.4f}")
    (out_dir / "results.json").write_text(json.dumps(
        {"config": cfg, "val_auprc": best_auprc, "test": test}, indent=1))


if __name__ == "__main__":
    main()
