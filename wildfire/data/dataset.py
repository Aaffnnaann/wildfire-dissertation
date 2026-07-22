"""PyTorch Dataset over the processed Mesogeos Track A npz files."""
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class MesogeosDataset(Dataset):
    """
    One item = one (cell, target-day) sample.

    Returns a dict:
        dynamic : [window, 13]  standardised daily drivers (last `window` days)
        static  : [14]          standardised per-cell constants
        y       : []            binary label
        w       : []            loss weight (log-scaled burned area, paper protocol)
    """

    def __init__(self, root, split, window=30):
        root = Path(root)
        data = np.load(root / f"{split}.npz")
        meta = json.loads((root / "meta.json").read_text())
        assert 1 <= window <= meta["window"]

        dyn = data["dynamic"][:, -window:, :]
        mean = np.array(meta["dyn_mean"], np.float32)
        std = np.array(meta["dyn_std"], np.float32)
        dyn = (dyn - mean) / np.maximum(std, 1e-6)
        self.dynamic = np.nan_to_num(dyn, nan=0.0)  # mean-impute after standardising

        sta = (data["static"] - np.array(meta["sta_mean"], np.float32)) / \
            np.maximum(np.array(meta["sta_std"], np.float32), 1e-6)
        self.static = np.nan_to_num(sta, nan=0.0)

        self.y = data["y"]

        # Loss weighting per the Mesogeos paper: positives weighted by
        # log-scaled burned area; negatives get the minimum positive weight.
        w = np.log1p(data["ba"].astype(np.float64))
        pos = self.y == 1
        w_min = w[pos].min() if pos.any() else 1.0
        w[~pos] = w_min
        w = w / w.mean()  # normalise so the effective learning rate is unchanged
        self.w = w.astype(np.float32)

        self.meta = meta

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {
            "dynamic": torch.from_numpy(self.dynamic[i]),
            "static": torch.from_numpy(self.static[i]),
            "y": torch.tensor(self.y[i]),
            "w": torch.tensor(self.w[i]),
        }
