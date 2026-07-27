"""
Multimodal Dataset: pairs each Track A sample's weather series + static features
(from the processed npz) with its satellite window (from the patch shards written
by notebooks/extract_patches.ipynb).

Patch shards: {split}_{offset}.npz with
    patches : float32 [n, WIN, WIN, C]
    idx     : int     [n]   position within the split (0..N-1)

Standardisation stats for the patches are computed once from the TRAIN split
(NaN-aware) and reused for val/test — no leakage.
"""
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from wildfire.data.dataset import MesogeosDataset


def _assemble_patches(patch_root: Path, split: str, n_expected: int):
    shards = sorted(patch_root.glob(f"{split}_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no patch shards for '{split}' in {patch_root}")
    first = np.load(shards[0])["patches"]
    win, ch = first.shape[1], first.shape[3]
    out = np.full((n_expected, win, win, ch), np.nan, np.float32)
    seen = 0
    for s in shards:
        d = np.load(s)
        out[d["idx"]] = d["patches"]
        seen += len(d["idx"])
    if seen != n_expected:
        print(f"[warn] {split}: assembled {seen} patches, expected {n_expected}")
    return out


class MultimodalDataset(Dataset):
    def __init__(self, root, patch_root, split, window=30, patch_stats=None):
        self.temporal = MesogeosDataset(root, split, window)
        n = len(self.temporal)
        patches = _assemble_patches(Path(patch_root), split, n)

        if patch_stats is None:                      # compute from this split (train)
            mean = np.nanmean(patches, axis=(0, 1, 2))
            std = np.nanstd(patches, axis=(0, 1, 2))
            self.stats = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            mean = np.array(patch_stats["mean"], np.float32)
            std = np.array(patch_stats["std"], np.float32)
            self.stats = patch_stats

        patches = (patches - mean) / np.maximum(std, 1e-6)
        self.patches = np.nan_to_num(patches, nan=0.0).astype(np.float32)

    def __len__(self):
        return len(self.temporal)

    def __getitem__(self, i):
        item = self.temporal[i]
        item["patch"] = torch.from_numpy(self.patches[i])
        return item

    def save_stats(self, path):
        Path(path).write_text(json.dumps(self.stats, indent=1))
