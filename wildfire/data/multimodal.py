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
    # Preferred: the single assembled file the extraction notebook produces.
    single = patch_root / f"{split}.npz"
    if single.exists():
        d = np.load(single)
        patches, idx = d["patches"], d["idx"]
        if len(patches) == n_expected and np.array_equal(idx, np.arange(n_expected)):
            return patches
        out = np.full((n_expected, *patches.shape[1:]), np.nan, np.float32)
        out[idx] = patches
        return out
    # Fallback: raw date-batch shards (each carries an idx + split column).
    shards = sorted(patch_root.glob("shard_*.npz")) + sorted(patch_root.glob(f"{split}_*.npz"))
    if not shards:
        raise FileNotFoundError(f"no patches ({split}.npz or shards) in {patch_root}")
    win = ch = None
    out = None
    seen = 0
    for s in shards:
        d = np.load(s, allow_pickle=True)
        if "split" in d.files:                       # shard_*.npz carries all splits
            m = d["split"].astype(str) == split
            if not m.any():
                continue
            idx, pat = d["idx"][m], d["patches"][m]
        else:
            idx, pat = d["idx"], d["patches"]
        if out is None:
            win, ch = pat.shape[1], pat.shape[3]
            out = np.full((n_expected, win, win, ch), np.nan, np.float32)
        out[idx] = pat
        seen += len(idx)
    if seen != n_expected:
        print(f"[warn] {split}: assembled {seen} patches, expected {n_expected}")
    return out


class MultimodalDataset(Dataset):
    def __init__(self, root, patch_root, split, window=30, patch_stats=None,
                 add_mask=False, augment=False):
        self.temporal = MesogeosDataset(root, split, window)
        self.augment = augment
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

        patches = (patches - mean) / np.maximum(std, 1e-6)   # NaNs stay NaN here
        mask = np.isfinite(patches).all(axis=-1, keepdims=True).astype(np.float32)
        patches = np.nan_to_num(patches, nan=0.0).astype(np.float32)
        if add_mask:                                  # append a valid/missing channel
            patches = np.concatenate([patches, mask], axis=-1)
        self.patches = patches

    def __len__(self):
        return len(self.temporal)

    def _aug(self, p):                                # p: [H, W, C]
        if np.random.rand() < 0.5:
            p = p[:, ::-1]
        if np.random.rand() < 0.5:
            p = p[::-1, :]
        k = np.random.randint(4)
        if k:
            p = np.rot90(p, k)
        return np.ascontiguousarray(p)

    def __getitem__(self, i):
        item = self.temporal[i]
        p = self.patches[i]
        if self.augment:
            p = self._aug(p)
        item["patch"] = torch.from_numpy(p)
        return item

    def save_stats(self, path):
        Path(path).write_text(json.dumps(self.stats, indent=1))
