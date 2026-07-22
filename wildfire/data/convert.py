"""
Convert the Mesogeos Track A CSVs (long format: 30 rows per sample) into
train/val/test .npz tensors ready for PyTorch.

Split follows the Mesogeos paper exactly: train 2006-2019, val 2020,
test 2021-2022, by the year of the sample's last observed day.

Outputs (in data/processed/):
    {split}.npz with:
        dynamic : float32 [N, 30, 13]  daily driver variables (leakage-safe set)
        static  : float32 [N, 14]      per-cell constants
        y       : float32 [N]          1 = fire ignited on target day, 0 = no fire
        ba      : float32 [N]          final burned area (ha), 0 for negatives
        lon,lat : float32 [N]
        year    : int16   [N]
        doy     : int16   [N]          day-of-year of last observed day
    meta.json with feature names and train-split normalisation stats.

Run:  python -m wildfire.data.convert
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA_RAW = Path(r"C:\Users\Afnan\Desktop\Dissertation\data")
DATA_OUT = DATA_RAW / "processed"

T = 30  # days per sample window

# Inputs only — burned_areas / ignition_points are label machinery, and
# burned_area_has is the loss weight, so none of them are features.
DYNAMIC = ["d2m", "lai", "lst_day", "lst_night", "ndvi", "rh", "smi",
           "sp", "ssrd", "t2m", "tp", "wind_direction", "wind_speed"]
STATIC = ["aspect", "curvature", "dem", "roads_distance", "slope",
          "lc_agriculture", "lc_forest", "lc_grassland", "lc_settlement",
          "lc_shrubland", "lc_sparse_vegetation", "lc_water_bodies",
          "lc_wetland", "population"]


def load_split_arrays(csv_path: Path, label: int):
    df = pd.read_csv(csv_path, parse_dates=["time"])
    df = df.sort_values(["sample", "time_idx"])
    n = df["sample"].nunique()
    assert len(df) == n * T, f"{csv_path.name}: expected {n * T} rows, got {len(df)}"

    dynamic = df[DYNAMIC].to_numpy(np.float32).reshape(n, T, len(DYNAMIC))

    last = df[df.time_idx == T - 1]
    static = last[STATIC].to_numpy(np.float32)
    ba = last["burned_area_has"].to_numpy(np.float32) if label else np.zeros(n, np.float32)
    return {
        "dynamic": dynamic,
        "static": static,
        "y": np.full(n, label, np.float32),
        "ba": ba,
        "lon": last["x"].to_numpy(np.float32),
        "lat": last["y"].to_numpy(np.float32),
        "year": last["time"].dt.year.to_numpy(np.int16),
        "doy": last["time"].dt.dayofyear.to_numpy(np.int16),
    }


def main():
    DATA_OUT.mkdir(exist_ok=True)
    pos = load_split_arrays(DATA_RAW / "positives.csv", 1)
    neg = load_split_arrays(DATA_RAW / "negatives.csv", 0)
    full = {k: np.concatenate([pos[k], neg[k]]) for k in pos}

    splits = {
        "train": full["year"] <= 2019,
        "val": full["year"] == 2020,
        "test": full["year"] >= 2021,
    }

    nan_dyn = int(np.isnan(full["dynamic"]).sum())
    nan_sta = int(np.isnan(full["static"]).sum())

    for name, mask in splits.items():
        np.savez_compressed(DATA_OUT / f"{name}.npz",
                            **{k: v[mask] for k, v in full.items()})
        n = int(mask.sum())
        n_pos = int(full["y"][mask].sum())
        print(f"{name}: {n} samples ({n_pos} fire / {n - n_pos} no-fire)")

    # Normalisation stats from the training split only (no test leakage).
    tr_dyn = full["dynamic"][splits["train"]]
    tr_sta = full["static"][splits["train"]]
    meta = {
        "dynamic_features": DYNAMIC,
        "static_features": STATIC,
        "window": T,
        "dyn_mean": np.nanmean(tr_dyn, axis=(0, 1)).tolist(),
        "dyn_std": np.nanstd(tr_dyn, axis=(0, 1)).tolist(),
        "sta_mean": np.nanmean(tr_sta, axis=0).tolist(),
        "sta_std": np.nanstd(tr_sta, axis=0).tolist(),
        "nan_count_dynamic": nan_dyn,
        "nan_count_static": nan_sta,
    }
    (DATA_OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"NaNs — dynamic: {nan_dyn}, static: {nan_sta}")
    print(f"written to {DATA_OUT}")


if __name__ == "__main__":
    main()
