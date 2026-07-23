"""
Export a manifest CSV (one row per sample: split, label, lon, lat, last observed
date) used by the Colab patch-extraction notebook to cut satellite windows from
the mesogeos zarr cube.

Run:  python -m wildfire.data.export_manifest
"""
import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(r"C:\Users\Afnan\Desktop\Dissertation\data\processed")
OUT = Path(__file__).resolve().parents[2] / "manifest.csv"


def main():
    rows = []
    for split in ("train", "val", "test"):
        d = np.load(DATA / f"{split}.npz")
        dates = [dt.date(y, 1, 1) + dt.timedelta(days=int(doy) - 1)
                 for y, doy in zip(d["year"], d["doy"])]
        rows.append(pd.DataFrame({
            "split": split,
            "idx": np.arange(len(d["y"])),   # position within the split npz
            "y": d["y"].astype(int),
            "lon": d["lon"],
            "lat": d["lat"],
            "date": dates,                    # last observed day (t-1)
        }))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
