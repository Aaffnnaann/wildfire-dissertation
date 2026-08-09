# Wildfire Dissertation — Project Guide

Shared context for AI coding assistants (Codex reads this file; Claude Code reads it via `CLAUDE.md`).

## What this is
MSc dissertation code — Afnan Faiyazahmed Darga, University of Glasgow.
**Multimodal deep learning for next-day wildfire risk prediction** on the
**Mesogeos Track A** benchmark. A dual-branch cross-attention transformer:
a **ViT** over satellite windows + a **temporal transformer** over weather
time series, fused for next-day fire-danger forecasting.

## Environment — READ FIRST
- OS: **Windows 11**, shell is **PowerShell**.
- **ML code runs in conda env `pytorch_env`** (Python + torch 2.5.1, **CPU-only**).
- The system `python` (C:\Python313) has **NO torch** — never use it for training/eval.
- Run project code with the env's interpreter directly (no activation needed):
  ```
  C:\Users\Afnan\anaconda3\envs\pytorch_env\python.exe -m wildfire.train --config configs/lstm.json
  ```
  or activate first: `conda activate pytorch_env`
- Because it's CPU-only, keep smoke tests small (`--limit`, few `--epochs`). Heavy
  training is done on Kaggle GPU (see `notebooks/train_kaggle.ipynb`).

## Layout
```
wildfire/
  data/convert.py         CSV -> npz (paper-exact split: train 2006-19 / val 2020 / test 2021-22)
  data/dataset.py         PyTorch Dataset (standardisation, log-BA loss weights)
  data/multimodal.py      multimodal (satellite + weather) dataset
  data/export_manifest.py manifest export
  models/lstm.py                  Model 0: LSTM baseline replication
  models/sequence_baselines.py    CNN1D / RNN / GRU / BiLSTM ladder
  models/temporal_transformer.py  Model B: weather branch
  models/vit.py                   ViT satellite encoder
  models/fusion.py                dual-branch cross-attention fusion (Models A/C/D)
  train.py                shared training loop (AdamW, cosine LR, early stop on val AUPRC)
  run_ladder.py           runs the progressive model ladder
configs/                  one JSON per experiment
runs/                     checkpoints + results (gitignored)
tests/smoke_models.py     fast model sanity checks
notebooks/                Kaggle GPU training + patch extraction
```
Data lives one level up in `../data/` (`positives.csv`, `negatives.csv`,
`norms.json`, `vars_dict.json`, `processed/`).

## Common commands
```
# convert raw CSVs to npz (needs ../data/positives.csv etc.)
python -m wildfire.data.convert

# train one experiment
python -m wildfire.train --config configs/lstm.json

# fast smoke test
python -m wildfire.train --config configs/lstm.json --epochs 2 --limit 2000

# run the full model ladder
python -m wildfire.run_ladder
```
(Prefix `python` with the `pytorch_env` interpreter path above.)

## What "better" means here — the target
Beat the Mesogeos baselines (Kondylatos et al. 2023). **Primary metric: val/test AUPRC.**

| Model | F1 | AUPRC |
|---|---|---|
| LSTM | 0.786 | 0.853 |
| Transformer | 0.780 | 0.856 |
| GTN | 0.786 | 0.858 |

**Target: AUPRC ≥ 0.86.** Early stopping selects on validation AUPRC.

## Conventions
- One JSON config per experiment in `configs/`; add a new file rather than editing a baseline.
- Keep the train/val/test split **paper-exact** (2006–19 / 2020 / 2021–22). Don't leak.
- Don't commit `runs/` (checkpoints/results) or data — they're gitignored.
- Keep changes reproducible: set seeds, log config + metrics with each run.
- Match existing code style; keep new models under `wildfire/models/` and register them the same way as the ladder.

## Repo
`origin` → https://github.com/Aaffnnaann/wildfire-dissertation.git
Commit/push only when asked. Branch off `main` for non-trivial changes.
