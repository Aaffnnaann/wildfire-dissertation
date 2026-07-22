# Multimodal Deep Learning for Wildfire Risk Prediction

MSc dissertation code — Afnan Faiyazahmed Darga, University of Glasgow.
Dual-branch cross-attention transformer (ViT over satellite windows +
temporal transformer over weather series) for next-day fire danger
forecasting on the Mesogeos Track A benchmark.

## Layout

```
wildfire/
  data/convert.py        CSV -> npz converter (paper-exact 2006-19/2020/2021-22 split)
  data/dataset.py        PyTorch Dataset (standardisation, log-BA loss weights)
  models/lstm.py         Model 0: LSTM baseline replication
  models/temporal_transformer.py   Model B: weather branch
  train.py               shared training loop (AdamW, cosine LR, early stop on val AUPRC)
configs/                 one json per experiment
runs/                    checkpoints + results (gitignored)
```

## Reproduce

```
pip install -r requirements.txt
python -m wildfire.data.convert          # needs ../data/positives.csv etc.
python -m wildfire.train --config configs/lstm.json
```

Smoke test: `python -m wildfire.train --config configs/lstm.json --epochs 2 --limit 2000`

## Baselines to beat (Mesogeos, Kondylatos et al. 2023)

| Model | F1 | AUPRC |
|---|---|---|
| LSTM | 0.786 | 0.853 |
| Transformer | 0.780 | 0.856 |
| GTN | 0.786 | 0.858 |

Target: AUPRC >= 0.86.
