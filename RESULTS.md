# Results — Mesogeos Track A (next-day fire-danger forecasting)

Primary metric: test-set AUPRC. Chronological split (train 2006–19 / val 2020 / test 2021–22).
Published baselines: LSTM 0.853, Transformer 0.856, GTN 0.858 (Kondylatos et al., NeurIPS 2023).

## Best result

**Ensemble (HistGradientBoosting + GRU + CNN-1D + gated fusion): test AUPRC 0.8847, F1 0.8078.**
Averaging the probabilities of four diverse model families beats every single model and the published
GTN SOTA (0.858) by +0.027. The multimodal fusion model contributes decorrelated predictions.

## Full model comparison (test AUPRC, ranked)

| Rank | Model | Family | Test AUPRC | Test F1 |
|---|---|---|---|---|
| — | **Ensemble (4 models)** | **classical + deep + multimodal** | **0.8847** | **0.808** |
| 1 | HistGradientBoosting | classical | 0.8739 | 0.783 |
| 2 | GRU | weather (deep) | 0.8643 | 0.768 |
| 3 | CNN-1D | weather (deep) | 0.8615 | 0.779 |
| 4 | LSTM | weather (deep) | 0.8601 | 0.771 |
| 5 | BiLSTM | weather (deep) | 0.8586 | 0.784 |
| 6 | RNN | weather (deep) | 0.8519 | 0.773 |
| 7 | Gated fusion (D-v2) | multimodal (improved) | 0.8478 | 0.775 |
| 8 | Transformer | weather (deep) | 0.8425 | 0.765 |
| 9 | Concat fusion (C) | multimodal | 0.8402 | 0.764 |
| 10 | Cross-attn fusion (D-v1) | multimodal | 0.8259 | 0.734 |
| 11 | Random Forest | classical | 0.8048 | 0.732 |
| 12 | ViT (A-v2, mask+aug) | satellite (improved) | 0.7609 | 0.669 |
| 13 | ViT (A-v1) | satellite | 0.7154 | 0.638 |

## Key findings

1. **Gradient boosting (0.874) is strongest** — classical trees beat all deep models on this
   small tabular dataset (cf. Grinsztajn et al. 2022).
2. **GRU (0.864) beats the published GTN (0.858)** — best neural model exceeds the benchmark SOTA.
3. **Naive multimodal fusion hurts**: concatenation (0.840) and un-gated cross-attention (0.826)
   fall below the weather-only models — a weak, noisy satellite branch corrupts the strong weather signal.
4. **Gated fusion recovers it** (0.848): a learned gate + validity-mask channel + patch augmentation
   lift fusion above naive concat and its own weather branch — but not above the best unimodal model.
5. **The satellite modality adds little for ignition** — foreshadowed by the EDA (mean NDVI 0.462 on
   fire days vs 0.464 on no-fire days). Weather dominates next-day ignition risk.

## Improvements (v1 → v2)

| Model | v1 | v2 | change |
|---|---|---|---|
| ViT (satellite) | 0.7154 | 0.7609 | +0.046 (validity mask + augmentation) |
| Cross-attn fusion | 0.8259 | 0.8478 | +0.022 (gated residual fusion) |

## Window-length ablation (GRU, all else fixed)

Answers "why 30 days and not 1/2/3?". Test AUPRC rises steeply then plateaus:

| Window (days) | Val AUPRC | Test AUPRC | Test F1 |
|---|---|---|---|
| 1 | 0.8454 | 0.8180 | 0.7423 |
| 3 | 0.8696 | 0.8382 | 0.7630 |
| 7 | 0.8828 | 0.8582 | 0.7775 |
| 14 | 0.8934 | 0.8605 | 0.7697 |
| 30 | 0.8946 | 0.8643 | 0.7679 |

~87% of the gain (1&rarr;30: +0.046 AUPRC) occurs within the first 7 days; it plateaus after ~14.
Short windows (1&ndash;3 days) discard antecedent-drought signal (a ~0.03&ndash;0.05 AUPRC penalty);
30 days is the benchmark standard sitting on the saturated plateau. Figure: figures/window_ablation.png.

## Feature ablation (can we drop variables?)

Random-Forest importance (top): t2m 21.8%, d2m 8.9%, lst_night 8.3%, ndvi 8.0%, ssrd 7.8%,
rh 7.2%, smi 7.2%. All 13 dynamic variables outrank every static one (each static &lt;1%).

Test AUPRC (gradient boosting) vs. number of variables:

| Variables | Test AUPRC |
|---|---|
| top-3 | 0.747 |
| top-5 | 0.801 |
| top-8 (weather) | 0.820 |
| top-13 (all dynamic) | 0.841 |
| top-20 | 0.871 |
| all 27 | 0.874 |

Performance rises monotonically &mdash; variables are NOT largely redundant; the full set is justified.
Dropping to 8 costs 0.054; dropping all statics costs 0.033. A lean top-8 weather model recovers ~94%.
Figure: figures/feature_ablation.png.

Interactive report: https://claude.ai/code/artifact/626b119d-2b39-42af-b98e-d8d621cf9050
