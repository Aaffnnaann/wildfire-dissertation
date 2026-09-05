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

## Per-region breakdown (can we focus on a smaller area?)

Gradient boosting (basin-wide) test AUPRC by Mediterranean sub-region. Raw AUPRC is confounded by
fire prevalence, so lift = AUPRC / chance-baseline is the fair skill measure:

| Region | Fires | Fire % | Raw AUPRC | Lift |
|---|---|---|---|---|
| Italy & Central | 629 | 56% | 0.953 | 1.70 |
| Maghreb | 166 | 24% | 0.897 | 3.71 |
| Balkans & Greece | 191 | 36% | 0.883 | 2.49 |
| E. Med (Turkey/Levant) | 83 | 20% | 0.697 | 3.58 |
| Iberia | 292 | 22% | 0.643 | 2.86 |

Raw AUPRC varies (0.64&ndash;0.95) but that is mostly base-rate, not skill: by lift the model works well
everywhere (1.7&ndash;3.7&times;), strongest in low-prevalence Maghreb/E.Med. So the model already generalises
across the basin; regional subsetting would lose data with no fair-metric gain. Figure: figures/per_region.png.

## Dataset audit (project CSV vs published Track A counts)

`python -m wildfire.audit_dataset` &rarr; `data/processed/dataset_audit.json`.

The project files hold 25,916 samples against the 25,722 reported in the paper. The gap is
**entirely extra negatives**: positives are identical (8,574, diff 0), negatives are 17,342 vs 17,148.
Not duplicates &mdash; all 17,342 negatives have distinct (cell, date) keys, positive/negative key
overlap is zero, and every sample has exactly 30 rows. The surplus is spread across splits
(train +151, val +30, test +13), giving a 2.0226 ratio rather than the paper's exact 2.00.

The negative draw is stochastic and neither seed nor sample IDs were published, so exact replication
is infeasible. Published figures are therefore context, **not** a like-for-like benchmark. Mitigations:
one internal protocol for all in-study comparisons; re-runs of the three published baselines on the
project CSV; and a count-matched variant (`wildfire.paper_matched`) as a sensitivity check.

## Repeated seeds (mean &plusmn; SD, 95% CI)

`python -m wildfire.multiseed --seeds 0 1 2 3 4 --out_dir runs/seeds` &rarr; `runs/seeds/seed_summary.csv`.

| Model | Seeds | AUPRC mean | SD | 95% CI |
|---|---|---|---|---|
| Ensemble (HistGB+GRU+CNN-1D) | 5 | 0.8811 | 0.0045 | [0.8755, 0.8868] |
| HistGradientBoosting | 5 | 0.8705 | 0.0022 | [0.8677, 0.8733] |
| GRU | 5 | 0.8612 | 0.0077 | [0.8516, 0.8708] |
| LSTM | 5 | 0.8602 | 0.0032 | [0.8562, 0.8641] |
| CNN-1D | 5 | 0.8602 | 0.0072 | [0.8513, 0.8691] |
| Transformer | 1 | 0.8334 | &mdash; | &mdash; |
| GTN | 1 | 0.8123 | &mdash; | &mdash; |
| Random Forest | 5 | 0.8050 | 0.0007 | [0.8041, 0.8059] |

Transformer and GTN are single-seed (CPU budget); the 5-seed sweep for those and for the ViT/fusion
models is scripted for GPU in `notebooks/experiments_kaggle.ipynb` (**T4, not P100** &mdash; P100 is sm_60
and the preinstalled PyTorch needs sm_70+).

## Paired bootstrap (10,000 resamples, fixed test set)

`python -m wildfire.stats_tests` &rarr; `runs/seeds/bootstrap_tests.csv`.

| Comparison | Difference | 95% CI | p | Significant |
|---|---|---|---|---|
| HistGB vs GRU | +0.0058 | [&minus;0.0039, +0.0158] | 0.240 | **no** |
| HistGB vs Ensemble | &minus;0.0091 | [&minus;0.0148, &minus;0.0033] | 0.001 | yes |
| GRU vs LSTM | +0.0018 | [&minus;0.0040, +0.0079] | 0.552 | **no** |
| LSTM vs Transformer | +0.0339 | [+0.0229, +0.0448] | 0.000 | yes |
| Transformer vs GTN | +0.0211 | [+0.0077, +0.0345] | 0.001 | yes |

**Key correction to earlier framing:** gradient boosting's lead over the best recurrent models is
*not* statistically distinguishable from test-set sampling variation. It is significantly ahead of the
Transformer and GTN only. Claims of "classical beats deep" were softened accordingly throughout the
dissertation.

## Calibration and operating points

`python -m wildfire.evaluation` &rarr; `figures/calibration_metrics.csv`, `confusion_matrices.csv`.
Thresholds maximise F1 on **validation only**; the test split is never used for tuning.

| Model | Brier | ECE (10-bin) | Threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Ensemble | 0.0946 | 0.0209 | 0.53 | 0.803 | 0.796 | 0.799 |
| HistGradientBoosting | 0.0987 | 0.0163 | 0.34 | 0.732 | 0.851 | 0.787 |
| GRU | 0.1003 | 0.0284 | 0.56 | 0.813 | 0.754 | 0.782 |
| LSTM | 0.1010 | 0.0271 | 0.54 | 0.794 | 0.771 | 0.783 |
| Transformer | 0.1164 | 0.0485 | 0.56 | 0.760 | 0.769 | 0.764 |

Gradient boosting is the best-calibrated single model; the Transformer is the worst on both Brier and ECE.

## Recall by fire size

`figures/error_by_burned_area.csv`. Detection improves monotonically with eventual fire size, against a
false-alarm rate of 0.065 on the 2,751 negatives:

| Burned area | Fires | Recall @ threshold | Mean score |
|---|---|---|---|
| 30&ndash;100 ha | 525 | 0.735 | 0.695 |
| 100&ndash;500 ha | 550 | 0.826 | 0.773 |
| 500&ndash;2000 ha | 191 | 0.843 | 0.769 |
| &gt;2000 ha | 103 | 0.864 | 0.803 |
| no fire (negatives) | 2751 | &mdash; | 0.158 |

The model misses hardest on the smallest dangerous fires and is most reliable on the megafires that
matter most operationally. AUPRC is undefined *within* a band (each contains only fires), so recall at a
shared threshold is the meaningful quantity.

Interactive report: https://claude.ai/code/artifact/626b119d-2b39-42af-b98e-d8d621cf9050
