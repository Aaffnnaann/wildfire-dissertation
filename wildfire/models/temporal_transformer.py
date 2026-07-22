"""
Temporal Transformer (weather/dynamic branch — Model B of the ablation).

Per-day linear projection -> sinusoidal positional encoding -> Transformer
encoder. Exposes the full day-token sequence (needed later as the Query set
for cross-modal fusion) plus a pooled classification path for standalone use.
"""
import math
import torch
import torch.nn as nn


def sinusoidal_encoding(length: int, dim: int) -> torch.Tensor:
    pos = torch.arange(length).unsqueeze(1).float()
    i = torch.arange(0, dim, 2).float()
    angle = pos / torch.pow(10000.0, i / dim)
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class TemporalTransformer(nn.Module):
    def __init__(self, n_dynamic=13, n_static=14, d_model=256, layers=6,
                 heads=4, ff=512, dropout=0.2, use_static=True):
        super().__init__()
        self.embed = nn.Linear(n_dynamic, d_model)
        self.register_buffer("pe", sinusoidal_encoding(64, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.use_static = use_static
        head_in = d_model + (n_static if use_static else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, dynamic: torch.Tensor) -> torch.Tensor:
        """dynamic [B, T, n_dynamic] -> day tokens [B, T, d_model]."""
        h = self.embed(dynamic) + self.pe[: dynamic.shape[1]]
        return self.encoder(h)

    def forward(self, batch):
        tokens = self.encode(batch["dynamic"])
        z = tokens.mean(dim=1)                      # pool over days
        if self.use_static:
            z = torch.cat([z, batch["static"]], dim=1)
        return self.head(z).squeeze(1)              # logits [B]
