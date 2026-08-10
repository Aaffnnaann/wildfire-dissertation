"""
Dual-branch model fusing the satellite (ViT) and weather (Temporal Transformer)
encoders. Covers two ablation arms with one `fusion` switch:

  fusion="concat"  Model C — naive fusion: [CLS_sat ; mean(weather) ; static] -> MLP
  fusion="cross"   Model D — cross-modal attention (the dissertation's contribution):
                   weather day-tokens are Queries; satellite patch tokens are
                   Keys/Values. The [B, T, 16] attention map is the interpretable
                   "given day t's weather, which regions matter" output.

Both reuse the standalone encoders' encode() methods, so Models A/B/C/D share
identical branch definitions.
"""
import torch
import torch.nn as nn

from wildfire.models.vit import ViTEncoder
from wildfire.models.temporal_transformer import TemporalTransformer


class DualBranchFusion(nn.Module):
    def __init__(self, fusion="cross", n_static=14, dropout=0.3, heads=8,
                 gated=False, sat_kwargs=None, weather_kwargs=None):
        super().__init__()
        assert fusion in ("concat", "cross")
        self.fusion = fusion
        self.gated = gated and fusion == "cross"

        sat_kwargs = {"d_model": 256, "layers": 6, "heads": 8, "ff": 512,
                      "dropout": dropout, **(sat_kwargs or {})}
        weather_kwargs = {"d_model": 192, "layers": 4, "heads": 4, "ff": 384,
                          "dropout": dropout, **(weather_kwargs or {})}
        d_sat = sat_kwargs["d_model"]
        self.d_weather = weather_kwargs["d_model"]

        # Encoders used for their token sequences only (no internal heads/static).
        self.sat = ViTEncoder(use_static=False, **sat_kwargs)
        self.weather = TemporalTransformer(use_static=False, **weather_kwargs)

        if fusion == "cross":
            self.q_proj = nn.Linear(self.d_weather, d_sat)  # bridge 192 -> 256
            self.attn = nn.MultiheadAttention(d_sat, heads, dropout=dropout,
                                              batch_first=True)
            if self.gated:
                # Weather is the base; satellite is admitted through a learned gate,
                # so at worst the gate closes and fusion == weather-only (never below).
                self.sat_to_w = nn.Linear(d_sat, self.d_weather)
                self.gate = nn.Linear(2 * self.d_weather, self.d_weather)
                fused_dim = self.d_weather
            else:
                fused_dim = d_sat
        else:  # concat
            fused_dim = d_sat + self.d_weather

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim + n_static),
            nn.Linear(fused_dim + n_static, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.last_attn = None   # stashed [B, T, n_patches] map for interpretability
        self.last_gate = None   # stashed mean gate value (how much satellite was used)

    def forward(self, batch):
        sat_tokens = self.sat.encode(batch["patch"])       # [B, 1+n_patches, d_sat]
        w_tokens = self.weather.encode(batch["dynamic"])   # [B, T, d_weather]

        if self.fusion == "cross":
            q = self.q_proj(w_tokens)                       # [B, T, d_sat]
            kv = sat_tokens[:, 1:]                           # drop CLS -> patch tokens
            ctx, attn = self.attn(q, kv, kv,
                                  need_weights=True, average_attn_weights=True)
            self.last_attn = attn.detach()                  # [B, T, n_patches]
            ctx = ctx.mean(dim=1)                            # pool over days -> [B, d_sat]
            if self.gated:
                w_sum = w_tokens.mean(dim=1)                 # [B, d_weather]
                c = self.sat_to_w(ctx)                       # [B, d_weather]
                g = torch.sigmoid(self.gate(torch.cat([w_sum, c], dim=1)))
                self.last_gate = g.detach().mean().item()
                fused = w_sum + g * c                        # gated residual
            else:
                fused = ctx
        else:
            fused = torch.cat([sat_tokens[:, 0], w_tokens.mean(dim=1)], dim=1)

        z = torch.cat([fused, batch["static"]], dim=1)
        return self.head(z).squeeze(1)
