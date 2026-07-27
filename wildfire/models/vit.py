"""
Vision Transformer encoder for the satellite branch (Model A of the ablation).

Input is a [B, H, W, C] window (C = NDVI, LAI, LST_day, LST_night). The image is
cut into non-overlapping patches, linearly embedded, given a learnable [CLS]
token and learned positional encodings, and passed through a Transformer encoder.

Exposes:
  encode(img)  -> [B, n_patches+1, d_model]   (CLS + patch tokens; the patch
                                                tokens serve as K/V in fusion)
  forward()    -> logits [B]                   (standalone satellite-only model)
"""
import torch
import torch.nn as nn


class ViTEncoder(nn.Module):
    def __init__(self, img_size=64, patch=16, in_ch=4, d_model=256, layers=6,
                 heads=8, ff=512, dropout=0.2, n_static=14, use_static=True):
        super().__init__()
        assert img_size % patch == 0
        self.patch = patch
        n_patches = (img_size // patch) ** 2
        self.proj = nn.Linear(in_ch * patch * patch, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

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

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, W, C] -> [B, n_patches, C*p*p]."""
        B, H, W, C = x.shape
        p = self.patch
        x = x.reshape(B, H // p, p, W // p, p, C)
        x = x.permute(0, 1, 3, 5, 2, 4).reshape(B, (H // p) * (W // p), C * p * p)
        return x

    def encode(self, img: torch.Tensor) -> torch.Tensor:
        z = self.proj(self.patchify(img))                 # [B, n_patches, d]
        cls = self.cls.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos         # [B, n_patches+1, d]
        return self.encoder(z)

    def forward(self, batch):
        tokens = self.encode(batch["patch"])
        z = tokens[:, 0]                                   # [CLS]
        if self.use_static:
            z = torch.cat([z, batch["static"]], dim=1)
        return self.head(z).squeeze(1)
