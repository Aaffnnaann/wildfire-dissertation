"""
Gated Transformer Network (GTN) — the third published Mesogeos baseline.

Follows Liu et al., "Gated Transformer Networks for Multivariate Time Series
Classification" (arXiv:2103.14438). Two parallel Transformer towers process the
same multivariate series along different axes:

  * step-wise tower    : tokens are time steps  (attends across time)
  * channel-wise tower : tokens are variables   (attends across channels)

Each tower's encoder output is flattened, and a learned gate (a softmax over two
scalars derived from both towers) weights their concatenation before the
classifier. Positional encoding is applied only to the step-wise tower, since
channel order carries no ordinal meaning.

This implementation follows the project convention of appending the static
per-cell features immediately before the classifier, so that GTN sees the same
inputs as the other models in this study.
"""
import torch
import torch.nn as nn

from wildfire.models.temporal_transformer import sinusoidal_encoding


class GatedTransformerNetwork(nn.Module):
    def __init__(self, n_dynamic=13, n_static=14, window=30, d_model=128,
                 layers=2, heads=4, ff=256, dropout=0.2, use_static=True):
        super().__init__()
        self.use_static = use_static
        self.window = window

        def encoder():
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=ff, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, layers)

        # step-wise tower: one token per time step, embedding over variables
        self.step_embed = nn.Linear(n_dynamic, d_model)
        self.register_buffer("pe", sinusoidal_encoding(256, d_model))
        self.step_encoder = encoder()

        # channel-wise tower: one token per variable, embedding over time
        self.chan_embed = nn.Linear(window, d_model)
        self.chan_encoder = encoder()

        step_flat = window * d_model
        chan_flat = n_dynamic * d_model
        self.gate = nn.Linear(step_flat + chan_flat, 2)

        head_in = step_flat + chan_flat + (n_static if use_static else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Dropout(dropout),
            nn.Linear(head_in, 1),
        )

    def forward(self, batch):
        x = batch["dynamic"]                                  # [B, T, C]
        b = x.shape[0]

        step = self.step_embed(x) + self.pe[: x.shape[1]]     # [B, T, d]
        step = self.step_encoder(step).reshape(b, -1)         # [B, T*d]

        chan = self.chan_embed(x.transpose(1, 2))             # [B, C, d]
        chan = self.chan_encoder(chan).reshape(b, -1)         # [B, C*d]

        both = torch.cat([step, chan], dim=1)
        g = torch.softmax(self.gate(both), dim=1)             # [B, 2]
        fused = torch.cat([step * g[:, 0:1], chan * g[:, 1:2]], dim=1)

        if self.use_static:
            fused = torch.cat([fused, batch["static"]], dim=1)
        return self.head(fused).squeeze(1)                    # logits [B]
