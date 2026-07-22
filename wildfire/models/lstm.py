"""LSTM baseline (replication of the Mesogeos Track A baseline)."""
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, n_dynamic=13, n_static=14, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_dynamic, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        _, (h, _) = self.lstm(batch["dynamic"])   # h: [layers, B, hidden]
        z = torch.cat([h[-1], batch["static"]], dim=1)
        return self.head(z).squeeze(1)            # logits [B]
