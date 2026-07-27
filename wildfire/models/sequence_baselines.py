"""
Progressive sequence-model ladder for the weather/dynamic branch. Each model
takes the same input (dynamic [B, T, 13] + static [B, 14]) and returns logits
[B], so they are directly comparable in the ablation.

The ladder and what each step demonstrates:
  CNN1D      local temporal patterns via convolution; receptive field limited by depth
  RNN        adds sequential memory, but vanishing gradients over 30 steps
  GRU        gated memory, lighter than LSTM
  LSTM       (see lstm.py) full gated memory — the Mesogeos baseline
  BiLSTM     reads the sequence forwards and backwards
  Transformer(see temporal_transformer.py) global attention, no recurrence
"""
import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """Temporal convolution over the day axis."""

    def __init__(self, n_dynamic=13, n_static=14, channels=64, layers=3, dropout=0.3):
        super().__init__()
        blocks, c_in = [], n_dynamic
        for _ in range(layers):
            blocks += [nn.Conv1d(c_in, channels, kernel_size=3, padding=1),
                       nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(dropout)]
            c_in = channels
        self.net = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels + n_static, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        x = batch["dynamic"].transpose(1, 2)     # [B, n_dynamic, T]
        h = self.net(x).mean(dim=2)              # global average pool over time
        z = torch.cat([h, batch["static"]], dim=1)
        return self.head(z).squeeze(1)


class RNNBaseline(nn.Module):
    """Generic recurrent baseline: cell in {rnn, gru, lstm}, optional bidirectional."""

    def __init__(self, cell="lstm", n_dynamic=13, n_static=14, hidden=128,
                 layers=2, dropout=0.3, bidirectional=False):
        super().__init__()
        self.cell = cell
        self.bidirectional = bidirectional
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(n_dynamic, hidden, num_layers=layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0,
                           bidirectional=bidirectional)
        out = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out + n_static, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch):
        if self.cell == "lstm":
            _, (h, _) = self.rnn(batch["dynamic"])
        else:
            _, h = self.rnn(batch["dynamic"])
        # h: [layers*directions, B, hidden] — take the last layer's final state(s)
        feat = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        z = torch.cat([feat, batch["static"]], dim=1)
        return self.head(z).squeeze(1)
