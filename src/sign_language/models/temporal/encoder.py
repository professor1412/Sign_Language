"""Time-preserving shared encoder for cached (batch, time, features) input."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        y = self.dropout(torch.relu(self.norm(y)))
        return (x + y).masked_fill(~mask, 0)


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=1024, width=512, hidden_size=256, layers=2, dropout=0.1):
        super().__init__()
        if min(input_dim, width, hidden_size, layers) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid encoder dimensions or dropout")
        self.input_dim = input_dim
        self.output_dim = hidden_size * 2
        self.projection = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width),
                                        nn.ReLU(), nn.Dropout(dropout))
        self.blocks = nn.ModuleList([ResidualTemporalBlock(width, dropout) for _ in range(2)])
        self.lstm = nn.LSTM(width, hidden_size, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout if layers > 1 else 0)

    def forward(self, features, lengths):
        if features.ndim != 3 or features.size(-1) != self.input_dim:
            raise ValueError("Expected features with shape (batch, time, input_dim)")
        lengths = lengths.to(device="cpu", dtype=torch.long)
        if (lengths.shape != (features.size(0),) or torch.any(lengths <= 0)
                or torch.any(lengths > features.size(1))):
            raise ValueError("Invalid feature lengths")
        mask = (torch.arange(features.size(1), device=features.device)[None, :]
                < lengths.to(features.device)[:, None]).unsqueeze(-1)
        x = self.projection(features.masked_fill(~mask, 0)).masked_fill(~mask, 0)
        for block in self.blocks:
            x = block(x, mask)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(packed, batch_first=True, total_length=features.size(1))
        return x
