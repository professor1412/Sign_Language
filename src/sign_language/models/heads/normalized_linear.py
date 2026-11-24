"""CorrNet checkpoint-compatible normalized classifier (not custom head A)."""
import torch
from torch import nn
from torch.nn import functional as F


class NormLinear(nn.Module):
    def __init__(self, input_dim, classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, classes))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        return x @ F.normalize(self.weight, dim=0)
