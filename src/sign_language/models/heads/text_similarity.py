"""Head B cosine classifier with fixed ordinary rows and learned special rows."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class TextSimilarityHead(nn.Module):
    def __init__(self, input_dim, vectors, special_ids, temperature=0.07, learn_temperature=True,
                 projection_dim=None):
        super().__init__()
        if not math.isfinite(temperature) or not 0.01 <= temperature <= 1:
            raise ValueError("Temperature must be between 0.01 and 1")
        if vectors.ndim != 2 or vectors.size(1) != 300 or not torch.isfinite(vectors).all():
            raise ValueError("Expected finite (classes, 300) text vectors")
        if 0 not in special_ids or len(set(special_ids)) != len(special_ids):
            raise ValueError("Special IDs must be unique and include blank 0")
        if min(special_ids) < 0 or max(special_ids) >= len(vectors):
            raise ValueError("Special ID outside vocabulary")
        self.register_buffer("text_vectors", vectors.detach().clone().float())
        self.register_buffer("special_ids", torch.tensor(special_ids, dtype=torch.long))
        dimension = projection_dim or 300
        self.video_projection = nn.Linear(input_dim, dimension)
        self.text_projection = nn.Linear(300, dimension) if projection_dim else nn.Identity()
        self.special_vectors = nn.Parameter(F.normalize(torch.randn(len(special_ids), dimension), dim=-1))
        self.log_tau = nn.Parameter(torch.tensor(math.log(temperature)), requires_grad=learn_temperature)

    @property
    def temperature(self):
        return self.log_tau.exp().clamp(0.01, 1.0)

    def forward(self, features):
        text = self.text_projection(self.text_vectors)
        text = text.index_copy(0, self.special_ids, self.special_vectors)
        video = F.normalize(self.video_projection(features), dim=-1)
        return video @ F.normalize(text, dim=-1).T / self.temperature
