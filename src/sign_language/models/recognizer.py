"""Recognition from cached features; the visual backbone stays frozen."""
from torch import nn
from .temporal.encoder import TemporalEncoder
from .heads.linear import LinearHead


class Recognizer(nn.Module):
    def __init__(self, num_classes, head_config=None, vectors=None, special_ids=None, **encoder_config):
        super().__init__()
        self.encoder = TemporalEncoder(**encoder_config)
        from .factory import create_head
        self.head = create_head(self.encoder.output_dim, num_classes, head_config or {}, vectors, special_ids)

    def forward(self, features, lengths):
        return self.head(self.encoder(features, lengths))
