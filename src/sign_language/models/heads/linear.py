"""Head A: one learned score per gloss, plus CTC blank."""
from torch import nn


class LinearHead(nn.Linear):
    def __init__(self, input_dim, num_classes):
        if num_classes < 2:
            raise ValueError("Need at least one gloss plus blank")
        super().__init__(input_dim, num_classes)
