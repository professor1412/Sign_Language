"""Pad features, preserving true lengths and concatenated CTC targets."""
import torch
from torch.nn.utils.rnn import pad_sequence


def collate_features(examples):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    return {"features": pad_sequence([e["features"] for e in examples], batch_first=True),
            "lengths": torch.tensor([len(e["features"]) for e in examples], dtype=torch.long),
            "targets": torch.tensor([t for e in examples for t in e["target_ids"]], dtype=torch.long),
            "target_lengths": torch.tensor([len(e["target_ids"]) for e in examples], dtype=torch.long),
            "references": [e["glosses"] for e in examples], "ids": [e["id"] for e in examples]}
