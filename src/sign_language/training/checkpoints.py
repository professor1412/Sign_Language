"""Checkpoints contain tensors/primitives and use safe weights-only loading."""
import torch
from ..features.cache import atomic_torch_save


def save_checkpoint(path, payload):
    atomic_torch_save({"schema_version": 1, **payload}, path)


def load_checkpoint(path):
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.get("schema_version") != 1:
        raise ValueError("Unsupported checkpoint schema")
    return value
