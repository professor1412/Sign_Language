"""Reproducible controls, applied only to ordinary label rows."""
import torch
from torch.nn.functional import normalize


def control_vectors(vectors, special_ids, kind, seed):
    result = vectors.clone()
    ordinary = [i for i in range(len(vectors)) if i not in special_ids]
    generator = torch.Generator().manual_seed(seed)
    if kind == "b_rand":
        result[ordinary] = normalize(torch.randn(len(ordinary), vectors.size(1), generator=generator), dim=-1)
    elif kind == "b_shuf":
        order = torch.randperm(len(ordinary), generator=generator)
        result[ordinary] = vectors[ordinary][order]
    return result
