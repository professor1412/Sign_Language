"""Prove a serialized feature tensor preserves downstream reference logits."""
import torch
from ..decoding.greedy import greedy_decode


def verify_equivalence(reference_logits, cached_logits, lengths, atol=1e-5, rtol=1e-4):
    if reference_logits.shape != cached_logits.shape:
        raise ValueError("Cached and original logit shapes differ")
    if not torch.allclose(reference_logits, cached_logits, atol=atol, rtol=rtol):
        raise ValueError("Cached features do not preserve reference logits")
    original = greedy_decode(reference_logits, lengths)
    if original != greedy_decode(cached_logits, lengths):
        raise ValueError("Cached features change reference decoded predictions")
    return {"passed": True, "max_absolute_logit_difference": float((reference_logits-cached_logits).abs().max()),
            "atol": atol, "rtol": rtol, "decoded_sequences_equal": True}
