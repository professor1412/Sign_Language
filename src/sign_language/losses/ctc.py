"""CTC with explicit impossible-alignment checks; never hide infinite losses."""
import torch
from torch.nn import functional as F


def ctc_loss(logits, input_lengths, targets, target_lengths):
    if logits.ndim != 3 or not torch.isfinite(logits).all():
        raise ValueError("CTC expects finite (batch, time, classes) logits")
    batch, time, classes = logits.shape
    il = input_lengths.to(device="cpu", dtype=torch.long)
    tl = target_lengths.to(device="cpu", dtype=torch.long)
    flat = targets.to(device="cpu", dtype=torch.long)
    if il.shape != (batch,) or tl.shape != (batch,) or flat.ndim != 1:
        raise ValueError("Invalid CTC length/target shapes")
    if torch.any(il <= 0) or torch.any(il > time) or torch.any(tl < 0) or int(tl.sum()) != flat.numel():
        raise ValueError("Invalid CTC lengths")
    if torch.any(flat <= 0) or torch.any(flat >= classes):
        raise ValueError("CTC targets must be valid gloss IDs and must not contain blank")
    offset = 0
    for i, n in enumerate(tl.tolist()):
        labels = flat[offset:offset+n]
        required = n + int((labels[1:] == labels[:-1]).sum())
        if il[i].item() < required:
            raise ValueError(f"Impossible CTC alignment in example {i}: needs {required} steps, has {il[i].item()}")
        offset += n
    loss = F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), flat.to(logits.device),
                      il, tl, blank=0, reduction="mean", zero_infinity=False)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite CTC loss")
    return loss
