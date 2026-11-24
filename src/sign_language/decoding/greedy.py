"""Collapse adjacent repeats BEFORE removing blanks."""
from itertools import groupby


def collapse_ctc(path, blank_id=0):
    return [int(label) for label, _ in groupby(path) if int(label) != blank_id]


def greedy_decode(logits, lengths, blank_id=0):
    if logits.ndim != 3 or len(lengths) != logits.size(0):
        raise ValueError("Expected (batch, time, classes) logits and one length per example")
    paths = logits.argmax(-1).detach().cpu().tolist()
    decoded = []
    for path, length in zip(paths, lengths):
        length = int(length)
        if not 0 < length <= len(path):
            raise ValueError("Invalid decoding length")
        decoded.append(collapse_ctc(path[:length], blank_id))
    return decoded
