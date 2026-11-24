"""Experiment-owned IDs; these are NOT asserted to match CorrNet checkpoints."""
from collections import Counter


def build_vocabulary(records):
    counts = Counter(g for r in records for g in r["glosses"])
    if "<blank>" in counts:
        raise ValueError("Reserved blank occurs in target labels")
    labels = ["<blank>"] + sorted(counts)
    return {"schema_version": 1, "blank_id": 0, "checkpoint_compatible": False,
            "id_to_gloss": labels, "gloss_to_id": {g: i for i, g in enumerate(labels)},
            "counts": dict(sorted(counts.items()))}
