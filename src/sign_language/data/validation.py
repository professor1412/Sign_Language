"""Cross-split checks independent of model training."""


def validate_disjoint(manifests):
    seen = set()
    for split, rows in manifests.items():
        ids = {r["id"] for r in rows}
        if seen & ids:
            raise ValueError(f"Duplicate video IDs across splits: {split}")
        seen.update(ids)
