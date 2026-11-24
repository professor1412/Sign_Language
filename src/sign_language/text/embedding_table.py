"""Auditable, ID-ordered text embedding bundles."""
from collections import defaultdict
import torch
from torch.nn.functional import normalize
from ..core.settings import validate_vocabulary
from .normalization import is_special, normalize_gloss


def build_table(vocabulary, get_vector, source, strip_prefixes=False):
    validate_vocabulary(vocabulary)
    labels = vocabulary["id_to_gloss"]
    vectors = torch.zeros(len(labels), 300)
    rows, special_ids, ordinary = [], [], []
    for i, gloss in enumerate(labels):
        special = is_special(gloss)
        text = None if special else normalize_gloss(gloss, strip_prefixes)
        rows.append({"id": i, "gloss": gloss, "text": text, "learned_special": special})
        if special:
            special_ids.append(i)
        else:
            vector = torch.as_tensor(get_vector(text), dtype=torch.float32)
            if vector.shape != (300,) or not torch.isfinite(vector).all() or vector.norm() < 1e-8:
                raise ValueError(f"Invalid/zero 300-dimensional vector for {gloss}")
            vectors[i] = normalize(vector, dim=0)
            ordinary.append(i)
    groups = defaultdict(list)
    for row in rows:
        if row["text"] is not None:
            groups[row["text"]].append(row["gloss"])
    similarities = vectors[ordinary] @ vectors[ordinary].T
    pairs = [(labels[ordinary[a]], labels[ordinary[b]], float(similarities[a, b]))
             for a, b in torch.triu(similarities > 0.95, diagonal=1).nonzero().tolist()]
    return {"schema_version": 1, "vocabulary": vocabulary, "vectors": vectors,
            "special_ids": special_ids, "source": source, "strip_prefixes": strip_prefixes,
            "rows": rows, "audit": {"changed_labels": [r for r in rows if r["text"] is not None and r["text"] != r["gloss"]],
            "normalization_collisions": {k: v for k, v in groups.items() if len(v) > 1},
            "near_duplicate_pairs": pairs}}


def load_table(path, vocabulary):
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle.get("schema_version") != 1 or bundle["vocabulary"] != vocabulary:
        raise ValueError("Embedding vocabulary differs from training vocabulary")
    vectors = bundle["vectors"]
    expected_special = [i for i, g in enumerate(vocabulary["id_to_gloss"]) if is_special(g)]
    if (vectors.shape != (len(vocabulary["id_to_gloss"]), 300)
            or bundle["special_ids"] != expected_special or not torch.isfinite(vectors).all()):
        raise ValueError("Invalid embedding bundle dimensions or special IDs")
    ordinary = [i for i in range(len(vectors)) if i not in expected_special]
    if torch.any(vectors[ordinary].norm(dim=-1) < 1e-8):
        raise ValueError("Zero ordinary text vectors")
    return bundle
