"""Use training labels only for seeded held-out selection."""
import random
from .vocabulary import build_vocabulary


def split_training(records, fraction=0.1, seed=42, held_out=None, min_retained=0.2):
    if not 0 < fraction < 1 or not 0 < min_retained <= 1:
        raise ValueError("Invalid split fractions")
    vocabulary = set(build_vocabulary(records)["counts"])
    eligible = sorted(g for g in vocabulary if not g.startswith("__"))
    explicit = held_out is not None
    if held_out is None:
        if len(eligible) < 2:
            raise ValueError("Need at least two eligible glosses")
        held_out = random.Random(seed).sample(eligible, max(1, int(len(eligible) * fraction)))
    held_out = set(held_out)
    if not held_out or not held_out.issubset(eligible):
        raise ValueError("Held-out list must contain known ordinary training glosses")
    kept = [r for r in records if held_out.isdisjoint(r["glosses"])]
    if not kept or len(kept) / len(records) < min_retained:
        raise ValueError("Too little training data remains; revise the declared split before experiments")
    seen = {g for r in kept for g in r["glosses"]}
    return kept, {"seed": seed, "selection": "explicit" if explicit else "random",
                  "held_out": sorted(held_out), "seen": sorted(seen),
                  "incidentally_unseen": sorted(vocabulary - seen - held_out),
                  "retained_videos": len(kept), "removed_videos": len(records)-len(kept)}
