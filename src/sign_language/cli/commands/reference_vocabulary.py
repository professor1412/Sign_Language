"""Convert the authors' trusted NumPy gloss dictionary to inspectable JSON."""
from pathlib import Path
import numpy as np
from ...core.provenance import file_sha256
from ...core.settings import write_json, validate_vocabulary


def import_vocabulary(args):
    if Path(args.output).exists():
        raise ValueError("Vocabulary output already exists; choose a new path")
    dictionary = np.load(args.trusted_npy, allow_pickle=True).item()
    pairs = sorted((int(value[0]), gloss) for gloss, value in dictionary.items())
    if [i for i, _ in pairs] != list(range(1, len(pairs)+1)):
        raise ValueError("Expected consecutive released gloss IDs starting at 1")
    labels = ["<blank>"] + [gloss for _, gloss in pairs]
    value = {"schema_version": 1, "blank_id": 0, "checkpoint_compatible": True,
             "source_dataset": "phoenix2014-T", "id_to_gloss": labels,
             "gloss_to_id": {gloss: i for i, gloss in enumerate(labels)},
             "source_npy_sha256": file_sha256(args.trusted_npy)}
    validate_vocabulary(value)
    write_json(args.output, value)
    print(f"Saved {len(pairs)} released gloss IDs plus blank: {args.output}")
