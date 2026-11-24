"""Feature-only datasets; unknown evaluation labels remain in WER references."""
from torch.utils.data import Dataset
from ..features.cache import FeatureCache, record_key
from ..core.settings import read_manifest, validate_vocabulary


class FeatureDataset(Dataset):
    def __init__(self, cache, vocabulary, manifest=None, training=False):
        self.cache = FeatureCache(cache)
        self.vocabulary = validate_vocabulary(vocabulary)
        self.entries = self.cache.entries
        if manifest is not None:
            self.entries = []
            for record in read_manifest(manifest):
                entry = self.cache.by_key.get(record_key(record))
                if entry is None or entry["record"] != record:
                    raise ValueError(f"Manifest record missing or changed in cache: {record['id']}")
                self.entries.append(entry)
        if training:
            for entry in self.entries:
                record = entry["record"]
                if record["split"] != "train":
                    raise ValueError("Training data must have split=train")
                unknown = set(record["glosses"]) - self.vocabulary["gloss_to_id"].keys()
                if unknown or "<blank>" in record["glosses"]:
                    raise ValueError(f"Unknown/reserved training glosses in {record['id']}: {sorted(unknown)}")
                labels = record["glosses"]
                required = len(labels) + sum(a == b for a, b in zip(labels, labels[1:]))
                if entry["length"] < required:
                    raise ValueError(f"Impossible CTC alignment for {record['id']}: needs {required} feature steps")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        record = entry["record"]
        mapping = self.vocabulary["gloss_to_id"]
        unknown = any(g not in mapping for g in record["glosses"])
        return {"features": self.cache.load(entry), "id": record["id"], "glosses": record["glosses"],
                "target_ids": [] if unknown else [mapping[g] for g in record["glosses"]]}
