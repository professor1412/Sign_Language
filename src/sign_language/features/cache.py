"""Versioned, resumable feature storage with per-file integrity checks."""
from pathlib import Path
import torch
from ..core.provenance import fingerprint, file_sha256
from ..core.settings import read_json, write_json


def record_key(record):
    return fingerprint([record["split"], record["id"]])


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def validate_features(features, dimension):
    if (not isinstance(features, torch.Tensor) or features.ndim != 2 or features.size(0) < 1
            or features.size(1) != dimension or features.dtype != torch.float32
            or not torch.isfinite(features).all()):
        raise ValueError(f"Features must be finite float32 with shape (positive time, {dimension})")


class CacheWriter:
    def __init__(self, root, producer, records, resume=False):
        self.root = Path(root)
        self.producer = producer
        self.records = records
        self.metadata = {"schema_version": 1, "producer": producer,
                         "producer_sha256": fingerprint(producer), "manifest_sha256": fingerprint(records)}
        if self.root.exists():
            if not resume:
                raise ValueError(f"Cache already exists: {root}; use --resume for the identical extraction")
            if read_json(self.root / "metadata.json") != self.metadata:
                raise ValueError("Cache metadata differs: use a new output folder")
        else:
            self.root.mkdir(parents=True)
            write_json(self.root / "metadata.json", self.metadata)
        index_path = self.root / "index.json"
        self.previous_hashes = {}
        if index_path.exists():
            index = read_json(index_path)
            if index["metadata"] != self.metadata or not index.get("complete"):
                raise ValueError("Existing feature index is inconsistent")
            self.previous_hashes = {e["file"]: e["sha256"] for e in index["entries"]}
        self.entries = []

    def path(self, record):
        return self.root / "tensors" / (record_key(record) + ".pt")

    def existing(self, record):
        path = self.path(record)
        if not path.exists():
            return None
        expected = self.previous_hashes.get(path.relative_to(self.root).as_posix())
        if expected and file_sha256(path) != expected:
            raise ValueError(f"Corrupt feature file: {path}")
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["record"] != record or item["producer_sha256"] != self.metadata["producer_sha256"]:
            raise ValueError(f"Feature identity/provenance mismatch: {path}")
        validate_features(item["features"], self.producer["feature_dim"])
        return item

    def add(self, record, features=None, extra=None):
        if features is not None:
            features = features.detach().cpu().float().contiguous()
            validate_features(features, self.producer["feature_dim"])
            if self.path(record).exists():
                raise ValueError("Refusing to overwrite an existing feature tensor")
            atomic_torch_save({"record": record, "features": features,
                               "producer_sha256": self.metadata["producer_sha256"], "extra": extra or {}}, self.path(record))
        item = self.existing(record)
        if item is None:
            raise ValueError("Feature file is missing")
        self.entries.append({"record": record, "file": self.path(record).relative_to(self.root).as_posix(),
                             "length": len(item["features"]), "sha256": file_sha256(self.path(record)),
                             "extra": item["extra"]})
        return item

    def finish(self):
        if [e["record"] for e in self.entries] != self.records:
            raise ValueError("Cannot finalize an incomplete or reordered cache")
        write_json(self.root / "index.json", {"schema_version": 1, "metadata": self.metadata,
                                             "entries": self.entries, "complete": True})


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root).resolve()
        index = read_json(self.root / "index.json")
        self.metadata = read_json(self.root / "metadata.json")
        if (not index.get("complete") or index["metadata"] != self.metadata
                or fingerprint(self.metadata["producer"]) != self.metadata["producer_sha256"]):
            raise ValueError("Incomplete or inconsistent feature cache")
        self.entries = index["entries"]
        if fingerprint([e["record"] for e in self.entries]) != self.metadata["manifest_sha256"]:
            raise ValueError("Feature index manifest checksum differs")
        self.by_key = {record_key(e["record"]): e for e in self.entries}
        if not self.entries or len(self.by_key) != len(self.entries):
            raise ValueError("Cache must contain unique, nonempty records")
        self.index_sha256 = file_sha256(self.root / "index.json")
        self._verified = set()

    def load(self, entry):
        path = (self.root / entry["file"]).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Feature path escapes cache")
        if str(path) not in self._verified:
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Corrupt feature file: {path}")
            self._verified.add(str(path))
        item = torch.load(path, map_location="cpu", weights_only=True)
        if item["record"] != entry["record"] or item["producer_sha256"] != self.metadata["producer_sha256"]:
            raise ValueError("Feature identity/provenance mismatch")
        validate_features(item["features"], self.metadata["producer"]["feature_dim"])
        if len(item["features"]) != entry["length"]:
            raise ValueError("Feature length mismatch")
        return item["features"]
