"""Small, strict JSON/YAML configuration helpers."""
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = [(r["split"], r["id"]) for r in rows]
    if not rows or len(keys) != len(set(keys)):
        raise ValueError("Manifest must be nonempty with unique split/video IDs")
    return rows


def validate_vocabulary(value):
    labels = value["id_to_gloss"]
    if (value["blank_id"] != 0 or not labels or labels[0] != "<blank>"
            or len(labels) != len(set(labels))
            or value["gloss_to_id"] != {g: i for i, g in enumerate(labels)}):
        raise ValueError("Vocabulary must have unique, consecutive IDs and <blank> at ID 0")
    return value
