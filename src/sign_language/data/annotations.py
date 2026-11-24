"""Read gloss targets separately from German translations."""
import csv
from pathlib import Path


def read_annotations(path):
    records = []
    names = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="|")
        required = {"name", "video", "start", "end", "speaker", "orth", "translation"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing annotation columns in {path}")
        for line, row in enumerate(reader, 2):
            name = row["name"]
            if not name or name in names or name in {".", ".."} or any(c in name for c in "/\\"):
                raise ValueError(f"Invalid or duplicate clip ID at {path}:{line}")
            glosses = (row["orth"] or "").split()
            if not glosses or not row["speaker"]:
                raise ValueError(f"Missing glosses/signer at {path}:{line}")
            names.add(name)
            records.append({"id": name, "speaker": row["speaker"], "glosses": glosses,
                            "translation": row["translation"], "annotation_video": row["video"]})
    if not records:
        raise ValueError(f"Empty annotations: {path}")
    return records
