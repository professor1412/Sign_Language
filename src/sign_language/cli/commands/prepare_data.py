"""Atomic, non-overwriting Phase 1 artifact generation."""
import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from ...data.manifest import build_manifest
from ...data.vocabulary import build_vocabulary
from ...data.splits import split_training
from ...data.validation import validate_disjoint
from ...core.provenance import fingerprint, file_sha256


def prepare(args):
    root = args.dataset_root.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset not found: {root}")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Output must be outside the original dataset and its ancestors")
    if output.exists():
        raise ValueError("Output already exists; use a new --output path to preserve the earlier experiment")
    manifests = {}
    for split in ("train", "dev", "test"):
        print(f"Checking {split} annotations and frame files...", flush=True)
        manifests[split] = build_manifest(root, split, args.all_png_headers)
    validate_disjoint(manifests)
    explicit = args.held_out_file.read_text(encoding="utf-8").splitlines() if args.held_out_file else None
    reduced, policy = split_training(manifests["train"], args.held_out_fraction, args.seed,
                                     explicit, args.min_retained)
    policy["selection"] = "explicit list" if explicit else "seeded random training vocabulary; not manually family-balanced"
    policy["requested_fraction"] = args.held_out_fraction
    policy["model_selection"] = "seen-only development videos; no final test tuning"
    seen = set(policy["seen"])
    held = set(policy["held_out"])
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "dataset_root": str(root), "python": platform.python_version(),
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "free_disk_bytes": shutil.disk_usage(output.parent if output.parent.exists() else Path.cwd()).free,
              "validation": "all frame names, continuity and nonzero sizes; " + ("all PNG headers" if args.all_png_headers else "first PNG header per clip"),
              "limitations": "PNG payload decoding and GPU/model execution not verified", "splits": {}}
    for split, rows in manifests.items():
        report["splits"][split] = {"videos": len(rows), "frames": sum(r["frame_count"] for r in rows),
                "gloss_types": len(build_vocabulary(rows)["counts"]),
                "signers": sorted({r["speaker"] for r in rows}),
                "manifest_sha256": fingerprint(rows),
                "annotations_sha256": file_sha256(root / "annotations/manual" / f"PHOENIX-2014-T.{split}.corpus.csv")}
    report["reduced_training_videos"] = len(reduced)
    report["split_sha256"] = fingerprint(policy)
    output.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        # The renamed bundle keeps its staging permissions. Inherit the output
        # parent's Windows ACL so the desktop user can open the published files.
        staging = output.parent / f".phase1-{uuid4().hex}"
        staging.mkdir()
    else:
        staging = Path(tempfile.mkdtemp(prefix=".phase1-", dir=output.parent))
    def write(name, value):
        path = staging / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".jsonl":
            text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in value)
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")
    try:
        for split, rows in manifests.items():
            write(f"manifests/reference/{split}.jsonl", rows)
        write("manifests/reduced/train.jsonl", reduced)
        for split in ("dev", "test"):
            rows = manifests[split]
            write(f"manifests/reduced/{split}_seen_only.jsonl", [r for r in rows if set(r["glosses"]).issubset(seen)])
            write(f"manifests/reduced/{split}_held_out.jsonl", [r for r in rows if held.intersection(r["glosses"])])
            report["splits"][split]["seen_only_videos"] = sum(set(r["glosses"]).issubset(seen) for r in rows)
            report["splits"][split]["held_out_videos"] = sum(bool(held.intersection(r["glosses"])) for r in rows)
        write("vocabularies/reference.json", build_vocabulary(manifests["train"]))
        write("vocabularies/reduced.json", build_vocabulary(reduced))
        write("splits/held_out.json", policy)
        write("validation_report.json", report)
        write("experiment_register.json", {"schema_version": 1, "runs": []})
        if output.exists():
            raise ValueError("Output appeared during preparation; refusing overwrite")
        os.rename(staging, output)
    except BaseException:
        if staging.resolve().parent != output.parent.resolve() or not staging.name.startswith(".phase1-"):
            raise RuntimeError("Unsafe staging cleanup path")
        shutil.rmtree(staging)
        raise
    print(f"Prepared {output}; retained {len(reduced)}/{len(manifests['train'])} training videos.")
