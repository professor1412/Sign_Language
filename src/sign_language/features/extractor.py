"""Frozen extraction with reference scoring and serialized-feature checks."""
from pathlib import Path
import cv2
import torch
from .cache import CacheWriter
from .validation import verify_equivalence
from ..core.settings import read_json, read_manifest, write_json
from ..core.provenance import file_sha256, fingerprint
from ..core.reproducibility import resolve_device
from ..data.transforms import load_corrnet_video, PREPROCESSING
from ..models.backbones.corrnet import load_reference, source_hashes
from ..decoding.greedy import greedy_decode
from ..evaluation.wer import corpus_wer


def extract_features(args):
    if args.limit is not None and args.limit < 1 or args.verify_first < 1:
        raise ValueError("--limit and --verify-first must be positive")
    root, output = args.dataset_root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Cache output must be outside the source dataset and its ancestors")
    records = read_manifest(args.manifest)
    if args.limit is not None:
        records = records[:args.limit]
    vocabulary = read_json(args.vocabulary)
    device = resolve_device(args.device)
    model = load_reference(args.source, args.checkpoint, vocabulary, device, args.trust_legacy_checkpoint)
    # Include local adapter/preprocessing code and versions in cache identity.
    code_root = Path(__file__).parents[1]
    producer = {"kind": "corrnet_temporalconv", "feature_dim": 1024,
                "exposure": "released full-training reference; unsuitable for clean unseen-sign claims",
                "checkpoint_sha256": file_sha256(args.checkpoint), "vocabulary_sha256": fingerprint(vocabulary),
                "upstream_source_sha256": source_hashes(args.source), "preprocessing": PREPROCESSING,
                "local_source_sha256": {p: file_sha256(code_root / p) for p in
                                        ("models/backbones/corrnet.py", "data/transforms.py")},
                "torch_version": str(torch.__version__), "opencv_version": cv2.__version__,
                "device_type": device.type, "dtype": "float32", "extraction_point": "TemporalConv.visual_feat before BiLSTM"}
    writer = CacheWriter(output, producer, records, resume=args.resume)
    references, hypotheses, checks = [], [], []
    with torch.inference_mode():
        for i, record in enumerate(records):
            existing = writer.existing(record)
            if existing is None or i < args.verify_first:
                video, lengths, frame_digest = load_corrnet_video(root, record)
                result = model(video.to(device), lengths)
                valid_length = int(result["lengths"][0])
                features = result["features"][0, :valid_length]
                ids = greedy_decode(result["logits"], result["lengths"])[0]
                predicted = [vocabulary["id_to_gloss"][j] for j in ids]
                if existing is None:
                    existing = writer.add(record, features,
                                          {"source_frames_sha256": frame_digest, "reference_prediction": predicted})
                else:
                    if existing["extra"]["source_frames_sha256"] != frame_digest:
                        raise ValueError("Source video changed since the cache was written")
                    existing = writer.add(record)
                if i < args.verify_first:
                    cached = model.from_features(existing["features"].unsqueeze(0).to(device), result["lengths"])
                    checks.append({"id": record["id"], **verify_equivalence(result["logits"], cached, result["lengths"])})
                del video, result, features
            else:
                existing = writer.add(record)
            references.append(record["glosses"])
            hypotheses.append(existing["extra"]["reference_prediction"])
            if i == 0 or (i+1) % 10 == 0 or i+1 == len(records):
                print(f"Cached {i+1}/{len(records)} videos: {record['id']}", flush=True)
    write_json(output / "reference_report.json", {"model": "released CorrNet", "decoder": "greedy",
               "scoring": "raw annotation glosses; no official sclite cleanup", "subset": args.limit is not None,
               "published_wer_reproduction": False, "equivalence": checks,
               "metrics": corpus_wer(references, hypotheses),
               "predictions": [{"id": r["id"], "reference": ref, "prediction": hyp}
                               for r, ref, hyp in zip(records, references, hypotheses)]})
    writer.finish()
    print(f"Feature cache complete: {output}")
