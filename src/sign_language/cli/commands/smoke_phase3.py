"""Exercise six heads and text-head learning with explicitly invented data."""
from dataclasses import asdict
from types import SimpleNamespace
import torch
import yaml
from .smoke import synthetic_examples
from .overfit import overfit
from .evaluate import evaluate
from ...core.settings import read_json, write_json
from ...data.vocabulary import build_vocabulary
from ...data.dataset import FeatureDataset
from ...features.cache import CacheWriter, atomic_torch_save
from ...text.embedding_table import build_table
from ...training.trainer import fit, TrainingConfig
from ...core.reproducibility import resolve_device
from ...evaluation.reporting import compare_runs


def smoke_phase3(args):
    output = args.output
    if output.exists():
        raise ValueError("Smoke-test output exists; choose a new --output")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    features, labels = synthetic_examples()
    names = ["<blank>", "A", "B"]
    records = [{"id": f"debug-train-{i}", "split": "train", "glosses": [names[j] for j in target]}
               for i, target in enumerate(labels)]
    vocabulary = build_vocabulary(records)
    output.mkdir(parents=True)
    write_json(output / "vocabulary.json", vocabulary)
    producer = {"kind": "synthetic_debug", "feature_dim": 1024,
                "exposure": "invented patterns; train and dev intentionally repeat for debugging"}
    for split in ("train", "dev"):
        rows = [{**r, "id": r["id"].replace("train", split), "split": split} for r in records]
        writer = CacheWriter(output / f"{split}_cache", producer, rows)
        for row, x in zip(rows, features):
            writer.add(row, x)
        writer.finish()
    # Orthogonal toy vectors: these are deliberately NOT semantic fastText embeddings.
    table = build_table(vocabulary, lambda text: torch.eye(2, 300)[0 if text == "a" else 1],
                        {"kind": "synthetic_debug", "description": "two invented unit vectors"})
    atomic_torch_save(table, output / "embeddings.pt")
    train = FeatureDataset(output / "train_cache", vocabulary, training=True)
    dev = FeatureDataset(output / "dev_cache", vocabulary)
    paths, checks = [], {}
    for kind in ("a", "a_norm", "b", "c", "b_rand", "b_shuf"):
        head = {"kind": kind}
        if kind not in {"a", "a_norm"}:
            head["embeddings"] = str(output / "embeddings.pt")
        config = TrainingConfig(epochs=3, batch_size=4, learning_rate=.01, weight_decay=0,
                                model={"input_dim": 1024, "width": 64, "hidden_size": 32,
                                       "layers": 2, "dropout": 0.0}, head=head)
        path = output / kind
        fit(train, dev, vocabulary, path, config, device)
        paths.append(path)
        evaluate(SimpleNamespace(checkpoint=path / "best.pt", cache=output / "dev_cache",
                                 manifest=None, output=path / "evaluation.json", batch_size=4, device=args.device))
        if kind in {"b", "c"}:
            config_path = output / f"{kind}_debug.yaml"
            config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
            overfit(SimpleNamespace(cache=output / "train_cache", vocabulary=output / "vocabulary.json",
                                    config=config_path, limit=4, max_steps=args.max_steps,
                                    output=output / f"{kind}_overfit", device=args.device, allow_synthetic=True))
            checks[kind] = read_json(output / f"{kind}_overfit/overfit_report.json")
    compare_runs(SimpleNamespace(runs=paths, output=output / "comparison.json"))
    write_json(output / "smoke_report.json", {"synthetic_debug_only": True,
               "real_phoenix_accuracy_measured": False, "device": str(device),
               "heads_trained_and_reloaded": [p.name for p in paths],
               "overfit": {k: {"passed": v["passed"], "steps": v["steps"],
                                "training_wer": v["training_metrics"]["wer_percent"]} for k, v in checks.items()}})
    print(f"Phase 3 synthetic checks passed: {output / 'smoke_report.json'}")
