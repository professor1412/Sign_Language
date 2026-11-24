"""Overfit a few real cached training clips before a full baseline run."""
from pathlib import Path
import torch
from ...core.settings import read_json, write_json
from ...core.provenance import fingerprint
from ...core.reproducibility import resolve_device, seed_everything
from ...data.dataset import FeatureDataset
from ...models.factory import build_recognizer
from ...training.trainer import read_training_config, make_loader
from ...training.loops import train_epoch, evaluate_model
from ...training.checkpoints import save_checkpoint
from dataclasses import asdict


def overfit(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Overfit output already exists; choose a new folder")
    if min(args.limit, args.max_steps) < 1:
        raise ValueError("--limit and --max-steps must be positive")
    config = read_training_config(args.config)
    vocabulary = read_json(args.vocabulary)
    dataset = FeatureDataset(args.cache, vocabulary, training=True)
    synthetic = dataset.cache.metadata["producer"]["kind"] == "synthetic_debug"
    if synthetic and not args.allow_synthetic:
        raise ValueError("For invented features only, pass --allow-synthetic")
    dataset.entries = dataset.entries[:args.limit]
    if config.model.get("input_dim", 1024) != dataset.cache.metadata["producer"]["feature_dim"]:
        raise ValueError("Model input dimension differs from feature cache")
    config.batch_size = len(dataset)
    device = resolve_device(args.device)
    seed_everything(config.seed)
    loader = make_loader(dataset, config)
    model = build_recognizer(vocabulary, asdict(config)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0)
    output.mkdir(parents=True)
    losses = []
    for step in range(args.max_steps):
        losses.append(train_epoch(model, loader, optimizer, device, config.gradient_clip))
        metrics, predictions = evaluate_model(model, loader, vocabulary, device)
        if step == 0 or (step+1) % 25 == 0 or metrics["errors"] == 0:
            print(f"Training-set overfit {step+1}/{args.max_steps}: loss={losses[-1]:.4f}, WER={metrics['wer_percent']}%", flush=True)
        if metrics["errors"] == 0:
            break
    provenance = {"producer_sha256": dataset.cache.metadata["producer_sha256"],
                  "cache_index_sha256": dataset.cache.index_sha256,
                  "subset_sha256": fingerprint([e["record"] for e in dataset.entries])}
    save_checkpoint(output / "overfit.pt", {"model": model.state_dict(), "vocabulary": vocabulary,
                    "config": asdict(config), "provenance": provenance,
                    "purpose": "training-set overfit check, not held-out accuracy"})
    write_json(output / "overfit_report.json", {"passed": metrics["errors"] == 0, "steps": step+1,
               "synthetic_debug_only": synthetic, "held_out_accuracy_measured": False,
               "weight_decay": 0, "learning_rate_schedule": "constant for this debugging check",
               "initial_loss": losses[0], "final_loss": losses[-1], "training_metrics": metrics,
               "predictions": predictions, "provenance": provenance})
    if metrics["errors"] != 0:
        raise ValueError("The small set is not fully learned yet; inspect overfit_report.json before full training")
