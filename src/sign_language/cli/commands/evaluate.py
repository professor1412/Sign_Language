"""Evaluate a saved model without modifying training state."""
from pathlib import Path
from ...core.settings import write_json
from ...core.provenance import file_sha256, fingerprint
from ...core.reproducibility import resolve_device
from ...data.dataset import FeatureDataset
from ...models.factory import build_recognizer
from ...training.checkpoints import load_checkpoint
from ...training.trainer import TrainingConfig, make_loader
from ...training.loops import evaluate_model
from ...evaluation.analysis import recall_report


def evaluate(args):
    if Path(args.output).exists():
        raise ValueError("Evaluation report exists; choose a new --output")
    state = load_checkpoint(args.checkpoint)
    dataset = FeatureDataset(args.cache, state["vocabulary"], args.manifest)
    if dataset.cache.metadata["producer_sha256"] != state["provenance"]["producer_sha256"]:
        raise ValueError("Evaluation cache uses a different feature producer from training")
    device = resolve_device(args.device)
    model = build_recognizer(state["vocabulary"], state["config"], state["model"]).to(device)
    config = TrainingConfig(**state["config"])
    config.batch_size = args.batch_size
    config.validate()
    metrics, predictions = evaluate_model(model, make_loader(dataset, config), state["vocabulary"], device)
    write_json(args.output, {"metrics": metrics, "predictions": predictions, "decoder": "greedy",
               "recall": recall_report(predictions, state["vocabulary"]),
               "scoring": "raw-gloss corpus WER", "checkpoint": str(args.checkpoint),
               "checkpoint_purpose": state.get("purpose", "dev-selected recognition training"),
               "head": state["config"].get("head", {}).get("kind", "a"),
               "split": sorted({e["record"]["split"] for e in dataset.entries}),
               "checkpoint_sha256": file_sha256(args.checkpoint),
               "evaluation_subset_sha256": fingerprint([e["record"] for e in dataset.entries]),
               "cache_index_sha256": dataset.cache.index_sha256, "synthetic_debug": dataset.cache.metadata["producer"]["kind"] == "synthetic_debug"})
    print(f"WER: {metrics['wer_percent']}%; report: {args.output}")
