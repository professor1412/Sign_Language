"""Train A/B/C and controls with dev selection and epoch-boundary resume."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import platform
import torch
import yaml
from torch.utils.data import DataLoader
from .checkpoints import load_checkpoint, save_checkpoint
from .loops import train_epoch, evaluate_model
from ..core.settings import read_json, write_json
from ..core.provenance import fingerprint, file_sha256
from ..core.reproducibility import seed_everything, resolve_device, rng_state, restore_rng
from ..data.dataset import FeatureDataset
from ..data.collate import collate_features
from ..models.factory import build_recognizer, HEADS
from ..data.unseen_protocol import require_clean_training
from ..text.embedding_table import load_table


@dataclass
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    gradient_clip: float = 5.0
    seed: int = 42
    num_workers: int = 0
    head: dict = field(default_factory=dict)
    model: dict = field(default_factory=lambda: {"input_dim": 1024, "width": 512, "hidden_size": 256,
                                                "layers": 2, "dropout": 0.1})

    def validate(self):
        if not isinstance(self.head, dict) or self.head.get("kind", "a") not in HEADS:
            raise ValueError("Unknown head kind")
        if set(self.head) - {"kind", "embeddings", "temperature", "learn_temperature", "projection_dim", "control_seed"}:
            raise ValueError("Unknown head settings")
        if any(type(v) is not int for v in (self.epochs, self.batch_size, self.num_workers, self.seed)):
            raise ValueError("Epochs, batch size, workers and seed must be integers")
        if min(self.epochs, self.batch_size) < 1 or self.num_workers < 0:
            raise ValueError("Epochs/batch size must be positive and workers nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        allowed = {"input_dim", "width", "hidden_size", "layers", "dropout"}
        if not isinstance(self.model, dict) or set(self.model) - allowed:
            raise ValueError("Unknown model configuration fields")
        return self


def read_training_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Training config must be a YAML mapping")
    try:
        return TrainingConfig(**value).validate()
    except TypeError as error:
        raise ValueError(f"Unknown training configuration key: {error}") from error


def make_loader(dataset, config, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, collate_fn=collate_features, generator=generator)


def fit(train_data, dev_data, vocabulary, output, config, device, resume=False, stop_after_epochs=None):
    config.validate()
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError("--stop-after-epochs must be positive")
    if any(e["record"]["split"] != "dev" for e in dev_data.entries):
        raise ValueError("Model selection requires a dev cache; test data cannot select a checkpoint")
    if any(e["record"]["split"] != "train" for e in train_data.entries):
        raise ValueError("Training requires a train cache")
    train_ids = {e["record"]["id"] for e in train_data.entries}
    if train_ids.intersection(e["record"]["id"] for e in dev_data.entries):
        raise ValueError("Train/dev video IDs overlap")
    producer = train_data.cache.metadata["producer_sha256"]
    if producer != dev_data.cache.metadata["producer_sha256"]:
        raise ValueError("Train/dev features were made with different extraction settings/checkpoints")
    if train_data.cache.metadata['producer']['kind'] == 'corrnet_clean':
        require_clean_training(train_data.cache.metadata['producer'], [e['record'] for e in train_data.entries],
                               [e['record'] for e in dev_data.entries], vocabulary)
    if config.model.get("input_dim", 1024) != train_data.cache.metadata["producer"]["feature_dim"]:
        raise ValueError("Model input dimension differs from cached features")
    provenance = {"train_cache_index_sha256": train_data.cache.index_sha256,
                  "dev_cache_index_sha256": dev_data.cache.index_sha256,
                  "train_subset_sha256": fingerprint([e["record"] for e in train_data.entries]),
                  "dev_subset_sha256": fingerprint([e["record"] for e in dev_data.entries]),
                  "vocabulary_sha256": fingerprint(vocabulary), "producer_sha256": producer}
    source_root = Path(__file__).parents[1]
    provenance["implementation_sha256"] = {name: file_sha256(source_root / name) for name in (
        "training/trainer.py", "training/loops.py", "training/checkpoints.py", "models/recognizer.py",
        "models/temporal/encoder.py", "models/heads/linear.py", "losses/ctc.py", "decoding/greedy.py",
        "evaluation/wer.py", "data/dataset.py", "data/collate.py", "core/reproducibility.py")}
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("Run folder exists; choose a new --output or use --resume")
    if resume and not (output / "last.pt").is_file():
        raise ValueError("Cannot resume: last.pt is missing")
    generator = seed_everything(config.seed)
    train_loader = make_loader(train_data, config, True, generator)
    dev_loader = make_loader(dev_data, config)
    resume_state = load_checkpoint(output / "last.pt") if resume else None
    model = build_recognizer(vocabulary, asdict(config), resume_state["model"] if resume else None).to(device)
    embedding_metadata = resume_state.get('embedding_metadata') if resume else None
    if config.head.get('kind', 'a') in {'b', 'c', 'b_rand', 'b_shuf'} and not resume:
        bundle = load_table(config.head['embeddings'], vocabulary)
        embedding_metadata = {'source': bundle['source'], 'strip_prefixes': bundle['strip_prefixes'],
                              'raw_vectors_sha256': fingerprint(bundle['vectors'].tolist())}
    if config.head.get("embeddings") and not resume:
        provenance["embeddings_sha256"] = file_sha256(config.head["embeddings"])
    if resume and "embeddings_sha256" in resume_state["provenance"]:
        provenance["embeddings_sha256"] = resume_state["provenance"]["embeddings_sha256"]
        if Path(config.head["embeddings"]).exists() and file_sha256(config.head["embeddings"]) != provenance["embeddings_sha256"]:
            raise ValueError("Embedding bundle changed since training")
    for name in ("models/factory.py", "models/heads/text_similarity.py", "models/heads/projected_text.py", "models/heads/controls.py", "models/heads/normalized_linear.py"):
        provenance["implementation_sha256"][name] = file_sha256(source_root / name)
    for name in ('data/unseen_protocol.py', 'text/embedding_table.py', 'text/normalization.py'):
        provenance['implementation_sha256'][name] = file_sha256(source_root / name)
    # Use the same dropout/shuffling random stream after each head's initialization.
    # Encoder initialization happens first and is identical for the same seed.
    seed_everything(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    history, start_epoch, best = [], 0, None
    environment = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device)}
    if resume:
        state = resume_state
        if state["config"] != asdict(config) or state["provenance"] != provenance or state["environment"] != environment:
            raise ValueError("Resume settings, environment, vocabulary or data changed; start a new run")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], generator)
        history, start_epoch, best = state["history"], state["epoch"], state["best_dev_wer"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "run.json", {"config": asdict(config), "provenance": provenance, "environment": environment,
               "vocabulary": vocabulary, "feature_producer": train_data.cache.metadata["producer"],
               "decoder": "greedy", "scoring": "raw-gloss corpus WER", "checkpoint_selection": "lowest dev WER",
               "head_parameters": sum(p.numel() for p in model.head.parameters() if p.requires_grad),
               "total_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
    end_epoch = min(config.epochs, start_epoch + stop_after_epochs) if stop_after_epochs else config.epochs
    for epoch in range(start_epoch, end_epoch):
        learning_rate = optimizer.param_groups[0]["lr"]
        loss = train_epoch(model, train_loader, optimizer, device, config.gradient_clip)
        metrics, predictions = evaluate_model(model, dev_loader, vocabulary, device)
        score = metrics["wer_percent"]
        if score is None:
            raise ValueError("Development references contain no glosses; WER is undefined")
        improved = best is None or score < best
        if improved:
            best = score
        scheduler.step()
        history.append({"epoch": epoch+1, "train_ctc_loss": loss, "learning_rate": learning_rate, "dev": metrics})
        if hasattr(model.head, "temperature"):
            history[-1]["temperature"] = float(model.head.temperature.detach())
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "rng": rng_state(generator), "epoch": epoch+1, "best_dev_wer": best, "config": asdict(config),
                 "provenance": provenance, "environment": environment, "vocabulary": vocabulary, "history": history}
        state['feature_producer'] = train_data.cache.metadata['producer']
        state['embedding_metadata'] = embedding_metadata
        if improved:
            save_checkpoint(output / "best.pt", state)
            write_json(output / "best_dev_predictions.json", predictions)
        save_checkpoint(output / "last.pt", state)
        write_json(output / "history.json", history)
        print(f"Epoch {epoch+1}/{config.epochs}: CTC loss={loss:.4f}, dev WER={score:.2f}%", flush=True)
    return history


def train_command(args):
    config = read_training_config(args.config)
    if getattr(args, "seed", None) is not None:
        config.seed = args.seed
    vocabulary = read_json(args.vocabulary)
    train_data = FeatureDataset(args.train_cache, vocabulary, args.train_manifest, training=True)
    dev_data = FeatureDataset(args.dev_cache, vocabulary, args.dev_manifest)
    if train_data.cache.metadata["producer"]["kind"] == "synthetic_debug" and not args.allow_synthetic:
        raise ValueError("Synthetic features are only for debugging; use --allow-synthetic explicitly")
    fit(train_data, dev_data, vocabulary, args.output, config, resolve_device(args.device), args.resume, args.stop_after_epochs)
