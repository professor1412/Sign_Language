"""Small synthetic overfit check. Never reports a real-data accuracy result."""
from pathlib import Path
import torch
from ...core.reproducibility import seed_everything, resolve_device
from ...core.settings import write_json
from ...features.cache import CacheWriter, atomic_torch_save
from ...data.vocabulary import build_vocabulary
from ...data.dataset import FeatureDataset
from ...data.collate import collate_features
from ...models.recognizer import Recognizer
from ...losses.ctc import ctc_loss
from ...decoding.greedy import greedy_decode


def synthetic_examples(dimension=1024):
    labels = [[1, 2], [2, 1], [1, 1], [2, 2]]
    features = []
    for target in labels:
        path = [0]*2 + [target[0]]*5 + [0]*3 + [target[1]]*5 + [0]*2
        x = torch.zeros(len(path), dimension)
        for t, label in enumerate(path):
            x[t, label] = 3.0
        features.append(x)
    return features, labels


def smoke_test(args):
    output = Path(args.output)
    if output.exists():
        raise ValueError("Smoke-test output exists; use a new --output")
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    seed_everything(42)
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    output.mkdir(parents=True)
    features, labels = synthetic_examples()
    names = ["<blank>", "A", "B"]
    producer = {"kind": "synthetic_debug", "feature_dim": 1024, "version": 1,
                "exposure": "invented patterns; not video features"}
    records = [{"id": f"synthetic-train-{i}", "split": "train", "glosses": [names[j] for j in target]}
               for i, target in enumerate(labels)]
    vocabulary = build_vocabulary(records)
    write_json(output / "vocabulary.json", vocabulary)
    for split in ("train", "dev"):
        rows = [{**r, "id": r["id"].replace("train", split), "split": split} for r in records]
        writer = CacheWriter(output / f"{split}_cache", producer, rows)
        for row, x in zip(rows, features):
            writer.add(row, x)
        writer.finish()
    dataset = FeatureDataset(output / "train_cache", vocabulary, training=True)
    batch = collate_features([dataset[i] for i in range(len(dataset))])
    config = {"input_dim": 1024, "width": 64, "hidden_size": 32, "layers": 2, "dropout": 0.0}
    model = Recognizer(3, **config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    x = batch["features"].to(device)
    losses, predictions = [], []
    for step in range(args.max_steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, batch["lengths"])
        loss = ctc_loss(logits, batch["lengths"], batch["targets"], batch["target_lengths"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits = model(x, batch["lengths"])
            predictions = greedy_decode(logits, batch["lengths"])
        if predictions == labels:
            break
    atomic_torch_save({"model": model.state_dict(), "config": config, "vocabulary": vocabulary}, output / "smoke_model.pt")
    restored = Recognizer(3, **config).to(device).eval()
    restored.load_state_dict(torch.load(output / "smoke_model.pt", map_location="cpu", weights_only=True)["model"])
    with torch.inference_mode():
        restored_logits = restored(x, batch["lengths"])
        torch.testing.assert_close(logits, restored_logits)
    passed = predictions == labels
    report = {"synthetic_debug_only": True, "real_phoenix_accuracy_measured": False,
              "device": str(device), "steps": step+1, "initial_loss": losses[0], "final_loss": losses[-1],
              "targets": labels, "predictions": predictions, "overfit_passed": passed,
              "checkpoint_reload_passed": True, "feature_cache_roundtrip_passed": True,
              "configuration": config}
    write_json(output / "smoke_report.json", report)
    print(f"Synthetic overfit: {passed}; {step+1} steps; report: {output / 'smoke_report.json'}")
    if not passed:
        raise ValueError("Synthetic model did not learn every sequence within the requested steps")
