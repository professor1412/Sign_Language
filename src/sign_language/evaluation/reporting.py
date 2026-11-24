"""Compare dev-selected checkpoints with strict data/protocol checks."""
import statistics
from collections import defaultdict
from dataclasses import asdict
from ..core.settings import read_json, write_json
from ..training.trainer import TrainingConfig


def compare_runs(args):
    if args.output.exists():
        raise ValueError("Comparison output exists")
    groups, comparison, rows = defaultdict(list), None, []
    group_settings, group_seeds, embedding_hashes = {}, defaultdict(set), set()
    seeds = set()
    for path in args.runs:
        run = read_json(path / "run.json")
        history = read_json(path / "history.json")
        config = asdict(TrainingConfig(**run["config"]))
        head = config.pop("head")
        seed = config.pop("seed")
        key = head.get("kind", "a")
        protocol = {"config": config, "data": {k: v for k, v in run["provenance"].items()
                                              if k != "embeddings_sha256"},
                    "environment": run["environment"],
                    "decoder": run["decoder"], "scoring": run["scoring"],
                    "feature_producer": run["feature_producer"]}
        if comparison is not None and protocol != comparison:
            raise ValueError("Runs use different data, temporal models or training protocols")
        comparison = protocol
        if (key, seed) in seeds:
            raise ValueError("Duplicate head/seed in comparison")
        seeds.add((key, seed))
        group_seeds[key].add(seed)
        settings = {k: v for k, v in head.items() if k != "embeddings"}
        if key in group_settings and settings != group_settings[key]:
            raise ValueError("Settings for the same head differ between seeds")
        group_settings[key] = settings
        embedding_hash = run["provenance"].get("embeddings_sha256")
        if key in {"b", "c", "b_rand", "b_shuf"}:
            if not embedding_hash:
                raise ValueError("Text run is missing embedding provenance")
            embedding_hashes.add(embedding_hash)
        if [row["epoch"] for row in history] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Finish the configured epoch budget before comparing runs")
        best = min(history, key=lambda row: row["dev"]["wer_percent"])
        rows.append({"run": str(path), "head": key, "seed": seed, "epoch": best["epoch"],
                     "dev_wer": best["dev"]["wer_percent"], "completed_epochs": len(history),
                     "head_parameters": run.get("head_parameters"), "head_config": head,
                     "embeddings_sha256": run["provenance"].get("embeddings_sha256")})
        groups[key].append(best["dev"]["wer_percent"])
    if len(embedding_hashes) > 1:
        raise ValueError("Text heads must use the same source embedding bundle")
    if len({tuple(sorted(seeds)) for seeds in group_seeds.values()}) > 1:
        raise ValueError("Each head must have the same set of training seeds")
    summary = {key: {"seeds": sorted(group_seeds[key]), "dev_wer_mean": statistics.mean(scores),
                     "dev_wer_sample_std": statistics.stdev(scores) if len(scores) > 1 else None}
               for key, scores in groups.items()}
    write_json(args.output, {"runs": rows, "summary": summary, "protocol": comparison,
                             "synthetic_debug_only": comparison["feature_producer"]["kind"] == "synthetic_debug",
                             "selection": "lowest dev WER over the full configured epoch budget"})
    print(f"Comparison saved: {args.output}")
