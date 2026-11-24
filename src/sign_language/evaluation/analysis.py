"""Per-gloss recall with frequency derived only from training vocabulary counts."""
from collections import Counter
from .alignment import align


def recall_report(predictions, vocabulary):
    support, correct = Counter(), Counter()
    for row in predictions:
        for target, predicted in align(row["reference"], row["prediction"]):
            if target is not None:
                support[target] += 1
                correct[target] += int(target == predicted)
    counts = vocabulary.get("counts", {})
    rows = {}
    bins = {name: {"correct": 0, "support": 0} for name in ("unseen", "1_to_10", "11_to_100", "over_100")}
    for gloss in sorted(support):
        frequency = counts.get(gloss, 0)
        group = "unseen" if frequency == 0 else "1_to_10" if frequency <= 10 else "11_to_100" if frequency <= 100 else "over_100"
        rows[gloss] = {"support": support[gloss], "correct": correct[gloss],
                       "recall": correct[gloss] / support[gloss], "training_count": frequency}
        bins[group]["correct"] += correct[gloss]
        bins[group]["support"] += support[gloss]
    for values in bins.values():
        values["recall"] = values["correct"] / values["support"] if values["support"] else None
    return {"per_gloss": rows, "frequency_bins": bins}
