"""Train with CTC; evaluate complete reference sequences with corpus WER."""
import torch
from ..losses.ctc import ctc_loss
from ..decoding.greedy import greedy_decode
from ..evaluation.wer import corpus_wer


def train_epoch(model, loader, optimizer, device, gradient_clip=5.0):
    model.train()
    total, examples = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch["features"].to(device), batch["lengths"])
        loss = ctc_loss(logits, batch["lengths"], batch["targets"], batch["target_lengths"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
        count = len(batch["ids"])
        total += float(loss.detach()) * count
        examples += count
    return total / examples


@torch.inference_mode()
def evaluate_model(model, loader, vocabulary, device):
    model.eval()
    references, hypotheses, predictions = [], [], []
    blank_steps, valid_steps = 0, 0
    for batch in loader:
        logits = model(batch["features"].to(device), batch["lengths"])
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite evaluation logits")
        decoded = greedy_decode(logits, batch["lengths"])
        for path, length in zip(logits.argmax(-1), batch["lengths"]):
            blank_steps += int((path[:int(length)] == 0).sum())
            valid_steps += int(length)
        for clip_id, ref, ids in zip(batch["ids"], batch["references"], decoded):
            hyp = [vocabulary["id_to_gloss"][i] for i in ids]
            references.append(ref)
            hypotheses.append(hyp)
            predictions.append({"id": clip_id, "reference": ref, "prediction": hyp})
    metrics = corpus_wer(references, hypotheses)
    metrics["blank_rate"] = blank_steps / valid_steps
    return metrics, predictions
