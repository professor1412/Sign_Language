"""Sequence-aligned unseen precision/recall with predefined candidate and clip groups."""
from collections import Counter
from .alignment import align
from .wer import corpus_wer
from .analysis import recall_report
from ..core.settings import write_json
from ..core.provenance import file_sha256, fingerprint
from ..core.reproducibility import resolve_device
from ..data.unseen_protocol import load_study, require_clean_checkpoint
from ..data.dataset import FeatureDataset
from ..models.factory import build_recognizer
from ..training.checkpoints import load_checkpoint
from ..training.trainer import TrainingConfig, make_loader
from ..training.loops import evaluate_model


def unseen_report(predictions, study):
    seen = set(study['seen_vocabulary']['counts'])
    held = set(study['held_out'])
    incidental = set(study['incidentally_unseen'])
    original = set(study['reference_training_labels'])
    observed = {g for row in predictions for g in row['reference'] + row['prediction']}
    categories = {'seen': seen, 'held_out': held, 'incidentally_unseen': incidental, 'natural_oov': observed - original}
    refs, guesses, correct = Counter(), Counter(), Counter()
    for row in predictions:
        refs.update(row['reference'])
        guesses.update(row['prediction'])
        for truth, predicted in align(row['reference'], row['prediction']):
            if truth is not None and truth == predicted:
                correct[truth] += 1
    def counts(labels):
        support = sum(refs[g] for g in labels)
        predicted = sum(guesses[g] for g in labels)
        tp = sum(correct[g] for g in labels)
        return {'reference_occurrences': support, 'predicted_occurrences': predicted, 'correct': tp,
                'false_positives': predicted-tp, 'false_negatives': support-tp,
                'precision': tp/predicted if predicted else None, 'recall': tp/support if support else None}
    def score(rows):
        return corpus_wer([r['reference'] for r in rows], [r['prediction'] for r in rows]) if rows else {'videos': 0, 'wer_percent': None}
    groups = {'all': predictions,
              'seen_only': [r for r in predictions if set(r['reference']) <= seen],
              'contains_held_out': [r for r in predictions if held.intersection(r['reference'])],
              'contains_incidentally_unseen': [r for r in predictions if incidental.intersection(r['reference'])],
              'contains_natural_oov': [r for r in predictions if set(r['reference']) - original]}
    seen_rows = groups['seen_only']
    false_guesses = sum(sum(g in held for g in r['prediction']) for r in seen_rows)
    return {'clip_groups': {name: score(rows) for name, rows in groups.items()},
            'label_groups': {name: counts(labels) for name, labels in categories.items()},
            'held_out_per_gloss': {g: counts({g}) for g in sorted(held)},
            'false_unseen_predictions_on_seen_only_clips': false_guesses,
            'group_note': 'contains-groups may overlap; natural OOV references are retained in WER'}


def evaluate_unseen(args):
    if args.output.exists():
        raise ValueError('Evaluation output exists; choose a new output')
    study = load_study(args.study)
    state = load_checkpoint(args.checkpoint)
    expansion = state.get('expansion')
    if expansion:
        if (state.get('purpose') != 'inference-only-expanded' or expansion['study_sha256'] != study['sha256']
                or state['vocabulary'] != study['expanded_vocabulary'] or expansion['optimizer_updates'] != 0):
            raise ValueError('Invalid expanded checkpoint or candidate set')
        # The clean training proof still refers to the original seen vocabulary.
        require_clean_checkpoint({**state, 'vocabulary': study['seen_vocabulary']}, study)
    else:
        require_clean_checkpoint(state, study)
    data = FeatureDataset(args.cache, state['vocabulary'])
    if (data.cache.metadata['producer_sha256'] != state['provenance']['producer_sha256']
            or [e['record'] for e in data.entries] != study['manifests'][args.split]):
        raise ValueError('Evaluation requires the complete frozen split from the same clean backbone')
    device = resolve_device(args.device)
    model = build_recognizer(state['vocabulary'], state['config'], state['model']).to(device)
    config = TrainingConfig(**state['config'])
    config.batch_size = args.batch_size
    config.validate()
    metrics, predictions = evaluate_model(model, make_loader(data, config), state['vocabulary'], device)
    write_json(args.output, {'study_sha256': study['sha256'], 'checkpoint_sha256': file_sha256(args.checkpoint),
               'cache_index_sha256': data.cache.index_sha256, 'split': args.split, 'metrics': metrics,
               'unseen': unseen_report(predictions, study), 'recall': recall_report(predictions, study['seen_vocabulary']),
               'candidate_labels': state['vocabulary']['id_to_gloss'], 'expanded': bool(expansion),
               'head': state['config'].get('head', {}).get('kind', 'a'), 'predictions': predictions,
               'decoder': 'greedy', 'scoring': 'raw-gloss corpus WER; all reference labels retained',
               'selection_rule': 'seen-only dev selects models; held-out/test results do not tune this run'})
    print(f"{args.split} WER={metrics['wer_percent']}%; unseen report: {args.output}")
