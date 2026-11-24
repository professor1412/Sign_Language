"""Freeze a training-derived candidate set and enforce the clean study boundary."""
from copy import deepcopy
from ..core.settings import read_json, read_manifest, write_json, validate_vocabulary
from ..core.provenance import fingerprint
from .vocabulary import build_vocabulary


def make_study(train, dev, test, held_out):
    held_out = sorted(set(held_out))
    full = build_vocabulary(train)
    if not held_out or not set(held_out) <= set(full['counts']) or any(g.startswith('__') for g in held_out):
        raise ValueError('Held-out labels must be ordinary labels from training only')
    ids = set()
    for split, records in (('train', train), ('dev', dev), ('test', test)):
        if not records or any(r['split'] != split for r in records):
            raise ValueError('Invalid official split records')
        for row in records:
            if row['id'] in ids:
                raise ValueError('Duplicate or overlapping clip IDs')
            ids.add(row['id'])
    reduced = [r for r in train if set(held_out).isdisjoint(r['glosses'])]
    seen = build_vocabulary(reduced)
    seen_labels = set(seen['counts'])
    dev_seen = [r for r in dev if set(r['glosses']) <= seen_labels]
    if not reduced or not dev_seen:
        raise ValueError('Need nonempty reduced training and seen-only development data')
    expanded = deepcopy(seen)
    expanded['id_to_gloss'] += held_out
    expanded['gloss_to_id'] = {g: i for i, g in enumerate(expanded['id_to_gloss'])}
    validate_vocabulary(expanded)
    payload = {'schema_version': 1, 'purpose': 'clean-unseen-study',
               'selection': 'Phase 1 pilot held-out list; no claim of manual family balancing',
               'held_out': held_out, 'seen_vocabulary': seen, 'expanded_vocabulary': expanded,
               'reference_training_labels': sorted(full['counts']),
               'incidentally_unseen': sorted(set(full['counts']) - seen_labels - set(held_out)),
               'manifests': {'train': reduced, 'dev_seen': dev_seen, 'dev': dev, 'test': test},
               'training_rule': 'remove entire clips containing held-out labels',
               'selection_rule': 'lowest seen-only development WER',
               'candidate_rule': 'seen + blank during training; append all declared held-out labels at inference'}
    return {**payload, 'sha256': fingerprint(payload)}


def load_study(path):
    study = read_json(path)
    payload = {k: v for k, v in study.items() if k != 'sha256'}
    if study.get('purpose') != 'clean-unseen-study' or study.get('schema_version') != 1 or fingerprint(payload) != study.get('sha256'):
        raise ValueError('Invalid or modified study; create a new version')
    validate_vocabulary(study['seen_vocabulary'])
    validate_vocabulary(study['expanded_vocabulary'])
    seen = study['seen_vocabulary']
    if build_vocabulary(study['manifests']['train']) != seen:
        raise ValueError('Seen vocabulary must come only from reduced training')
    if set(study['held_out']) & set(seen['counts']):
        raise ValueError('Held-out labels leaked into training')
    if study['expanded_vocabulary']['id_to_gloss'] != seen['id_to_gloss'] + study['held_out']:
        raise ValueError('Expanded IDs must preserve all seen IDs and append held-out labels')
    expected_dev = [r for r in study['manifests']['dev'] if set(r['glosses']) <= set(seen['counts'])]
    if study['manifests']['dev_seen'] != expected_dev or not expected_dev:
        raise ValueError('Model selection must use the complete seen-only dev subset')
    return study


def prepare_study(args):
    if args.output.exists():
        raise ValueError('Study output exists; choose a new version')
    root = args.prepared
    records = [read_manifest(root / f'manifests/reference/{s}.jsonl') for s in ('train', 'dev', 'test')]
    study = make_study(*records, read_json(root / 'splits/held_out.json')['held_out'])
    if (study['manifests']['train'] != read_manifest(root / 'manifests/reduced/train.jsonl')
            or study['manifests']['dev_seen'] != read_manifest(root / 'manifests/reduced/dev_seen_only.jsonl')
            or study['seen_vocabulary'] != read_json(root / 'vocabularies/reduced.json')):
        raise ValueError('Phase 1 reduced outputs do not match the declared held-out list')
    args.output.mkdir(parents=True)
    write_json(args.output / 'study.json', study)
    write_json(args.output / 'seen_vocabulary.json', study['seen_vocabulary'])
    write_json(args.output / 'expanded_vocabulary.json', study['expanded_vocabulary'])
    for name, rows in study['manifests'].items():
        import json
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f"Study frozen: {args.output}; {len(study['manifests']['train'])} train clips, {len(study['held_out'])} held-out labels")


def clean_producer(study, checkpoint_sha256, upstream, implementation, initialization):
    return {'kind': 'corrnet_clean', 'feature_dim': 1024, 'study_sha256': study['sha256'],
            'checkpoint_sha256': checkpoint_sha256, 'upstream_source_sha256': upstream,
            'implementation_sha256': implementation, 'initialization': initialization,
            'training_manifest_sha256': fingerprint(study['manifests']['train']),
            'selection_manifest_sha256': fingerprint(study['manifests']['dev_seen']),
            'seen_vocabulary_sha256': fingerprint(study['seen_vocabulary']),
            'exposure': 'generic ImageNet initialization; sign training only on frozen reduced manifest'}


def require_clean_training(producer, train_records, dev_records, vocabulary):
    if producer.get('kind') != 'corrnet_clean':
        raise ValueError('Clean study requires a clean CorrNet cache, never full-data reference features')
    if (fingerprint(train_records) != producer['training_manifest_sha256']
            or fingerprint(dev_records) != producer['selection_manifest_sha256']
            or fingerprint(vocabulary) != producer['seen_vocabulary_sha256']):
        raise ValueError('Clean head training requires exactly reduced train, seen-only dev and seen vocabulary')


def require_clean_checkpoint(state, study):
    producer = state.get('feature_producer', {})
    require_clean_training(producer, study['manifests']['train'], study['manifests']['dev_seen'], study['seen_vocabulary'])
    if (producer.get('study_sha256') != study['sha256']
            or state['provenance']['producer_sha256'] != fingerprint(producer)
            or state['vocabulary'] != study['seen_vocabulary']):
        raise ValueError('Checkpoint does not belong to this clean study')
