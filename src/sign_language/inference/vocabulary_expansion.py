"""Append declared unseen candidates to trained text heads without optimizer updates."""
from copy import deepcopy
import torch
from torch.nn.functional import normalize
from ..core.provenance import fingerprint, file_sha256
from ..data.unseen_protocol import load_study, require_clean_checkpoint
from ..text.embedding_table import load_table
from ..models.factory import build_recognizer
from ..training.checkpoints import load_checkpoint, save_checkpoint


def expand_state(state, study, bundle):
    require_clean_checkpoint(state, study)
    kind = state['config'].get('head', {}).get('kind', 'a')
    if kind not in {'b', 'c', 'b_rand', 'b_shuf'}:
        raise ValueError('Only B/C and text controls support vocabulary expansion; A remains seen-only')
    metadata = state.get('embedding_metadata')
    seen_count = len(state['vocabulary']['id_to_gloss'])
    if bundle['vocabulary'] != study['expanded_vocabulary']:
        raise ValueError('Expanded bundle must use the frozen seen-plus-held-out candidate list')
    if (not metadata or metadata['source'] != bundle['source'] or metadata['strip_prefixes'] != bundle['strip_prefixes']
            or fingerprint(bundle['vectors'][:seen_count].tolist()) != metadata['raw_vectors_sha256']):
        raise ValueError('Expanded text source, normalization or seen vectors differ from training')
    old_vectors = state['model']['head.text_vectors']
    new_vectors = bundle['vectors'][seen_count:].clone()
    seed = state['config']['head'].get('control_seed', 42) + 1
    generator = torch.Generator().manual_seed(seed)
    if kind == 'b_rand':
        new_vectors = normalize(torch.randn(new_vectors.shape, generator=generator), dim=-1)
    elif kind == 'b_shuf':
        new_vectors = new_vectors[torch.randperm(len(new_vectors), generator=generator)]
    weights = {key: value.clone() for key, value in state['model'].items()}
    weights['head.text_vectors'] = torch.cat((old_vectors, new_vectors))
    expanded = {'purpose': 'inference-only-expanded', 'model': weights, 'config': deepcopy(state['config']),
                'vocabulary': study['expanded_vocabulary'], 'feature_producer': state['feature_producer'],
                'provenance': state['provenance'], 'embedding_metadata': metadata,
                'expansion': {'study_sha256': study['sha256'], 'seen_class_count': seen_count,
                              'added_labels': study['held_out'], 'optimizer_updates': 0,
                              'control_rule': 'seen rows unchanged; new rows randomized/shuffled separately',
                              'new_row_control_seed': seed if kind in {'b_rand', 'b_shuf'} else None}}
    # Check architecture compatibility now; it remains entirely self-contained.
    build_recognizer(expanded['vocabulary'], expanded['config'], weights)
    return expanded


def expand_vocabulary(args):
    if args.output.exists():
        raise ValueError('Expanded checkpoint exists; choose a new output')
    study = load_study(args.study)
    bundle = load_table(args.embeddings, study['expanded_vocabulary'])
    expanded = expand_state(load_checkpoint(args.checkpoint), study, bundle)
    expanded['expansion']['source_checkpoint_sha256'] = file_sha256(args.checkpoint)
    expanded['expansion']['embedding_bundle_sha256'] = file_sha256(args.embeddings)
    save_checkpoint(args.output, expanded)
    print(f"Added {len(study['held_out'])} candidate labels without training: {args.output}")
