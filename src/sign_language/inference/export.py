"""Export a self-contained recognizer checkpoint with feature provenance."""
from copy import deepcopy
from ..core.settings import validate_vocabulary
from ..core.provenance import file_sha256, fingerprint
from ..models.factory import build_recognizer
from ..training.checkpoints import load_checkpoint, save_checkpoint


def export_model(args):
    if args.output.exists():
        raise ValueError('Export output exists; choose a new output')
    state = load_checkpoint(args.checkpoint)
    validate_vocabulary(state['vocabulary'])
    build_recognizer(state['vocabulary'], state['config'], state['model']).eval()
    expected = state['provenance']['producer_sha256']
    if 'feature_producer' in state and fingerprint(state['feature_producer']) != expected:
        raise ValueError('Checkpoint feature producer metadata is inconsistent')
    fields = ('model', 'config', 'vocabulary', 'provenance', 'feature_producer',
              'embedding_metadata', 'expansion', 'environment', 'epoch', 'best_dev_wer')
    exported = {name: deepcopy(state[name]) for name in fields if name in state}
    exported['purpose'] = 'inference-only-expanded' if 'expansion' in state else 'inference-only'
    exported['export'] = {'source_checkpoint_sha256': file_sha256(args.checkpoint),
                          'decoder': 'greedy', 'input': 'cached visual features from recorded producer'}
    save_checkpoint(args.output, exported)
    restored = load_checkpoint(args.output)
    build_recognizer(restored['vocabulary'], restored['config'], restored['model'])
    print(f'Exported recognizer: {args.output}')
