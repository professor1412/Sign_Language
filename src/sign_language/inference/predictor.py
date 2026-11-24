"""Predict a cached clip using the same feature producer as training."""
import torch
from ..core.settings import validate_vocabulary
from ..core.provenance import fingerprint
from ..features.cache import FeatureCache
from ..models.factory import build_recognizer
from ..decoding.greedy import greedy_decode


def predict_clip(state, cache_path, clip_id, device):
    validate_vocabulary(state['vocabulary'])
    cache = FeatureCache(cache_path)
    expected = state['provenance']['producer_sha256']
    if cache.metadata['producer_sha256'] != expected:
        raise ValueError('Clip features were produced by a different backbone or preprocessing setup')
    if 'feature_producer' in state and fingerprint(state['feature_producer']) != expected:
        raise ValueError('Checkpoint feature producer metadata is inconsistent')
    matches = [entry for entry in cache.entries if entry['record']['id'] == clip_id]
    if len(matches) != 1:
        raise ValueError('Choose a clip ID that occurs exactly once in this cache')
    features = cache.load(matches[0])
    model = build_recognizer(state['vocabulary'], state['config'], state['model']).to(device).eval()
    lengths = torch.tensor([len(features)])
    with torch.inference_mode():
        logits = model(features.unsqueeze(0).to(device), lengths)
        if not torch.isfinite(logits).all():
            raise ValueError('Prediction contains non-finite scores')
        ids = greedy_decode(logits, lengths)[0]
    return {'id': clip_id, 'prediction': [state['vocabulary']['id_to_gloss'][i] for i in ids],
            'decoder': 'greedy', 'device': str(device), 'feature_steps': len(features),
            'producer_sha256': expected, 'cache_index_sha256': cache.index_sha256,
            'synthetic_debug': cache.metadata['producer']['kind'] == 'synthetic_debug'}
