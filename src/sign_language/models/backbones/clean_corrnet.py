"""CorrNet trained afresh on allowed clips; only generic ResNet18 weights may initialize it."""
from pathlib import Path
import torch
from .corrnet import CorrNetReference, source_hashes
from ...core.provenance import file_sha256
from ...training.checkpoints import load_checkpoint

IMAGENET_URL = 'https://download.pytorch.org/models/resnet18-f37072fd.pth'
IMAGENET_SHA256_PREFIX = 'f37072fd'


def implementation_hashes():
    root = Path(__file__).parents[2]
    files = ('models/backbones/corrnet.py', 'models/backbones/clean_corrnet.py',
             'models/heads/normalized_linear.py', 'training/clean_backbone.py',
             'data/unseen_protocol.py', 'data/transforms.py', 'data/frame_reader.py',
             'losses/ctc.py', 'decoding/greedy.py', 'evaluation/wer.py', 'core/reproducibility.py')
    return {p: file_sha256(root / p) for p in files}


class CleanCorrNet(CorrNetReference):
    def forward(self, video, lengths):
        result = super().forward(video, lengths)
        result['conv_logits'] = self.classifier(result['features'])
        return result


def initialize_generic(model, path):
    if not Path(path).is_file():
        raise ValueError('Generic ImageNet ResNet18 weights missing; see docs/phase4.md')
    digest = file_sha256(path)
    if not digest.startswith(IMAGENET_SHA256_PREFIX):
        raise ValueError('Expected official resnet18-f37072fd.pth; sign-pretrained weights are forbidden here')
    weights = torch.load(path, map_location='cpu', weights_only=True)
    # Inflate spatial 2-D kernels to time-size-1 3-D kernels exactly as upstream does.
    expected = {k: v for k, v in model.conv2d.state_dict().items() if not k.startswith('corr') and k != 'alpha'}
    supplied = {k: v for k, v in weights.items() if not k.startswith('fc.')}
    if set(supplied) != set(expected):
        raise ValueError('Generic checkpoint has unexpected keys')
    for key, target in expected.items():
        value = supplied[key]
        if target.ndim == 5 and value.ndim == 4:
            value = value.unsqueeze(2)
        if value.shape != target.shape:
            raise ValueError(f'Generic weight shape mismatch: {key}')
        supplied[key] = value
    model.conv2d.load_state_dict({**model.conv2d.state_dict(), **supplied}, strict=True)
    return {'kind': 'ImageNet1K-ResNet18', 'url': IMAGENET_URL, 'sha256': digest,
            'new_parameters': 'correlation modules, TemporalConv, BiLSTM and classifiers'}


def load_clean(source, checkpoint, study, device):
    state = load_checkpoint(checkpoint)
    if (state.get('purpose') != 'clean-backbone-training'
            or state['study_sha256'] != study['sha256']
            or state['vocabulary'] != study['seen_vocabulary']
            or state['provenance']['upstream_source_sha256'] != source_hashes(source)
            or state['provenance']['implementation_sha256'] != implementation_hashes()
            or state['provenance']['initialization']['kind'] != 'ImageNet1K-ResNet18'
            or not state['provenance']['initialization']['sha256'].startswith(IMAGENET_SHA256_PREFIX)):
        raise ValueError('Checkpoint is not a compatible clean backbone for this study')
    model = CleanCorrNet(source, len(study['seen_vocabulary']['id_to_gloss']))
    model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval().requires_grad_(False), state
