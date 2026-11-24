"""Optional actual CorrNet graph check using random pixels/weights, never research accuracy."""
from pathlib import Path
import argparse
import torch
from sign_language.models.backbones.clean_corrnet import CleanCorrNet
from sign_language.losses.ctc import ctc_loss
from sign_language.features.validation import verify_equivalence
from sign_language.core.reproducibility import seed_everything, resolve_device
from sign_language.core.settings import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', type=Path, default=Path('runs/phase4_checks/corrnet_graph.json'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Output already exists; choose a new file')
    seed_everything(42)
    torch.set_num_threads(2)
    device = resolve_device(args.device)
    model = CleanCorrNet(Path('external/corrnet/upstream'), 3).to(device).train()
    pixels = torch.randn(1, 16, 3, 224, 224, device=device)
    lengths = torch.tensor([16])
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    output = model(pixels, lengths)
    losses = [ctc_loss(output[key], output['lengths'], torch.tensor([1]), torch.tensor([1]))
              for key in ('logits', 'conv_logits')]
    loss = sum(losses)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
    assert model.conv2d.conv1.weight.grad.abs().sum() > 0
    optimizer.step()
    model.eval()
    with torch.inference_mode():
        result = model(pixels, lengths)
        restored = model.from_features(result['features'], result['lengths'])
        equivalence = verify_equivalence(result['logits'], restored, result['lengths'])
    write_json(args.output, {'synthetic_debug_only': True, 'real_phoenix_accuracy_measured': False,
               'initialization': 'random, no pretrained weights', 'input': '16 random 224x224 RGB frames',
               'device': str(device), 'loss': float(loss.detach()), 'backward_passed': True,
               'optimizer_step_passed': True, 'feature_shape': list(result['features'].shape),
               'cache_equivalence': equivalence})
    print(f'Actual CorrNet graph check passed: {args.output}')


if __name__ == '__main__':
    main()
