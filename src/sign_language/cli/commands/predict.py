"""Command-line demo for a saved clip and recognition checkpoint."""
from ...core.settings import write_json
from ...core.provenance import file_sha256
from ...core.reproducibility import resolve_device
from ...training.checkpoints import load_checkpoint
from ...inference.predictor import predict_clip


def predict(args):
    if args.output.exists():
        raise ValueError('Prediction output exists; choose a new output')
    result = predict_clip(load_checkpoint(args.checkpoint), args.cache, args.clip_id,
                          resolve_device(args.device))
    result['checkpoint_sha256'] = file_sha256(args.checkpoint)
    write_json(args.output, result)
    print('Predicted glosses: ' + ' '.join(result['prediction']))
    print(f'Prediction saved: {args.output}')
