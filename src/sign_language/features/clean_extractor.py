"""Create separate caches from a clean, reduced-data backbone."""
import cv2
import torch
from .cache import CacheWriter
from .validation import verify_equivalence
from ..core.settings import write_json
from ..core.provenance import file_sha256
from ..core.reproducibility import resolve_device
from ..data.unseen_protocol import load_study, clean_producer
from ..data.transforms import load_corrnet_video, PREPROCESSING
from ..models.backbones.clean_corrnet import load_clean


def extract_clean(args):
    study = load_study(args.study)
    root, output = args.dataset_root.resolve(), args.output.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Cache output must be outside the dataset and its ancestors')
    device = resolve_device(args.device)
    model, state = load_clean(args.source, args.checkpoint, study, device)
    producer = clean_producer(study, file_sha256(args.checkpoint), state['provenance']['upstream_source_sha256'],
                              state['provenance']['implementation_sha256'], state['provenance']['initialization'])
    producer.update({'preprocessing': PREPROCESSING, 'torch_version': str(torch.__version__),
                     'opencv_version': cv2.__version__, 'device_type': device.type, 'dtype': 'float32'})
    records = study['manifests'][args.split]
    writer = CacheWriter(output, producer, records, args.resume)
    checks = []
    with torch.inference_mode():
        for index, row in enumerate(records):
            existing = writer.existing(row)
            if existing is None or index < 3:
                video, lengths, digest = load_corrnet_video(root, row)
                trained_digest = state['frame_digests'].get(row['split'] + '/' + row['id'])
                if trained_digest is not None and trained_digest != digest:
                    raise ValueError('Source frames differ from clean training/selection frames')
                result = model(video.to(device), lengths)
                features = result['features'][0, :int(result['lengths'][0])]
                if existing is None:
                    existing = writer.add(row, features, {'source_frames_sha256': digest})
                else:
                    if existing['extra']['source_frames_sha256'] != digest:
                        raise ValueError('Source frames changed since cached extraction')
                    existing = writer.add(row)
                if index < 3:
                    cached = model.from_features(existing['features'].unsqueeze(0).to(device), result['lengths'])
                    checks.append({'id': row['id'], **verify_equivalence(result['logits'], cached, result['lengths'])})
                del video, features, result
            else:
                writer.add(row)
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(records):
                print(f'Clean cache: {index+1}/{len(records)} clips', flush=True)
    writer.finish()
    write_json(output / 'validation.json', {'study_sha256': study['sha256'], 'split': args.split, 'equivalence': checks})
