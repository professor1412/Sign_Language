"""Reduced-data CorrNet: sequence CTC + convolutional CTC, seen-only dev selection."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import platform
import torch
import yaml
from .checkpoints import load_checkpoint, save_checkpoint
from ..core.settings import write_json
from ..core.provenance import fingerprint
from ..core.reproducibility import seed_everything, resolve_device, rng_state, restore_rng
from ..data.unseen_protocol import load_study
from ..data.transforms import load_corrnet_video, PREPROCESSING
from ..models.backbones.clean_corrnet import CleanCorrNet, initialize_generic, implementation_hashes
from ..models.backbones.corrnet import source_hashes
from ..losses.ctc import ctc_loss
from ..decoding.greedy import greedy_decode
from ..evaluation.wer import corpus_wer


@dataclass
class BackboneConfig:
    epochs: int = 40
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    accumulation_steps: int = 8
    gradient_clip: float = 5.0
    conv_ctc_weight: float = 1.0
    seed: int = 42

    def validate(self):
        if (any(type(v) is not int for v in (self.epochs, self.accumulation_steps, self.seed))
                or min(self.epochs, self.accumulation_steps) < 1):
            raise ValueError('Epochs and accumulation steps must be positive integers')
        if (not all(math.isfinite(v) for v in (self.learning_rate, self.weight_decay, self.gradient_clip, self.conv_ctc_weight))
                or self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0 or self.conv_ctc_weight <= 0):
            raise ValueError('Invalid backbone loss/optimizer settings')
        return self


def fit_clean(model, study, load_video, output, config, device, provenance, resume=False, stop_after_epochs=None):
    config.validate()
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError('Run exists; choose a new output or resume the same run')
    if stop_after_epochs is not None and stop_after_epochs < 1:
        raise ValueError('Stop-after-epochs must be positive')
    generator = seed_everything(config.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    vocabulary = study['seen_vocabulary']
    train, dev = study['manifests']['train'], study['manifests']['dev_seen']
    environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'device': str(device)}
    history, start, best, frame_digests = [], 0, None, {}
    if resume:
        state = load_checkpoint(output / 'last.pt')
        if (state.get('purpose') != 'clean-backbone-training' or state['study_sha256'] != study['sha256']
                or state['config'] != asdict(config) or state['provenance'] != provenance or state['environment'] != environment):
            raise ValueError('Resume study, settings, implementation or environment changed')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'], generator)
        history, start, best = state['history'], state['epoch'], state['best_dev_wer']
        frame_digests = state['frame_digests']
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'run.json', {'config': asdict(config), 'provenance': provenance, 'study_sha256': study['sha256'],
               'environment': environment, 'training_clips': len(train), 'selection_clips': len(dev),
               'loss': 'sequence CTC + conv_ctc_weight * convolutional CTC; no VAC distillation',
               'preprocessing': PREPROCESSING, 'microbatch_size': 1, 'selection': 'lowest seen-only dev raw-gloss WER'})

    def read_video(row):
        video, lengths, digest = load_video(row)
        key = row['split'] + '/' + row['id']
        if key in frame_digests and frame_digests[key] != digest:
            raise ValueError(f'Source frames changed during the experiment: {key}')
        frame_digests[key] = digest
        return video.to(device), lengths

    end = min(config.epochs, start + stop_after_epochs) if stop_after_epochs else config.epochs
    for epoch in range(start, end):
        model.train()
        order = torch.randperm(len(train), generator=generator).tolist()
        total_loss = 0.0
        for offset in range(0, len(order), config.accumulation_steps):
            group = order[offset:offset + config.accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            for index in group:
                row = train[index]
                video, lengths = read_video(row)
                result = model(video, lengths)
                targets = torch.tensor([vocabulary['gloss_to_id'][g] for g in row['glosses']], dtype=torch.long)
                target_lengths = torch.tensor([len(targets)])
                loss = ctc_loss(result['logits'], result['lengths'], targets, target_lengths)
                loss = loss + config.conv_ctc_weight * ctc_loss(result['conv_logits'], result['lengths'], targets, target_lengths)
                (loss / len(group)).backward()
                total_loss += float(loss.detach())
                del video, result, loss
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            print(f"Backbone epoch {epoch+1}/{config.epochs}: {min(offset+len(group), len(train))}/{len(train)} clips", flush=True)
        model.eval()
        predictions = []
        with torch.inference_mode():
            for row in dev:
                video, lengths = read_video(row)
                result = model(video, lengths)
                if not torch.isfinite(result['logits']).all():
                    raise ValueError('Non-finite development scores')
                ids = greedy_decode(result['logits'], result['lengths'])[0]
                predictions.append({'id': row['id'], 'reference': row['glosses'],
                                    'prediction': [vocabulary['id_to_gloss'][i] for i in ids]})
                del video, result
        metrics = corpus_wer([r['reference'] for r in predictions], [r['prediction'] for r in predictions])
        score = metrics['wer_percent']
        if score is None:
            raise ValueError('Seen-only dev contains no reference glosses')
        improved = best is None or score < best
        if improved:
            best = score
        history.append({'epoch': epoch+1, 'train_loss': total_loss / len(train), 'dev': metrics,
                        'learning_rate': optimizer.param_groups[0]['lr']})
        scheduler.step()
        state = {'purpose': 'clean-backbone-training', 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'rng': rng_state(generator), 'config': asdict(config),
                 'vocabulary': vocabulary, 'study_sha256': study['sha256'], 'provenance': provenance,
                 'environment': environment, 'epoch': epoch+1, 'history': history, 'best_dev_wer': best,
                 'frame_digests': frame_digests}
        if improved:
            save_checkpoint(output / 'best.pt', state)
            write_json(output / 'best_dev_predictions.json', predictions)
        save_checkpoint(output / 'last.pt', state)
        write_json(output / 'history.json', history)
        print(f"Backbone epoch {epoch+1}: seen-only dev WER={score:.2f}%", flush=True)
    return history


def train_clean(args):
    study = load_study(args.study)
    value = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('Backbone config must be a YAML mapping')
    try:
        config = BackboneConfig(**value).validate()
    except TypeError as error:
        raise ValueError(f'Unknown backbone settings: {error}') from error
    if args.output.exists() and not args.resume:
        raise ValueError('Backbone output exists; choose a new run or --resume')
    root = args.dataset_root.resolve()
    if args.output.resolve().is_relative_to(root) or root.is_relative_to(args.output.resolve()):
        raise ValueError('Training output must be outside the source dataset and its ancestors')
    seed_everything(config.seed)
    device = resolve_device(args.device)
    model = CleanCorrNet(args.source, len(study['seen_vocabulary']['id_to_gloss']))
    initialization = (load_checkpoint(args.output / 'last.pt')['provenance']['initialization'] if args.resume
                      else initialize_generic(model, args.imagenet_checkpoint))
    provenance = {'initialization': initialization, 'upstream_source_sha256': source_hashes(args.source),
                  'implementation_sha256': implementation_hashes(), 'dataset_root': str(root),
                  'train_manifest_sha256': fingerprint(study['manifests']['train']),
                  'dev_manifest_sha256': fingerprint(study['manifests']['dev_seen'])}
    fit_clean(model.to(device), study, lambda row: load_corrnet_video(root, row), args.output,
              config, device, provenance, args.resume, args.stop_after_epochs)
