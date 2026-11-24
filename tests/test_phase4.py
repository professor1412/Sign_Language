"""Leakage boundaries, inference-only expansion and clean training orchestration."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from sign_language.data.unseen_protocol import make_study, load_study, clean_producer, require_clean_training
from sign_language.core.settings import write_json, read_json
from sign_language.core.provenance import fingerprint
from sign_language.core.reproducibility import seed_everything
from sign_language.text.embedding_table import build_table
from sign_language.features.cache import CacheWriter, atomic_torch_save
from sign_language.data.dataset import FeatureDataset
from sign_language.training.trainer import fit, TrainingConfig
from sign_language.training.clean_backbone import fit_clean, BackboneConfig
from sign_language.training.checkpoints import load_checkpoint, save_checkpoint
from sign_language.models.factory import build_recognizer
from sign_language.inference.vocabulary_expansion import expand_state, expand_vocabulary
from sign_language.evaluation.unseen import unseen_report, evaluate_unseen


class ToyBackbone(nn.Module):
    """Small differentiable substitute; explicitly not CorrNet or real video data."""
    def __init__(self, classes, dimension=8):
        super().__init__()
        self.encoder = nn.Linear(8, dimension)
        self.classifier = nn.Linear(dimension, classes)
        self.drop = nn.Dropout(.1)

    def from_features(self, features, lengths):
        return self.classifier(self.drop(features))

    def forward(self, video, lengths):
        features = self.encoder(video)
        return {'features': features, 'lengths': lengths, 'logits': self.from_features(features, lengths),
                'conv_logits': self.classifier(features)}


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        def row(split, i, glosses):
            return {'id': f'{split}-{i}', 'split': split, 'glosses': glosses}
        self.study = make_study(
            [row('train', 0, ['A', 'B']), row('train', 1, ['__ON__', 'A']),
             row('train', 2, ['C', 'INCIDENTAL']), row('train', 3, ['D'])],
            [row('dev', 0, ['A', 'B']), row('dev', 1, ['C']), row('dev', 2, ['INCIDENTAL']), row('dev', 3, ['NATURAL'])],
            [row('test', 0, ['A']), row('test', 1, ['D']), row('test', 2, ['NATURAL'])], ['C', 'D'])
        self.vocab = self.study['seen_vocabulary']
        self.study_path = self.root / 'study.json'
        write_json(self.study_path, self.study)
        self.producer = clean_producer(self.study, 'fake-checkpoint-fixture', {}, {}, {'kind': 'test-fixture'})
        self.vector = lambda word: torch.nn.functional.one_hot(torch.tensor(sum(map(ord, word)) % 300), 300).float()
        self.table = build_table(self.vocab, self.vector, {'kind': 'synthetic_debug'})
        self.expanded = build_table(self.study['expanded_vocabulary'], self.vector, {'kind': 'synthetic_debug'})
        self.embedding_path = self.root / 'embeddings.pt'
        atomic_torch_save(self.table, self.embedding_path)

    def tearDown(self):
        self.tmp.cleanup()

    def loader(self, row):
        generator = torch.Generator().manual_seed(sum(map(ord, row['id'])))
        return torch.randn(1, 7, 8, generator=generator), torch.tensor([7]), fingerprint(row)

    def datasets(self):
        for split in ('train', 'dev_seen', 'dev', 'test'):
            writer = CacheWriter(self.root / split, {**self.producer, 'feature_dim': 8}, self.study['manifests'][split])
            for row in self.study['manifests'][split]:
                writer.add(row, self.loader(row)[0][0])
            writer.finish()
        return (FeatureDataset(self.root / 'train', self.vocab, training=True),
                FeatureDataset(self.root / 'dev_seen', self.vocab))

    def config(self, kind):
        return TrainingConfig(epochs=2, batch_size=2, head={'kind': kind, 'embeddings': str(self.embedding_path)},
                              model={'input_dim': 8, 'width': 12, 'hidden_size': 6, 'layers': 2, 'dropout': .1})

    def test_study_freezes_ids_and_excludes_entire_clips(self):
        self.assertEqual(load_study(self.study_path), self.study)
        self.assertEqual(len(self.study['manifests']['train']), 2)
        self.assertEqual(self.study['incidentally_unseen'], ['INCIDENTAL'])
        self.assertEqual(self.study['expanded_vocabulary']['id_to_gloss'], self.vocab['id_to_gloss'] + ['C', 'D'])
        bad = {**self.study, 'held_out': ['A']}
        write_json(self.study_path, bad)
        with self.assertRaisesRegex(ValueError, 'modified'):
            load_study(self.study_path)

    def test_rejects_reference_features_and_unseen_training_candidates(self):
        with self.assertRaisesRegex(ValueError, 'never full-data'):
            require_clean_training({'kind': 'corrnet_temporalconv'}, [], [], self.vocab)
        train, dev = self.datasets()
        with self.assertRaisesRegex(ValueError, 'seen vocabulary'):
            fit(train, dev, self.study['expanded_vocabulary'], self.root / 'invalid', self.config('a'), torch.device('cpu'))
        wrong_dev = FeatureDataset(self.root / 'dev', self.vocab)
        with self.assertRaisesRegex(ValueError, 'seen-only dev'):
            fit(train, wrong_dev, self.vocab, self.root / 'invalid', self.config('a'), torch.device('cpu'))

    def test_expansion_preserves_weights_scores_and_evaluates_complete_split(self):
        train, dev = self.datasets()
        for kind in ('b', 'c', 'b_rand', 'b_shuf'):
            fit(train, dev, self.vocab, self.root / kind, self.config(kind), torch.device('cpu'))
            state = load_checkpoint(self.root / kind / 'best.pt')
            expanded = expand_state(state, self.study, self.expanded)
            original_model = build_recognizer(self.vocab, state['config'], state['model']).eval()
            expanded_model = build_recognizer(expanded['vocabulary'], expanded['config'], expanded['model']).eval()
            for name, value in original_model.named_parameters():
                torch.testing.assert_close(value, dict(expanded_model.named_parameters())[name], atol=0, rtol=0)
            x, lengths, _ = self.loader(self.study['manifests']['train'][0])
            with torch.inference_mode():
                torch.testing.assert_close(original_model(x, lengths), expanded_model(x, lengths)[..., :len(self.vocab['id_to_gloss'])], atol=1e-6, rtol=1e-6)
            self.assertNotIn('optimizer', expanded)
            path = self.root / kind / 'expanded.pt'
            save_checkpoint(path, expanded)
            output = self.root / kind / 'unseen.json'
            args = SimpleNamespace(study=self.study_path, checkpoint=path, cache=self.root / 'test', split='test',
                                   output=output, device='cpu', batch_size=2)
            evaluate_unseen(args)
            report = read_json(output)
            self.assertEqual(report['unseen']['label_groups']['natural_oov']['reference_occurrences'], 1)
            self.assertEqual(report['metrics']['videos'], 3)
            args.output = self.root / kind / 'wrong.json'
            args.split = 'dev'
            with self.assertRaisesRegex(ValueError, 'complete frozen split'):
                evaluate_unseen(args)
        bad_bundle = {**self.expanded, 'source': {'kind': 'wrong-source'}}
        with self.assertRaisesRegex(ValueError, 'source'):
            expand_state(state, self.study, bad_bundle)

    def test_precision_recall_counts_false_guesses_and_missing_support(self):
        predictions = [{'reference': ['A', 'C', 'C'], 'prediction': ['A', 'C', 'D']},
                       {'reference': ['A'], 'prediction': ['A', 'C']},
                       {'reference': ['INCIDENTAL', 'NATURAL'], 'prediction': []}]
        report = unseen_report(predictions, self.study)
        held = report['label_groups']['held_out']
        self.assertEqual((held['correct'], held['reference_occurrences'], held['predicted_occurrences']), (1, 2, 3))
        self.assertAlmostEqual(held['precision'], 1/3)
        self.assertEqual(held['recall'], .5)
        self.assertEqual(report['false_unseen_predictions_on_seen_only_clips'], 1)
        self.assertIsNone(report['held_out_per_gloss']['D']['recall'])
        empty = unseen_report([{'reference': ['A'], 'prediction': []}], self.study)
        self.assertIsNone(empty['label_groups']['held_out']['precision'])

    def test_backbone_resume_and_changed_frames(self):
        config = BackboneConfig(epochs=2, accumulation_steps=3)
        def train(path, **kwargs):
            seed_everything(42)
            return fit_clean(ToyBackbone(len(self.vocab['id_to_gloss'])), self.study, self.loader,
                             path, config, torch.device('cpu'), {'fixture': True}, **kwargs)
        train(self.root / 'full')
        train(self.root / 'resume', stop_after_epochs=1)
        train(self.root / 'resume', resume=True)
        full, resumed = [load_checkpoint(self.root / p / 'last.pt') for p in ('full', 'resume')]
        self.assertEqual(full['history'], resumed['history'])
        for key in full['model']:
            torch.testing.assert_close(full['model'][key], resumed['model'][key], atol=0, rtol=0)
        train(self.root / 'changed', stop_after_epochs=1)
        with self.assertRaisesRegex(ValueError, 'frames changed'):
            fit_clean(ToyBackbone(len(self.vocab['id_to_gloss'])), self.study,
                      lambda r: (*self.loader(r)[:2], 'different'), self.root / 'changed', config,
                      torch.device('cpu'), {'fixture': True}, resume=True)

    def test_clean_extraction_cache_roundtrip_and_resume(self):
        from sign_language.features.clean_extractor import extract_clean
        from sign_language.features.cache import FeatureCache
        model = ToyBackbone(len(self.vocab['id_to_gloss']), 1024).eval()
        checkpoint = self.root / 'fake.pt'
        checkpoint.write_bytes(b'unit-test stub checkpoint')
        state = {'provenance': {'upstream_source_sha256': {}, 'implementation_sha256': {}, 'initialization': {'kind': 'fixture'}},
                 'frame_digests': {}}
        args = SimpleNamespace(study=self.study_path, dataset_root=self.root / 'pngs', output=self.root / 'cache',
                               source=self.root, checkpoint=checkpoint, split='dev_seen', device='cpu', resume=False)
        with patch('sign_language.features.clean_extractor.load_clean', return_value=(model, state)), \
             patch('sign_language.features.clean_extractor.load_corrnet_video', side_effect=lambda _, row: self.loader(row)):
            extract_clean(args)
            args.resume = True
            extract_clean(args)
        cache = FeatureCache(args.output)
        self.assertEqual(cache.load(cache.entries[0]).shape, (7, 1024))
        self.assertEqual(cache.metadata['producer']['study_sha256'], self.study['sha256'])

    def test_generic_loader_rejects_sign_checkpoint(self):
        from sign_language.models.backbones.clean_corrnet import initialize_generic
        path = self.root / 'wrong.pt'
        path.write_bytes(b'not generic ImageNet weights')
        with self.assertRaisesRegex(ValueError, 'sign-pretrained'):
            initialize_generic(None, path)

    def test_generic_spatial_weights_inflate_and_new_parameters_stay_fresh(self):
        from sign_language.models.backbones.clean_corrnet import initialize_generic
        backbone = nn.Module()
        backbone.conv1 = nn.Conv3d(3, 4, (1, 3, 3), bias=False)
        backbone.bn1 = nn.BatchNorm3d(4)
        backbone.corr1 = nn.Linear(4, 4)
        backbone.alpha = nn.Parameter(torch.zeros(3))
        generic = {k: v.clone() for k, v in backbone.state_dict().items() if not k.startswith('corr') and k != 'alpha'}
        generic['conv1.weight'] = torch.ones(4, 3, 3, 3)
        generic['fc.weight'], generic['fc.bias'] = torch.zeros(1000, 4), torch.zeros(1000)
        original_corr = backbone.corr1.weight.detach().clone()
        path = self.root / 'generic.pt'
        path.write_bytes(b'mocked official generic checkpoint')
        with patch('sign_language.models.backbones.clean_corrnet.file_sha256', return_value='f37072fd-fixture'), \
             patch('sign_language.models.backbones.clean_corrnet.torch.load', return_value=generic):
            initialize_generic(SimpleNamespace(conv2d=backbone), path)
        torch.testing.assert_close(backbone.conv1.weight, torch.ones_like(backbone.conv1.weight))
        torch.testing.assert_close(backbone.corr1.weight, original_corr, atol=0, rtol=0)

    def test_seen_only_baseline_and_expansion_command(self):
        train, dev = self.datasets()
        for kind in ('a', 'b'):
            path = self.root / kind
            fit(train, dev, self.vocab, path, self.config(kind), torch.device('cpu'))
            if kind == 'a':
                state = load_checkpoint(path / 'best.pt')
                with self.assertRaisesRegex(ValueError, 'Only B/C'):
                    expand_state(state, self.study, self.expanded)
                evaluate_unseen(SimpleNamespace(study=self.study_path, checkpoint=path / 'best.pt',
                                cache=self.root / 'test', split='test', output=path / 'unseen.json', device='cpu', batch_size=2))
                self.assertEqual(read_json(path / 'unseen.json')['unseen']['label_groups']['held_out']['recall'], 0)
            else:
                bundle_path = self.root / 'expanded_embeddings.pt'
                atomic_torch_save(self.expanded, bundle_path)
                expand_vocabulary(SimpleNamespace(study=self.study_path, checkpoint=path / 'best.pt',
                                  embeddings=bundle_path, output=path / 'expanded.pt'))
                self.assertEqual(load_checkpoint(path / 'expanded.pt')['expansion']['optimizer_updates'], 0)


if __name__ == '__main__':
    unittest.main()
