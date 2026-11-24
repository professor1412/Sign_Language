"""Check exported inference parity, feature identity and report integrity."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from sign_language.cli.main import main
from sign_language.core.settings import read_json, write_json
from sign_language.core.provenance import fingerprint
from sign_language.data.vocabulary import build_vocabulary
from sign_language.features.cache import CacheWriter
from sign_language.models.recognizer import Recognizer
from sign_language.training.checkpoints import save_checkpoint, load_checkpoint
from sign_language.inference.predictor import predict_clip
from sign_language.evaluation.final_report import final_report
from sign_language.evaluation.wer import corpus_wer


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.vocab = build_vocabulary([{'glosses': ['A', 'B']}])
        self.producer = {'kind': 'synthetic_debug', 'feature_dim': 8}
        self.record = {'id': 'example', 'split': 'dev', 'glosses': ['A', 'B']}
        writer = CacheWriter(self.root / 'cache', self.producer, [self.record])
        writer.add(self.record, torch.randn(7, 8))
        writer.finish()

    def checkpoint(self, kind):
        config = {'model': {'input_dim': 8, 'width': 12, 'hidden_size': 6,
                            'layers': 2, 'dropout': 0.0}, 'head': {'kind': kind}}
        vectors = torch.nn.functional.normalize(torch.randn(3, 300), dim=-1)
        model = Recognizer(3, head_config=config['head'], vectors=vectors,
                           special_ids=[0], **config['model'])
        state = {'model': model.state_dict(), 'config': config, 'vocabulary': self.vocab,
                 'feature_producer': self.producer,
                 'provenance': {'producer_sha256': fingerprint(self.producer)},
                 'optimizer': {'training_only': True}}
        path = self.root / f'{kind}.pt'
        save_checkpoint(path, state)
        return path

    def test_export_and_predict_all_heads(self):
        for kind in ('a', 'b', 'c'):
            source = self.checkpoint(kind)
            before = predict_clip(load_checkpoint(source), self.root / 'cache', 'example', torch.device('cpu'))
            exported = self.root / f'{kind}-export.pt'
            main(['export-model', '--checkpoint', str(source), '--output', str(exported)])
            restored = load_checkpoint(exported)
            self.assertNotIn('optimizer', restored)
            for name, weight in load_checkpoint(source)['model'].items():
                torch.testing.assert_close(weight, restored['model'][name], rtol=0, atol=0)
            output = self.root / f'{kind}-prediction.json'
            main(['predict', '--checkpoint', str(exported), '--cache', str(self.root / 'cache'),
                  '--clip-id', 'example', '--device', 'cpu', '--output', str(output)])
            self.assertEqual(before['prediction'], read_json(output)['prediction'])
            self.assertTrue(read_json(output)['synthetic_debug'])

    def test_predict_rejects_wrong_producer_and_missing_clip(self):
        state = load_checkpoint(self.checkpoint('a'))
        with self.assertRaisesRegex(ValueError, 'clip ID'):
            predict_clip(state, self.root / 'cache', 'missing', torch.device('cpu'))
        state['provenance']['producer_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'different backbone'):
            predict_clip(state, self.root / 'cache', 'example', torch.device('cpu'))

    def test_older_checkpoint_export_keeps_producer_hash(self):
        path = self.checkpoint('a')
        state = load_checkpoint(path)
        del state['feature_producer']
        save_checkpoint(path, state)
        exported = self.root / 'legacy-export.pt'
        main(['export-model', '--checkpoint', str(path), '--output', str(exported)])
        result = predict_clip(load_checkpoint(exported), self.root / 'cache', 'example', torch.device('cpu'))
        self.assertEqual(result['producer_sha256'], fingerprint(self.producer))

    def evaluation(self):
        predictions = [{'id': 'one', 'reference': ['A', 'B'], 'prediction': ['A']},
                       {'id': 'two', 'reference': ['B'], 'prediction': ['B']}]
        return {'predictions': predictions, 'metrics': corpus_wer([r['reference'] for r in predictions],
                                                                 [r['prediction'] for r in predictions]),
                'synthetic_debug': True, 'head': 'a', 'split': 'dev'}

    def test_report_preserves_errors_and_synthetic_status(self):
        path = self.root / 'evaluation.json'
        write_json(path, self.evaluation())
        main(['final-report', '--evaluations', str(path), '--output', str(self.root / 'report')])
        result = read_json(self.root / 'report/results.json')['evaluations'][0]
        self.assertEqual(result['status'], 'synthetic diagnostic')
        self.assertEqual(result['metrics']['deletions'], 1)
        self.assertIn({'reference': 'B', 'prediction': None, 'count': 1}, result['confusions'])
        self.assertEqual(result['correct_examples'][0]['id'], 'two')
        self.assertTrue((self.root / 'report/results.csv').is_file())

    def test_report_rejects_wrong_metrics_without_publishing(self):
        report = self.evaluation()
        report['metrics']['wer_percent'] = 0
        path = self.root / 'wrong.json'
        write_json(path, report)
        output = self.root / 'bad-report'
        with self.assertRaisesRegex(ValueError, 'differs'):
            final_report(SimpleNamespace(evaluations=[path], output=output))
        self.assertFalse(output.exists())
