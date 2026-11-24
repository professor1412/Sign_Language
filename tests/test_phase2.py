"""Run with: python -m unittest discover -s tests -p test_phase2.py -v."""
import math
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import torch
from sign_language.core.settings import read_json, write_json
from sign_language.core.reproducibility import seed_everything
from sign_language.data.vocabulary import build_vocabulary
from sign_language.data.dataset import FeatureDataset
from sign_language.data.collate import collate_features
from sign_language.features.cache import CacheWriter, FeatureCache
from sign_language.features.validation import verify_equivalence
from sign_language.models.recognizer import Recognizer
from sign_language.losses.ctc import ctc_loss
from sign_language.decoding.greedy import collapse_ctc, greedy_decode
from sign_language.evaluation.wer import corpus_wer, edit_counts
from sign_language.training.trainer import TrainingConfig, fit
from sign_language.training.checkpoints import load_checkpoint
from sign_language.training.loops import evaluate_model
from torch.utils.data import DataLoader


class Phase2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        seed_everything(17)

    def tearDown(self):
        self.temp.cleanup()

    def cache(self, name, split, unknown=False, producer=None):
        records = [{"id": f"{split}-{i}", "split": split,
                    "glosses": ["UNSEEN"] if unknown else (["A", "A"] if i == 0 else ["B"])} for i in range(2)]
        producer = producer or {"kind": "synthetic_debug", "feature_dim": 8, "version": 1}
        path = self.root / name
        writer = CacheWriter(path, producer, records)
        for i, record in enumerate(records):
            writer.add(record, torch.randn(7+i*3, 8))
        writer.finish()
        return path, records

    def data(self):
        train, records = self.cache("train", "train")
        dev, _ = self.cache("dev", "dev")
        vocabulary = build_vocabulary(records)
        return FeatureDataset(train, vocabulary, training=True), FeatureDataset(dev, vocabulary), vocabulary

    def config(self):
        return TrainingConfig(epochs=3, batch_size=1, learning_rate=0.01, seed=42,
                              model={"input_dim": 8, "width": 12, "hidden_size": 6, "layers": 2, "dropout": 0.2})

    def test_ctc_repeat_and_blank_order(self):
        self.assertEqual(collapse_ctc([1, 1, 0, 1, 2, 2, 0]), [1, 1, 2])
        self.assertEqual(collapse_ctc([0, 0, 0]), [])
        logits = torch.nn.functional.one_hot(torch.tensor([[1, 0, 1, 2, 2]]), 3).float()
        self.assertEqual(greedy_decode(logits, [3]), [[1, 1]])

    def test_ctc_matches_hand_calculated_probability(self):
        logits = torch.zeros(1, 2, 2, requires_grad=True)
        loss = ctc_loss(logits, torch.tensor([2]), torch.tensor([1]), torch.tensor([1]))
        self.assertAlmostEqual(float(loss.detach()), -math.log(0.75), places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ctc_rejects_impossible_repeats_and_blank_targets(self):
        with self.assertRaisesRegex(ValueError, "Impossible"):
            ctc_loss(torch.zeros(1, 2, 3), torch.tensor([2]), torch.tensor([1, 1]), torch.tensor([2]))
        with self.assertRaisesRegex(ValueError, "blank"):
            ctc_loss(torch.zeros(1, 2, 3), torch.tensor([2]), torch.tensor([0]), torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "finite"):
            ctc_loss(torch.full((1, 2, 3), float("nan")), torch.tensor([2]), torch.tensor([1]), torch.tensor([1]))

    def test_wer_counts_and_corpus_weighting(self):
        self.assertEqual(edit_counts(["A", "B", "C"], ["A", "D"]),
                         {"reference_words": 3, "errors": 2, "substitutions": 1, "deletions": 1, "insertions": 0})
        result = corpus_wer([["A"], ["B"]*9], [["Z"], ["B"]*9])
        self.assertEqual(result["wer_percent"], 10)
        self.assertEqual(corpus_wer([["A"]], [["A", "B", "C", "D"]])["wer_percent"], 300)
        self.assertIsNone(corpus_wer([[]], [["A"]])["wer_percent"])

    def test_padding_cannot_change_valid_outputs(self):
        model = Recognizer(3, input_dim=8, width=12, hidden_size=6, dropout=0).eval()
        short = torch.randn(1, 7, 8)
        mixed = torch.randn(2, 12, 8) * 100
        mixed[0, :7] = short[0]
        with torch.no_grad():
            alone = model(short, torch.tensor([7]))
            batched = model(mixed, torch.tensor([7, 12]))
        torch.testing.assert_close(alone[0], batched[0, :7], atol=1e-6, rtol=1e-5)
        with self.assertRaisesRegex(ValueError, "length"):
            model(short, torch.tensor([8]))

    def test_full_sized_model_backward(self):
        model = Recognizer(4)
        x = torch.randn(2, 9, 1024)
        logits = model(x, torch.tensor([9, 7]))
        self.assertEqual(tuple(logits.shape), (2, 9, 4))
        loss = ctc_loss(logits, torch.tensor([9, 7]), torch.tensor([1, 1, 2]), torch.tensor([2, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())

    def test_feature_cache_resume_and_no_overwrite(self):
        path, records = self.cache("cache", "train")
        cache = FeatureCache(path)
        self.assertEqual(tuple(cache.load(cache.entries[0]).shape), (7, 8))
        with self.assertRaisesRegex(ValueError, "already exists"):
            CacheWriter(path, cache.metadata["producer"], records)
        writer = CacheWriter(path, cache.metadata["producer"], records, resume=True)
        for record in records:
            writer.add(record)
        writer.finish()
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            CacheWriter(path, {**cache.metadata["producer"], "version": 2}, records, resume=True)

    def test_cache_corruption_rejected(self):
        path, _ = self.cache("cache", "train")
        cache = FeatureCache(path)
        target = path / cache.entries[0]["file"]
        target.write_bytes(target.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            cache.load(cache.entries[0])
        writer = CacheWriter(path, cache.metadata["producer"], [e["record"] for e in cache.entries], resume=True)
        with self.assertRaisesRegex(ValueError, "Corrupt"):
            writer.existing(cache.entries[0]["record"])

    def test_incomplete_cache_cannot_be_used(self):
        path, _ = self.cache("cache", "train")
        index = read_json(path / "index.json")
        index["complete"] = False
        write_json(path / "index.json", index)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            FeatureCache(path)

    def test_manifest_mismatch_rejected(self):
        train, dev, vocab = self.data()
        path = self.root / "changed.jsonl"
        import json
        record = {**train.entries[0]["record"], "glosses": ["WRONG"]}
        path.write_text(json.dumps(record) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            FeatureDataset(train.cache.root, vocab, path, training=True)

    def test_unknown_dev_labels_are_scored_not_removed(self):
        train, records = self.cache("train", "train")
        dev, _ = self.cache("dev", "dev", unknown=True)
        vocabulary = build_vocabulary(records)
        dataset = FeatureDataset(dev, vocabulary)
        model = Recognizer(3, input_dim=8, width=12, hidden_size=6).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            model.head.bias.zero_()
            model.head.bias[0] = 10
        metrics, predictions = evaluate_model(model, DataLoader(dataset, batch_size=2, collate_fn=collate_features), vocabulary, torch.device("cpu"))
        self.assertEqual(metrics["reference_words"], 2)
        self.assertEqual(metrics["deletions"], 2)
        self.assertEqual(predictions[0]["reference"], ["UNSEEN"])
        with self.assertRaisesRegex(ValueError, "split=train"):
            FeatureDataset(dev, vocabulary, training=True)

    def test_feature_equivalence_checks_logits_and_predictions(self):
        logits = torch.tensor([[[0., 2.], [2., 0.]]])
        self.assertTrue(verify_equivalence(logits, logits.clone(), [2])["passed"])
        with self.assertRaises(ValueError):
            verify_equivalence(logits, logits.flip(-1), [2])

    def test_extraction_orchestration_and_resume(self):
        from sign_language.features.extractor import extract_features
        import json
        records = [{"id": f"dev-{i}", "split": "dev", "glosses": ["A"]} for i in range(2)]
        manifest = self.root / "dev.jsonl"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        vocabulary = build_vocabulary(records)
        write_json(self.root / "vocab.json", vocabulary)
        (self.root / "weights.pt").write_bytes(b"fake-checkpoint-for-mocked-backbone")
        args = SimpleNamespace(limit=None, verify_first=1, dataset_root=self.root / "raw", output=self.root / "cache",
                               manifest=manifest, vocabulary=self.root / "vocab.json", device="cpu",
                               source=self.root / "upstream", checkpoint=self.root / "weights.pt",
                               trust_legacy_checkpoint=False, resume=False)

        class FakeReference:
            def from_features(self, features, lengths):
                return features[:, :, :2]

            def __call__(self, video, lengths):
                features = torch.zeros(1, 3, 1024)
                features[0, :, 1] = 2
                return {"features": features, "lengths": torch.tensor([3]), "logits": features[:, :, :2]}

        with patch("sign_language.features.extractor.load_reference", return_value=FakeReference()), \
             patch("sign_language.features.extractor.source_hashes", return_value={"fake": "mock"}), \
             patch("sign_language.features.extractor.load_corrnet_video", return_value=(torch.zeros(1), torch.tensor([3]), "frame-hash")) as read_video:
            extract_features(args)
            self.assertEqual(read_video.call_count, 2)
            report = read_json(args.output / "reference_report.json")
            self.assertEqual(report["metrics"]["wer_percent"], 0)
            self.assertFalse(report["published_wer_reproduction"])
            args.resume = True
            extract_features(args)
            self.assertEqual(read_video.call_count, 3)  # One equivalence recheck, other video reused.
        self.assertEqual(len(FeatureCache(args.output).entries), 2)

    def test_reference_refuses_project_vocabulary(self):
        from sign_language.models.backbones.corrnet import load_reference
        vocabulary = build_vocabulary([{"glosses": ["A"]}])
        with self.assertRaisesRegex(ValueError, "released PHOENIX"):
            load_reference(self.root, self.root / "missing.pt", vocabulary, torch.device("cpu"))

    def test_overfit_saves_failure_report_without_claiming_success(self):
        from sign_language.cli.commands.overfit import overfit
        from dataclasses import asdict
        import yaml
        path, records = self.cache("train", "train")
        write_json(self.root / "vocab.json", build_vocabulary(records))
        config_path = self.root / "config.yaml"
        config_path.write_text(yaml.safe_dump(asdict(self.config())), encoding="utf-8")
        args = SimpleNamespace(output=self.root / "overfit", limit=2, max_steps=1, config=config_path,
                               vocabulary=self.root / "vocab.json", cache=path, allow_synthetic=True, device="cpu")
        with self.assertRaisesRegex(ValueError, "not fully learned"):
            overfit(args)
        report = read_json(args.output / "overfit_report.json")
        self.assertFalse(report["passed"])
        self.assertFalse(report["held_out_accuracy_measured"])
        self.assertTrue((args.output / "overfit.pt").is_file())

    def test_resume_matches_uninterrupted_training(self):
        train, dev, vocabulary = self.data()
        config = self.config()
        fit(train, dev, vocabulary, self.root / "full", config, torch.device("cpu"))
        fit(train, dev, vocabulary, self.root / "resumed", config, torch.device("cpu"), stop_after_epochs=1)
        fit(train, dev, vocabulary, self.root / "resumed", config, torch.device("cpu"), resume=True)
        full = load_checkpoint(self.root / "full/last.pt")
        resumed = load_checkpoint(self.root / "resumed/last.pt")
        self.assertEqual(full["history"], resumed["history"])
        for key, value in full["model"].items():
            torch.testing.assert_close(value, resumed["model"][key], atol=0, rtol=0)
        config.learning_rate = 0.2
        with self.assertRaisesRegex(ValueError, "Resume settings"):
            fit(train, dev, vocabulary, self.root / "resumed", config, torch.device("cpu"), resume=True)

    def test_train_test_leakage_and_cache_mixing_rejected(self):
        train, dev, vocabulary = self.data()
        test_path, _ = self.cache("test", "test")
        test = FeatureDataset(test_path, vocabulary)
        with self.assertRaisesRegex(ValueError, "dev cache"):
            fit(train, test, vocabulary, self.root / "bad", self.config(), torch.device("cpu"))
        changed_path, _ = self.cache("changed", "dev", producer={"kind": "synthetic_debug", "feature_dim": 8, "version": 2})
        changed = FeatureDataset(changed_path, vocabulary)
        with self.assertRaisesRegex(ValueError, "different extraction"):
            fit(train, changed, vocabulary, self.root / "bad", self.config(), torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
