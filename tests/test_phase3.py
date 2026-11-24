"""Phase 3 gradients, label identity, controls and resumable text heads."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import asdict
import torch
from sign_language.models.factory import build_recognizer
from sign_language.text.embedding_table import build_table, load_table
from sign_language.text.normalization import normalize_gloss
from sign_language.data.vocabulary import build_vocabulary
from sign_language.features.cache import CacheWriter, atomic_torch_save
from sign_language.data.dataset import FeatureDataset
from sign_language.training.trainer import fit, TrainingConfig
from sign_language.training.checkpoints import load_checkpoint
from sign_language.losses.ctc import ctc_loss


class Phase3Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.vocab = build_vocabulary([{"glosses": ["A", "B", "__ON__"]}])
        self.table = build_table(self.vocab, lambda text: torch.arange(1, 301).float() if text == "a" else torch.arange(300, 0, -1).float(), {"kind": "synthetic_debug"})
        self.path = self.root / "embeddings.pt"
        atomic_torch_save(self.table, self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, kind):
        return TrainingConfig(epochs=2, batch_size=2, model={"input_dim": 8, "width": 12, "hidden_size": 6, "layers": 2, "dropout": .1},
                              head={"kind": kind, "embeddings": str(self.path)})

    def test_normalization_and_identity(self):
        self.assertEqual(normalize_gloss("loc-WETTER"), "loc-wetter")
        self.assertEqual(normalize_gloss("loc-WETTER", True), "wetter")
        self.assertEqual(load_table(self.path, self.vocab)["special_ids"], [0, 3])
        changed = dict(self.vocab, id_to_gloss=["<blank>", "B", "A", "__ON__"])
        with self.assertRaises(ValueError):
            load_table(self.path, changed)
        with self.assertRaisesRegex(ValueError, "zero"):
            build_table(self.vocab, lambda _: torch.zeros(300), {})

    def test_all_heads_gradients_and_checkpoint_roundtrip(self):
        initial_encoder = None
        for kind in ("a", "a_norm", "b", "c", "b_rand", "b_shuf"):
            torch.manual_seed(42)
            config = asdict(self.config(kind))
            model = build_recognizer(self.vocab, config)
            if initial_encoder is None:
                initial_encoder = {k: v.clone() for k, v in model.encoder.state_dict().items()}
            for k, v in model.encoder.state_dict().items():
                torch.testing.assert_close(v, initial_encoder[k], atol=0, rtol=0)
            features, lengths = torch.randn(2, 7, 8), torch.tensor([7, 5])
            logits = model(features, lengths)
            ctc_loss(logits, lengths, torch.tensor([1, 1, 2]), torch.tensor([2, 1])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if kind.startswith("b") or kind == "c":
                self.assertIsNone(model.head.text_vectors.grad)
                self.assertGreater(float(model.head.special_vectors.grad.abs().sum()), 0)
                if kind == "c":
                    self.assertGreater(float(model.head.text_projection.weight.grad.abs().sum()), 0)
            model.eval()
            atomic_torch_save(model.state_dict(), self.root / "state.pt")
            state = torch.load(self.root / "state.pt", weights_only=True)
            config["head"]["embeddings"] = "missing-source-file.pt"
            restored = build_recognizer(self.vocab, config, state).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(features, lengths), restored(features, lengths), atol=0, rtol=0)

    def test_cosine_formula_special_bypass_and_controls(self):
        from sign_language.models.heads.controls import control_vectors
        vectors = self.table["vectors"]
        random = control_vectors(vectors, [0, 3], "b_rand", 7)
        torch.testing.assert_close(random, control_vectors(vectors, [0, 3], "b_rand", 7))
        torch.testing.assert_close(random[[0, 3]], vectors[[0, 3]])
        for kind in ("b", "c"):
            model = build_recognizer(self.vocab, asdict(self.config(kind)))
            head = model.head
            x = torch.randn(2, 4, 12)
            table = head.text_projection(head.text_vectors).clone()
            table[head.special_ids] = head.special_vectors
            expected = torch.nn.functional.normalize(head.video_projection(x), dim=-1) @ torch.nn.functional.normalize(table, dim=-1).T / head.temperature
            torch.testing.assert_close(head(x), expected)

    def test_text_head_resume_matches_full_training(self):
        datasets = []
        for split in ("train", "dev"):
            records = [{"id": split+str(i), "split": split, "glosses": ["A", "B"]} for i in range(2)]
            writer = CacheWriter(self.root / split, {"kind": "synthetic_debug", "feature_dim": 8}, records)
            for row in records:
                writer.add(row, torch.randn(7, 8))
            writer.finish()
            datasets.append(FeatureDataset(self.root / split, self.vocab, training=split == "train"))
        cfg = self.config("c")
        fit(*datasets, self.vocab, self.root / "full", cfg, torch.device("cpu"))
        fit(*datasets, self.vocab, self.root / "resume", cfg, torch.device("cpu"), stop_after_epochs=1)
        self.path.unlink()  # Resume is self-contained, including the frozen text rows.
        fit(*datasets, self.vocab, self.root / "resume", cfg, torch.device("cpu"), resume=True)
        a = load_checkpoint(self.root / "full/last.pt")
        b = load_checkpoint(self.root / "resume/last.pt")
        self.assertEqual(a["history"], b["history"])
        for key in a["model"]:
            torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)

    def test_recall_counts_insertions_deletions_and_unknowns(self):
        from sign_language.evaluation.analysis import recall_report
        report = recall_report([
            {"reference": ["A", "B", "UNKNOWN"], "prediction": ["A", "B"]},
            {"reference": ["A"], "prediction": ["B", "A"]},
        ], self.vocab)
        self.assertEqual(report["per_gloss"]["A"]["correct"], 2)
        self.assertEqual(report["per_gloss"]["B"]["correct"], 1)
        self.assertEqual(report["frequency_bins"]["unseen"]["recall"], 0)
        self.assertEqual(report["frequency_bins"]["1_to_10"]["recall"], 1)

    def test_comparison_rejects_unfair_runs(self):
        from types import SimpleNamespace
        from sign_language.core.settings import read_json, write_json
        from sign_language.evaluation.reporting import compare_runs
        paths = [self.root / "a", self.root / "b"]
        for kind, path in zip(("a", "b"), paths):
            config = asdict(self.config(kind))
            write_json(path / "run.json", {"config": config,
                       "provenance": {"data": "same", "embeddings_sha256": "same"},
                       "environment": {"device": "cpu"}, "decoder": "greedy", "scoring": "raw",
                       "feature_producer": {"kind": "synthetic_debug"}})
            write_json(path / "history.json", [{"epoch": i, "dev": {"wer_percent": 25.0}} for i in (1, 2)])
        args = SimpleNamespace(runs=paths, output=self.root / "comparison.json")
        compare_runs(args)
        self.assertTrue(read_json(args.output)["synthetic_debug_only"])
        args.output = self.root / "bad.json"
        run = read_json(paths[1] / "run.json")
        run["config"]["seed"] += 1
        write_json(paths[1] / "run.json", run)
        with self.assertRaisesRegex(ValueError, "same set"):
            compare_runs(args)
        run["config"]["seed"] -= 1
        write_json(paths[1] / "run.json", run)
        write_json(paths[1] / "history.json", [{"epoch": 1, "dev": {"wer_percent": 25.0}}])
        with self.assertRaisesRegex(ValueError, "epoch budget"):
            compare_runs(args)

    def test_embedding_command_writes_auditable_id_order(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from sign_language.cli.commands.build_embeddings import build_embeddings
        from sign_language.core.settings import write_json, read_json
        vocab_path, model_path = self.root / "vocab.json", self.root / "fake.bin"
        write_json(vocab_path, self.vocab)
        model_path.write_bytes(b"synthetic fixture, not a fastText model")
        args = SimpleNamespace(output=self.root / "bundle", vocabulary=vocab_path,
                               fasttext_model=model_path, strip_prefixes=False)
        model = SimpleNamespace(get_word_vector=lambda _: torch.ones(300))
        with patch("sign_language.cli.commands.build_embeddings.load_fasttext", return_value=model):
            build_embeddings(args)
        self.assertEqual(load_table(args.output / "embeddings.pt", self.vocab)["vectors"].shape, (4, 300))
        self.assertEqual(read_json(args.output / "audit.json")["rows"][1]["gloss"], "A")
        with self.assertRaisesRegex(ValueError, "exists"):
            build_embeddings(args)


if __name__ == "__main__":
    unittest.main()
