import csv
import tempfile
import unittest
from pathlib import Path
from sign_language.data.annotations import read_annotations
from sign_language.data.splits import split_training
from sign_language.data.vocabulary import build_vocabulary
from sign_language.data.validation import validate_disjoint
from sign_language.data.frame_reader import resolve_frames

class PhaseOneTests(unittest.TestCase):
    def rows(self):
        return [{"id": str(i), "glosses": g} for i,g in enumerate([["A","B"],["A"],["C"],["D"]])]

    def test_whole_video_exclusion_and_incidental_labels(self):
        kept, policy = split_training(self.rows(), held_out=["A"])
        self.assertEqual([r["id"] for r in kept], ["2","3"])
        self.assertEqual(policy["incidentally_unseen"], ["B"])

    def test_deterministic_and_no_target_blank(self):
        self.assertEqual(split_training(self.rows(), seed=7), split_training(self.rows(), seed=7))
        v=build_vocabulary(self.rows())
        self.assertEqual(v["gloss_to_id"]["<blank>"],0)
        self.assertFalse(v["checkpoint_compatible"])

    def test_reject_data_loss_and_unknown_labels(self):
        with self.assertRaises(ValueError):
            split_training(self.rows(), held_out=["UNKNOWN"])
        with self.assertRaises(ValueError):
            split_training(self.rows(), held_out=["A"], min_retained=0.9)

    def test_split_overlap(self):
        with self.assertRaises(ValueError):
            validate_disjoint({"train":self.rows(), "test":self.rows()})

    def test_pipe_annotations_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"a.csv"
            header="name|video|start|end|speaker|orth|translation\n"
            row="clip|clip/1/*.png|-1|-1|Signer01|A B|a sentence\n"
            p.write_text(header+row,encoding="utf-8")
            self.assertEqual(read_annotations(p)[0]["glosses"],["A","B"])
            p.write_text(header+row+row,encoding="utf-8")
            with self.assertRaises(ValueError):
                read_annotations(p)

    def test_end_to_end_bundle_and_no_overwrite(self):
        import json
        import struct
        from sign_language.cli.main import main
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"dataset"
            ann=root/"annotations/manual"
            ann.mkdir(parents=True)
            for split in ("train","dev","test"):
                rows=[]
                for i,g in enumerate(("A","B","C")):
                    name=f"{split}-{i}"
                    folder=root/"features/fullFrame-210x260px"/split/name
                    folder.mkdir(parents=True)
                    (folder/"images0001.png").write_bytes(b"\x89PNG\r\n\x1a\n"+b"\x00\x00\x00\x0dIHDR"+struct.pack(">II",210,260))
                    rows.append(f"{name}|{name}/1/*.png|-1|-1|Signer01|{g}|translation")
                (ann/f"PHOENIX-2014-T.{split}.corpus.csv").write_text("name|video|start|end|speaker|orth|translation\n"+"\n".join(rows),encoding="utf-8")
            output=Path(tmp)/"prepared"
            args=["prepare-data","--dataset-root",str(root),"--output",str(output)]
            main(args)
            policy=json.loads((output/"splits/held_out.json").read_text())
            self.assertEqual(policy["retained_videos"],2)
            self.assertEqual(len((output/"manifests/reduced/train.jsonl").read_text().splitlines()),2)
            with self.assertRaises(SystemExit):
                main(args)

    def test_flat_nested_and_missing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/"features/fullFrame-210x260px/train/clip"
            base.mkdir(parents=True)
            (base/"images0001.png").write_bytes(b"test")
            self.assertEqual(len(resolve_frames(tmp,"train","clip")[1]),1)
            (base/"1").mkdir()
            (base/"images0001.png").rename(base/"1/images0001.png")
            self.assertEqual(resolve_frames(tmp,"train","clip")[0],base/"1")
            (base/"1/images0003.png").write_bytes(b"test")
            with self.assertRaises(ValueError):
                resolve_frames(tmp,"train","clip")

if __name__ == "__main__":
    unittest.main()
