"""Build portable records without changing original dataset files."""
from .annotations import read_annotations
from .frame_reader import resolve_frames, png_size


def build_manifest(root, split, all_headers=False):
    path = root / "annotations" / "manual" / f"PHOENIX-2014-T.{split}.corpus.csv"
    rows = read_annotations(path)
    for row in rows:
        folder, frames = resolve_frames(root, split, row["id"])
        checked = frames if all_headers else [frames[0]]
        for image in checked:
            if png_size(image) != (210, 260):
                raise ValueError(f"Unexpected frame dimensions: {image}")
        row.update(split=split, frame_dir=folder.relative_to(root).as_posix(),
                   frame_count=len(frames), fps=25, duration_seconds=len(frames)/25)
    return rows
