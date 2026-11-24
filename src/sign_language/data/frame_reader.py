"""Resolve both released nested and extracted flat frame layouts."""
import re
import struct
from pathlib import Path


def resolve_frames(root, split, clip_id):
    base = Path(root).resolve() / "features" / "fullFrame-210x260px" / split / clip_id
    candidates = [p for p in (base, base / "1") if p.is_dir() and next(p.glob("*.png"), None)]
    if len(candidates) != 1:
        raise ValueError(f"Missing or ambiguous frame layout: {base}")
    folder = candidates[0].resolve()
    if not folder.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Frame path escapes dataset: {folder}")
    frames = sorted(folder.glob("*.png"), key=lambda p: int(re.search(r"(\d+)\.png$", p.name).group(1)) if re.search(r"(\d+)\.png$", p.name) else -1)
    numbers = []
    for p in frames:
        match = re.fullmatch(r"images(\d+)\.png", p.name)
        if not match:
            raise ValueError(f"Unexpected frame name: {p}")
        numbers.append(int(match.group(1)))
        if p.stat().st_size == 0:
            raise ValueError(f"Empty frame: {p}")
    if numbers != list(range(1, len(frames) + 1)):
        raise ValueError(f"Missing or duplicated frame indices: {folder}")
    return folder, frames


def png_size(path):
    with Path(path).open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])
