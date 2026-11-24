"""Deterministic CorrNet evaluation preprocessing from the original PNGs."""
import hashlib
import cv2
import numpy as np
import torch
from .frame_reader import resolve_frames


PREPROCESSING = {"version": 1, "color": "RGB", "resize": [256, 256],
                 "resize_interpolation": "OpenCV INTER_LANCZOS4", "center_crop": [224, 224],
                 "normalization": "pixel / 127.5 - 1", "frame_interval": 1,
                 "temporal_padding": "repeat first 6; repeat last to ceil(T/4)*4+12",
                 "augmentation": "none", "batch_size": 1}


def load_corrnet_video(root, record):
    folder, paths = resolve_frames(root, record["split"], record["id"])
    if len(paths) != record["frame_count"] or folder.relative_to(root.resolve()).as_posix() != record["frame_dir"]:
        raise ValueError(f"Source frames differ from manifest for {record['id']}")
    digest = hashlib.sha256()
    frames = []
    for path in paths:
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None or frame.shape != (260, 210, 3):
            raise ValueError(f"Invalid or unexpected image: {path}")
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[16:240, 16:240]
        frames.append(frame)
    video = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
    padded_length = ((len(video) + 3) // 4) * 4 + 12
    video = torch.cat((video[:1].expand(6, -1, -1, -1), video,
                       video[-1:].expand(padded_length - len(video) - 6, -1, -1, -1)))
    return video.unsqueeze(0), torch.tensor([padded_length]), digest.hexdigest()
