"""Inference adapter for the released ResNet18/conv_type=2 CorrNet graph.

Loads three upstream modules directly, avoiding unrelated training packages and
the platform-specific ctcdecode extension. No ImageNet download is performed:
every parameter must instead be present in the supplied CorrNet checkpoint.
"""
import importlib.util
from pathlib import Path
import torch
from torch import nn
from ..heads.normalized_linear import NormLinear
from ...core.provenance import file_sha256
from ...core.settings import validate_vocabulary


UPSTREAM_COMMIT = "a812814f4deefda0b75ab1dcc90b106c62912263"


def upstream_module(root, name):
    path = Path(root) / "modules" / f"{name}.py"
    if not path.is_file():
        raise ValueError(f"CorrNet source missing: {path}; see docs/phase2.md")
    spec = importlib.util.spec_from_file_location(f"_phase2_corrnet_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_hashes(root):
    return {name: file_sha256(Path(root) / "modules" / f"{name}.py")
            for name in ("resnet", "tconv", "BiLSTM")}


class CorrNetReference(nn.Module):
    def __init__(self, source, num_classes):
        super().__init__()
        resnet = upstream_module(source, "resnet")
        temporal = upstream_module(source, "tconv")
        recurrent = upstream_module(source, "BiLSTM")
        self.conv2d = resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2])
        self.conv2d.fc = nn.Identity()
        self.conv1d = temporal.TemporalConv(512, 1024, conv_type=2, use_bn=True, num_classes=num_classes)
        self.temporal_model = recurrent.BiLSTMLayer(input_size=1024, hidden_size=1024,
                                                  num_layers=2, bidirectional=True, rnn_type="LSTM")
        self.classifier = NormLinear(1024, num_classes)
        self.conv1d.fc = self.classifier

    def extract(self, video, lengths):
        if video.ndim != 5 or video.size(0) != 1 or tuple(video.shape[2:]) != (3, 224, 224):
            raise ValueError("Reference adapter requires one (1, time, 3, 224, 224) video")
        batch, time = video.shape[:2]
        framewise = self.conv2d(video.permute(0, 2, 1, 3, 4)).view(batch, time, -1).permute(0, 2, 1)
        result = self.conv1d(framewise, lengths)
        lengths = result["feat_len"]
        if not torch.equal(lengths, lengths.floor()):
            raise ValueError("Reference padding produced fractional feature lengths")
        return result["visual_feat"].permute(1, 0, 2), lengths.long()

    def from_features(self, features, lengths):
        result = self.temporal_model(features.permute(1, 0, 2), lengths.cpu())
        return self.classifier(result["predictions"]).permute(1, 0, 2)

    def forward(self, video, lengths):
        features, lengths = self.extract(video, lengths)
        return {"features": features, "lengths": lengths, "logits": self.from_features(features, lengths)}


def load_reference(source, checkpoint, vocabulary, device, trust_legacy=False):
    validate_vocabulary(vocabulary)
    if not vocabulary.get("checkpoint_compatible") or vocabulary.get("source_dataset") != "phoenix2014-T":
        raise ValueError("Use the released PHOENIX-2014-T vocabulary, not the Phase 1 project dictionary")
    if not Path(checkpoint).is_file():
        raise ValueError(f"CorrNet checkpoint missing: {checkpoint}; download instructions: docs/phase2.md")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=not trust_legacy)
    except Exception as error:
        raise ValueError("Cannot read checkpoint. For an independently trusted legacy file only, use "
                         "--trust-legacy-checkpoint to permit its pickled metadata. " + str(error)) from error
    weights = payload.get("model_state_dict", payload.get("state_dict", payload))
    weights = {".".join(p for p in key.split(".") if p != "module"): value for key, value in weights.items()}
    model = CorrNetReference(source, len(vocabulary["id_to_gloss"]))
    if "classifier.weight" in weights and "conv1d.fc.weight" in weights:
        if not torch.equal(weights["classifier.weight"], weights["conv1d.fc.weight"]):
            raise ValueError("Checkpoint does not use the expected shared classifier")
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval().requires_grad_(False)
