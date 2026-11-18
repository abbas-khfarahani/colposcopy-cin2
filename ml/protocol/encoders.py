"""Image encoders used by the protocol (GAP, spatial attention, embeddings)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ml.models import SpatialAttentionClassifier, build_model, build_resnet_spatial_body
from ml.models.classifier import BinaryImageClassifier


@dataclass
class EncoderOutput:
    logits: torch.Tensor
    embedding: torch.Tensor
    attn: torch.Tensor | None = None


class ProtocolImageModel(nn.Module):
    """Unified wrapper: logits always, embeddings when needed for MIL/SSL."""

    def __init__(self, net: nn.Module, feat_dim: int) -> None:
        super().__init__()
        self.net = net
        self.feat_dim = int(feat_dim)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        if isinstance(self.net, SpatialAttentionClassifier):
            pooled, attn = self.net.forward_maps(x)
            logits = self.net.head(self.net.dropout(pooled)).squeeze(1)
            if return_aux:
                return EncoderOutput(logits=logits, embedding=pooled, attn=attn)
            return logits

        logits = self.net(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if not return_aux:
            return logits
        emb = _gap_embedding(self.net, x)
        return EncoderOutput(logits=logits, embedding=emb, attn=None)

    def embeddings(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        out = self.forward(x, return_aux=True)
        assert isinstance(out, EncoderOutput)
        return out.embedding, out.attn


def _gap_embedding(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    backbone = net.backbone if isinstance(net, BinaryImageClassifier) else net
    if hasattr(backbone, "layer4"):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)
        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)
        x = backbone.avgpool(x)
        return torch.flatten(x, 1)
    if hasattr(backbone, "features") and hasattr(backbone, "avgpool"):
        x = backbone.features(x)
        x = backbone.avgpool(x)
        classifier = getattr(backbone, "classifier", None)
        if isinstance(classifier, nn.Sequential) and len(classifier) >= 2:
            x = classifier[0](x)
            x = classifier[1](x)
            if x.ndim > 2:
                x = torch.flatten(x, 1)
            return x
        return torch.flatten(x, 1)
    raise TypeError(f"Cannot extract embeddings from {type(backbone)}")


def feat_dim_for(backbone: str) -> int:
    name = backbone.lower()
    if name == "resnet50":
        return 2048
    if name == "resnet18":
        return 512
    if name == "convnext_tiny":
        return 768
    if name.startswith("efficientnet_b0"):
        return 1280
    if name.startswith("efficientnet_b3"):
        return 1536
    raise KeyError(backbone)


def build_protocol_model(
    backbone: str = "resnet50",
    model_type: str = "image",
    pretrained: bool = True,
    dropout: float = 0.25,
    attn_hidden: int = 256,
    center_prior: bool = True,
    center_sigma: float = 0.45,
) -> ProtocolImageModel:
    net = build_model(
        backbone=backbone,  # type: ignore[arg-type]
        pretrained=pretrained,
        model_type=model_type,
        dropout=dropout,
        attn_hidden=attn_hidden,
        center_prior=center_prior,
        center_sigma=center_sigma,
    )
    return ProtocolImageModel(net, feat_dim=feat_dim_for(backbone))


def load_body_weights(model: ProtocolImageModel, state: dict) -> None:
    """Load SSL / previous body weights when keys match."""
    own = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    own.update(filtered)
    model.load_state_dict(own)
