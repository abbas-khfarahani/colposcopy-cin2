"""ImageNet-pretrained backbones with binary or multimodal heads."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


BackboneName = Literal[
    "resnet50",
    "resnet18",
    "efficientnet_b0",
    "efficientnet_b3",
    "densenet121",
    "convnext_tiny",
    "swin_t",
]


class BinaryImageClassifier(nn.Module):
    """Wraps a torchvision backbone; forward returns logits of shape (N,)."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.backbone(x)
        if logits.ndim == 2 and logits.size(1) == 1:
            return logits.squeeze(1)
        return logits


class SpatialAttentionClassifier(nn.Module):
    """ResNet body + spatial attention pooling (forces a soft 'where').

    Forward returns logits (N,). Call forward(..., return_attn=True) to also
    get the attention map (N, 1, H, W) over layer4 feature space.
    """

    def __init__(
        self,
        body: nn.Module,
        feat_dim: int,
        attn_hidden: int = 256,
        center_prior: bool = True,
        center_sigma: float = 0.45,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.body = body
        self.feat_dim = feat_dim
        self.center_prior = bool(center_prior)
        self.center_sigma = float(center_sigma)
        self.attn = nn.Sequential(
            nn.Conv2d(feat_dim, attn_hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(attn_hidden, 1, kernel_size=1, bias=True),
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(feat_dim, 1)
        # Expose as .backbone for Grad-CAM scripts that expect layer3/layer4.
        self.backbone = body

    def _center_bias(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        dist2 = xx * xx + yy * yy
        # Log-space Gaussian prior (higher in center).
        return (-dist2 / (2.0 * self.center_sigma * self.center_sigma)).view(1, 1, h, w)

    def forward_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.body(x)  # (N, C, H, W)
        logits_map = self.attn(feats)
        if self.center_prior:
            logits_map = logits_map + self._center_bias(
                feats.shape[-2], feats.shape[-1], feats.device, feats.dtype
            )
        n, _, h, w = logits_map.shape
        attn = torch.softmax(logits_map.view(n, -1), dim=1).view(n, 1, h, w)
        pooled = (feats * attn).sum(dim=(2, 3))  # (N, C)
        return pooled, attn

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        pooled, attn = self.forward_maps(x)
        logits = self.head(self.dropout(pooled)).squeeze(1)
        if return_attn:
            return logits, attn
        return logits


def build_resnet_spatial_body(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """ResNet truncated after layer4 → (N, C, H, W)."""
    if name not in ("resnet50", "resnet18"):
        raise ValueError(f"spatial attention currently supports resnet18/50, got {name!r}")
    net, feat_dim = _build_torchvision(name, pretrained)
    net.avgpool = nn.Identity()
    net.fc = nn.Identity()

    class ResNetSpatialBody(nn.Module):
        def __init__(self, m: nn.Module):
            super().__init__()
            self.conv1 = m.conv1
            self.bn1 = m.bn1
            self.relu = m.relu
            self.maxpool = m.maxpool
            self.layer1 = m.layer1
            self.layer2 = m.layer2
            self.layer3 = m.layer3
            self.layer4 = m.layer4

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return x

    return ResNetSpatialBody(net), feat_dim


class MultimodalFusionClassifier(nn.Module):
    """Image encoder features + clinical MLP → binary logit."""

    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int,
        clinical_dim: int,
        clinical_hidden: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, clinical_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(clinical_hidden, clinical_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(feat_dim + clinical_hidden, 1)

    def forward(self, x: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        c = self.clinical_mlp(clinical)
        logits = self.head(torch.cat([feats, c], dim=1))
        return logits.squeeze(1)


def _replace_classifier(in_features: int) -> nn.Linear:
    return nn.Linear(in_features, 1)


def _build_torchvision(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return full classification net and its final in_features."""
    if name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        net = models.resnet50(weights=weights)
        return net, int(net.fc.in_features)
    if name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.resnet18(weights=weights)
        return net, int(net.fc.in_features)
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        return net, int(net.classifier[1].in_features)
    if name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.efficientnet_b3(weights=weights)
        return net, int(net.classifier[1].in_features)
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.densenet121(weights=weights)
        return net, int(net.classifier.in_features)
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.convnext_tiny(weights=weights)
        return net, int(net.classifier[2].in_features)
    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.swin_t(weights=weights)
        return net, int(net.head.in_features)
    raise ValueError(f"Unsupported backbone: {name!r}")


def _set_binary_head(net: nn.Module, name: str, feat_dim: int) -> None:
    head = _replace_classifier(feat_dim)
    if name.startswith("resnet"):
        net.fc = head
    elif name.startswith("efficientnet"):
        net.classifier[1] = head
    elif name == "densenet121":
        net.classifier = head
    elif name == "convnext_tiny":
        net.classifier[2] = head
    elif name == "swin_t":
        net.head = head
    else:
        raise ValueError(name)


def _set_identity_head(net: nn.Module, name: str) -> None:
    if name.startswith("resnet"):
        net.fc = nn.Identity()
    elif name.startswith("efficientnet"):
        net.classifier = nn.Identity()
    elif name == "densenet121":
        net.classifier = nn.Identity()
    elif name == "convnext_tiny":
        net.classifier = nn.Identity()
    elif name == "swin_t":
        net.head = nn.Identity()
    else:
        raise ValueError(name)


def build_model(
    backbone: BackboneName = "resnet50",
    pretrained: bool = True,
    model_type: str = "image",
    clinical_dim: int = 0,
    clinical_hidden: int = 64,
    dropout: float = 0.25,
    attn_hidden: int = 256,
    center_prior: bool = True,
    center_sigma: float = 0.45,
) -> nn.Module:
    name = backbone.lower().strip()
    model_type = (model_type or "image").lower().strip()

    if model_type in ("image_attn", "spatial_attn", "attn"):
        body, feat_dim = build_resnet_spatial_body(name, pretrained)
        return SpatialAttentionClassifier(
            body=body,
            feat_dim=feat_dim,
            attn_hidden=int(attn_hidden),
            center_prior=bool(center_prior),
            center_sigma=float(center_sigma),
            dropout=float(dropout),
        )

    net, feat_dim = _build_torchvision(name, pretrained)

    if model_type == "image":
        _set_binary_head(net, name, feat_dim)
        return BinaryImageClassifier(net)

    if model_type == "multimodal":
        if clinical_dim <= 0:
            raise ValueError("clinical_dim must be > 0 for multimodal models")
        _set_identity_head(net, name)
        return MultimodalFusionClassifier(
            encoder=net,
            feat_dim=feat_dim,
            clinical_dim=clinical_dim,
            clinical_hidden=clinical_hidden,
            dropout=dropout,
        )

    if model_type == "multistate":
        from ml.models.multistate import build_multistate_model

        return build_multistate_model(
            backbone=backbone, pretrained=pretrained, dropout=dropout
        )

    raise ValueError(f"Unsupported model_type: {model_type!r}")


def param_groups_for_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_mult: float = 0.1,
) -> list[dict]:
    """Lower LR for pretrained backbone / encoder; higher for new heads."""
    head_keys = (
        "fc",
        "classifier",
        "head",
        "clinical_mlp",
        "type_emb",
        "attn",
        "spatial_attn",
    )
    backbone_params = []
    head_params = []
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in pname for k in head_keys):
            head_params.append(param)
        else:
            backbone_params.append(param)
    groups = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": lr * backbone_lr_mult,
                "weight_decay": weight_decay,
            }
        )
    if head_params:
        groups.append(
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        )
    if not groups:
        groups.append(
            {
                "params": list(model.parameters()),
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return groups
