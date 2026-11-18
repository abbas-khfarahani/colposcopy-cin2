"""Multistate attention-fusion classifier (acetic + green + Lugol)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.classifier import BackboneName, _build_torchvision


class MultistateAttentionFusion(nn.Module):
    """Shared CNN encoder over V views + type embedding + masked attention pool."""

    def __init__(
        self,
        encoder: nn.Module,
        feat_dim: int,
        n_types: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.feat_dim = feat_dim
        self.type_emb = nn.Embedding(n_types, feat_dim)
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feat_dim // 2, 1),
        )

    def encode_views(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, C, H, W) -> (B, V, D)"""
        b, v, c, h, w = x.shape
        flat = x.reshape(b * v, c, h, w)
        feats = self.encoder(flat)
        if feats.ndim > 2:
            feats = torch.flatten(feats, 1)
        return feats.view(b, v, -1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        x: (B, V, C, H, W)
        mask: (B, V) with 1=present
        type_ids: (B, V) long
        returns logits (B,)
        """
        feats = self.encode_views(x)
        feats = feats + self.type_emb(type_ids)
        scores = self.attn(feats).squeeze(-1)  # (B, V)
        # Softmax over present views only; keep autograd path for present slots.
        neg_large = torch.finfo(scores.dtype).min / 4
        scores = scores.masked_fill(mask < 0.5, neg_large)
        weights = F.softmax(scores, dim=1) * mask
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / denom
        pooled = torch.sum(feats * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(1)


def build_multistate_model(
    backbone: BackboneName = "resnet50",
    pretrained: bool = True,
    dropout: float = 0.25,
) -> MultistateAttentionFusion:
    from ml.models.classifier import _set_identity_head

    name = backbone.lower().strip()
    net, feat_dim = _build_torchvision(name, pretrained)
    _set_identity_head(net, name)
    return MultistateAttentionFusion(
        encoder=net, feat_dim=feat_dim, n_types=3, dropout=dropout
    )
