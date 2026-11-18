"""Fold-safe SimCLR / patient-aware (MICLe-like) pretraining."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models import build_resnet_spatial_body
from ml.protocol.constants import SSL_PROJECTOR_DIM


class SSLResNet(nn.Module):
    """ResNet50 body + GAP + MLP projector. Body transfers into attention/MIL."""

    def __init__(self, pretrained: bool = True, projector_dim: int = SSL_PROJECTOR_DIM) -> None:
        super().__init__()
        body, feat_dim = build_resnet_spatial_body("resnet50", pretrained=pretrained)
        self.body = body
        self.feat_dim = feat_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projector_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.body(x)
        pooled = feats.mean(dim=(2, 3))
        return F.normalize(self.projector(pooled), dim=1)


def nt_xent(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
    person_ids: list[str] | None = None,
) -> torch.Tensor:
    """InfoNCE with optional same-person negative masking (MICLe)."""
    # Always fp32: AMP float16 cannot store -1e9, and the original crash was
    # `masked_fill(..., -1e9)` under autocast.
    amp_device = "cuda" if z1.is_cuda else z1.device.type
    with torch.amp.autocast(device_type=amp_device, enabled=False):
        z1 = F.normalize(z1.float(), dim=1)
        z2 = F.normalize(z2.float(), dim=1)
        z = torch.cat([z1, z2], dim=0)
        n = z1.size(0)
        sim = (z @ z.T) / float(temperature)
        mask = torch.eye(2 * n, device=z.device, dtype=torch.bool)
        if person_ids is not None:
            pids = list(person_ids) + list(person_ids)
            for i in range(2 * n):
                for j in range(2 * n):
                    if i != j and pids[i] == pids[j]:
                        # Keep the designated positive (i ↔ i+n). Mask extra same-person.
                        if j != (i + n) % (2 * n):
                            mask[i, j] = True
        # -65504 fits float16 and is plenty for a softmax mask in float32.
        sim = sim.masked_fill(mask, torch.finfo(torch.float16).min)
        positives = torch.cat(
            [torch.arange(n, 2 * n, device=z.device), torch.arange(0, n, device=z.device)]
        )
        return F.cross_entropy(sim, positives)


def body_state_from_ssl(ssl: SSLResNet) -> dict[str, torch.Tensor]:
    """Map SSL body keys onto ProtocolImageModel spatial-attention body."""
    raw = ssl.body.state_dict()
    return {f"net.body.{k}": v for k, v in raw.items()}
