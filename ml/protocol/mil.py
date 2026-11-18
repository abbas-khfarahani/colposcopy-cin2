"""Gated attention multiple-instance learning (Ilse et al.)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.protocol.encoders import ProtocolImageModel


class GatedAttentionMIL(nn.Module):
    """Two-level model: per-image spatial encoder + patient-level gated MIL."""

    def __init__(
        self,
        encoder: ProtocolImageModel,
        embed_dim: int = 512,
        attn_hidden: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.feat_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.mil_attn_v = nn.Linear(embed_dim, attn_hidden)
        self.mil_attn_u = nn.Linear(embed_dim, attn_hidden)
        self.mil_attn_w = nn.Linear(attn_hidden, 1)
        self.mil_head = nn.Linear(embed_dim, 1)

    def bag_forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """images: (n_inst, C, H, W) → logit, instance attention, spatial attn."""
        emb, spatial = self.encoder.embeddings(images)
        h = self.proj(emb)
        a = self.mil_attn_w(torch.tanh(self.mil_attn_v(h)) * torch.sigmoid(self.mil_attn_u(h)))
        a = torch.softmax(a, dim=0)
        z = (a * h).sum(dim=0)
        logit = self.mil_head(z).squeeze(-1)
        return logit, a.squeeze(-1), spatial

    def forward(self, bags: list[torch.Tensor]) -> torch.Tensor:
        logits = [self.bag_forward(bag)[0] for bag in bags]
        return torch.stack(logits, dim=0)
