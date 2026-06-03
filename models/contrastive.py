"""Contrastive EEGNet for open-set verification (A4).

Wraps braindecode's EEGNet as a backbone, strips the task-classification
head, and projects the pre-classifier feature map to an L2-normalized
64-d embedding. Trained with batch-hard triplet loss using subject IDs
as the supervisory signal: anchors and positives are different windows
from the same subject; negatives come from any other subject.

Used by attacks/verification.py to test whether the learned embedding
generalizes to subjects the network has never seen during training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGNet


class ContrastiveEEGNet(nn.Module):
    """EEGNet backbone + small projection head producing unit-norm embeddings."""

    def __init__(
        self,
        n_chans: int,
        n_times: int,
        embed_dim: int = 64,
        input_scale: float = 1e6,
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.backbone = EEGNet(n_chans=n_chans, n_outputs=4, n_times=n_times)
        # Strip braindecode's task classifier — we project the pre-classifier
        # feature map ourselves.
        self.backbone.final_layer = nn.Identity()
        # Probe the backbone's output shape so the projection's input dim is correct
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            feat = self.backbone(dummy)
        self.feat_dim = int(feat.flatten(1).shape[1])
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.feat_dim, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale
        h = self.backbone(x)
        z = self.head(h)
        return F.normalize(z, dim=1)


def batch_hard_triplet_loss(
    emb: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Batch-hard triplet loss with the subject-id labels currently in the batch.

    For each anchor, picks the same-subject sample with the LARGEST distance
    in the batch (hardest positive) and the different-subject sample with
    the SMALLEST distance (hardest negative). This is more sample-efficient
    than random triplets at comparable batch sizes (Schroff et al., 2015).
    """
    # Pairwise Euclidean distance. emb is L2-normalized, so this is monotone
    # in negative cosine similarity.
    dists = torch.cdist(emb, emb, p=2)

    same = labels[:, None] == labels[None, :]
    not_self = ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    pos_mask = same & not_self
    neg_mask = ~same

    pos_dists = dists.masked_fill(~pos_mask, -1.0)
    neg_dists = dists.masked_fill(~neg_mask, float("inf"))

    hardest_pos = pos_dists.max(dim=1).values
    hardest_neg = neg_dists.min(dim=1).values

    loss = (hardest_pos - hardest_neg + margin).clamp(min=0.0)
    # Skip anchors whose batch had no valid positive (pathological tiny batches)
    valid = pos_mask.any(dim=1)
    if valid.sum() == 0:
        return torch.zeros((), device=emb.device, requires_grad=True)
    return loss[valid].mean()

