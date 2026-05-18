"""Shared helper: warm-start an n-way subject-id head from a trained victim.

Both the encoder-fine-tune adaptive attacker (experiments 15 / 18 / 23) and
the Fredrikson-style model-inversion attack (experiment 28) need the same
operation: take a trained victim (vanilla EEGNet or DP-SGD-Opacus EEGNet),
strip the task head, attach a fresh n_subjects-way classification head, and
train the whole thing end-to-end on subject-id labels.

The differences between source types — Opacus's GradSampleModule hooks vs
braindecode's parametrize maxnorm — are isolated here, behind a single
`finetune_to_reid_head(victim, X, subject_ids, ..., source="dp"|"vanilla")`
entry point.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _strip_hooks(module: nn.Module) -> None:
    for m in module.modules():
        m._forward_hooks.clear()
        m._backward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_full_backward_hooks"):
            m._full_backward_hooks.clear()
        if hasattr(m, "_full_backward_pre_hooks"):
            m._full_backward_pre_hooks.clear()


def _find_head_attr(module: nn.Module) -> str:
    for candidate in ("final_layer", "classifier", "fc", "head"):
        if hasattr(module, candidate):
            return candidate
    raise RuntimeError("Cannot locate classifier head on victim module")


def _clean_dp_backbone(victim) -> nn.Module:
    """Rebuild a fresh GroupNorm-EEGNet (architecture-equivalent to the
    one DP-SGD trained) and copy weights from the trained victim.

    Sidesteps Opacus's hook bookkeeping which otherwise prevents deepcopy.
    """
    fresh = victim._build()
    sd = victim.model_.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        cleaned[k[len("_module."):]] = v if not k.startswith("_module.") else v
        # The above doesn't actually replace the key — fix:
    cleaned = {}
    for k, v in sd.items():
        if k.startswith("_module."):
            cleaned[k[len("_module."):]] = v
        else:
            cleaned[k] = v
    fresh.load_state_dict(cleaned, strict=False)
    _strip_hooks(fresh)
    return fresh


def _clean_vanilla_backbone(victim) -> nn.Module:
    """Deep-copy the vanilla EEGNet backbone for fine-tuning.

    Vanilla EEGNets have braindecode's parametrize maxnorm on the spatial
    conv. We can keep that parametrization through fine-tuning — unlike
    DP-SGD it doesn't interfere with our optimisation, and it preserves
    the original training-time inductive bias.
    """
    from copy import deepcopy
    fresh = deepcopy(victim.model_)
    _strip_hooks(fresh)
    return fresh


class _FineTunedReID(nn.Module):
    """Encoder (warm-started) + linear n_subjects-way head."""

    def __init__(self, backbone: nn.Module, n_subjects: int,
                 *, input_scale: float = 1e6) -> None:
        super().__init__()
        self.input_scale = input_scale
        head_attr = _find_head_attr(backbone)
        setattr(backbone, head_attr, nn.Identity())
        self.backbone = backbone
        self._n_subjects = n_subjects
        self.head: nn.Module | None = None  # lazy init on first forward

    def _build_head(self, feat_dim: int, device: torch.device) -> None:
        self.head = nn.Linear(feat_dim, self._n_subjects).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale
        h = self.backbone(x)
        if h.ndim > 2:
            h = h.flatten(1)
        if self.head is None:
            self._build_head(h.shape[1], h.device)
        return self.head(h)


def finetune_to_reid_head(
    victim,
    X_train: np.ndarray,
    subject_ids: np.ndarray,
    *,
    device: str = "cuda",
    n_epochs: int = 15,
    lr: float = 5e-4,
    batch_size: int = 64,
    weight_decay: float = 1e-4,
    seed: int = 0,
    source: str = "vanilla",          # "vanilla" | "dp"
) -> tuple[nn.Module, dict[int, int]]:
    """Fine-tune `victim` into an n-way subject-id classifier.

    Returns the trained nn.Module plus a `subject_id -> output_index` map
    so callers can resolve the model's output indices back to subject IDs.
    """
    if source not in ("vanilla", "dp"):
        raise ValueError(f"unknown source={source!r}; expected 'vanilla' or 'dp'")
    torch.manual_seed(seed)
    sid_to_idx = {int(s): i for i, s in enumerate(sorted(set(int(x) for x in subject_ids)))}
    y_idx = np.array([sid_to_idx[int(s)] for s in subject_ids], dtype=np.int64)
    n_subjects = len(sid_to_idx)

    backbone = (_clean_dp_backbone(victim) if source == "dp"
                else _clean_vanilla_backbone(victim))
    model = _FineTunedReID(backbone, n_subjects,
                           input_scale=getattr(victim, "input_scale", 1e6)).to(device)
    # Warm up the lazy head with one dummy forward so AdamW sees all
    # parameters at epoch 0.
    n_channels = getattr(victim, "n_channels")
    n_times = getattr(victim, "n_times")
    with torch.no_grad():
        dummy = torch.zeros(2, n_channels, n_times,
                            device=device, dtype=torch.float32)
        model.eval(); model(dummy); model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    Xt = torch.from_numpy(X_train.astype(np.float32, copy=False))
    yt = torch.from_numpy(y_idx)
    for epoch in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            sl = idx[i:i + batch_size]
            xb = Xt[sl].to(device)
            yb = yt[sl].to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
    return model, sid_to_idx
