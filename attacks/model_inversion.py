"""Model-inversion attack on a re-identification head (Fredrikson-style).

Given white-box access to a trained re-ID model
    f: R^{n_chans x n_times} -> R^{n_subjects}
the attacker chooses a target subject id k and reconstructs a synthetic
EEG window x* that the model maps to high probability of subject k:

    x*_k = argmin_{x in feasible}  CE(f(x), one_hot(k))

with x initialised near zero, optimised via projected gradient descent on
the input. The only regulariser is the feasibility projection that keeps
each channel's std within the empirical bounds of real motor-imagery EEG
(so we recover something EEG-shaped rather than adversarial garbage); there
is no separate lambda*R(x) penalty term.

Reconstruction success metric: cosine similarity in a frozen reference
embedding (the project's pretrained contrastive EEGNet from experiment
06) between the reconstructed window x*_k and that subject's real
training-set windows. A successful inversion is one where
sim(x*_k, real_x_k) is much higher than sim(x*_k, real_x_j != k) — the
attack identifies subject k.

The point of the experiment is the asymmetry: under no defense the
reconstruction should be recognisable as subject k; under DP-SGD the
reconstruction should look like generic EEG (similarity distribution
collapses to the null distribution).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class InversionResult:
    n_subjects: int
    n_reconstructions: int
    n_target_real: int
    n_other_real: int
    median_sim_to_target: float
    median_sim_to_other: float
    advantage_top1_recovery: float
    advantage_top5_recovery: float
    rank1_acc: float
    rank5_acc: float


def _per_channel_std_bounds(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (10th, 90th) percentile of per-channel std across real windows.

    The inversion projects each reconstructed window's channel-stds to lie
    inside [10th, 90th] percentile of real-data channel-stds so the
    optimiser cannot escape to unbounded amplitudes.
    """
    chan_std = X.std(axis=2)                    # (n_windows, n_channels)
    lo = np.percentile(chan_std, 10, axis=0)
    hi = np.percentile(chan_std, 90, axis=0)
    return lo, hi


def _project_to_eeg_feasible(x: torch.Tensor,
                             std_lo: torch.Tensor,
                             std_hi: torch.Tensor) -> torch.Tensor:
    """Project x into the per-channel-std feasibility envelope."""
    cur_std = x.std(dim=-1, keepdim=True)            # (B, C, 1)
    # Avoid div-by-zero
    cur_std = cur_std.clamp_min(1e-9)
    target_std = torch.clamp(cur_std,
                             min=std_lo[None, :, None],
                             max=std_hi[None, :, None])
    return x * (target_std / cur_std)


def invert_subject(
    f: nn.Module,
    *,
    target_subject_logit_idx: int,
    n_chans: int,
    n_times: int,
    std_lo: np.ndarray,
    std_hi: np.ndarray,
    input_scale: float,
    n_steps: int = 600,
    lr: float = 0.05,
    weight_decay: float = 0.0,
    device: str = "cuda",
    seed: int = 0,
) -> torch.Tensor:
    """Reconstruct a single subject-targeted EEG window.

    f must take (B, C, T) inputs scaled by `input_scale` and output
    (B, n_subjects) logits.

    `weight_decay` defaults to 0: Adam weight decay here would decay the
    optimised INPUT x toward zero every step, fighting the feasibility
    projection and biasing reconstructions toward null EEG. The per-channel
    std projection is the intended (and only) regulariser; there is no
    separate lambda*R(x) penalty term.
    """
    torch.manual_seed(seed)
    std_lo_t = torch.as_tensor(std_lo, dtype=torch.float32, device=device)
    std_hi_t = torch.as_tensor(std_hi, dtype=torch.float32, device=device)
    x = (torch.randn(1, n_chans, n_times, device=device) * 1e-6).requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    f.eval()
    target = torch.tensor([target_subject_logit_idx], device=device, dtype=torch.long)
    for _step in range(n_steps):
        opt.zero_grad()
        # f handles input_scale internally — pass raw EEG-scale x.
        logits = f(x)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.copy_(_project_to_eeg_feasible(x, std_lo_t, std_hi_t))
    return x.detach()


def evaluate_reconstructions(
    reconstructions: dict[int, torch.Tensor],  # subject_id -> (C, T) tensor
    real_windows: np.ndarray,
    real_subject_ids: np.ndarray,
    *,
    reference_embedder: nn.Module,
    device: str = "cuda",
    batch_size: int = 256,
) -> InversionResult:
    """Score reconstructions by cosine similarity to real-subject windows in a
    frozen reference embedding space.

    `reference_embedder` is expected to L2-normalize its outputs (e.g. the
    project's ContrastiveEEGNet). The metric reports both median pairwise
    cosine similarity and rank-recovery accuracy: for each reconstructed
    window x*_k, compute average similarity to each subject j's real
    windows; rank-1 recovery is the fraction of k for which argmax_j is k.
    """
    reference_embedder.eval()
    with torch.no_grad():
        emb_real_list = []
        for i in range(0, len(real_windows), batch_size):
            xb = torch.from_numpy(
                real_windows[i:i + batch_size].astype(np.float32, copy=False)
            ).to(device)
            emb_real_list.append(reference_embedder(xb).cpu())
        emb_real = torch.cat(emb_real_list, dim=0)        # (N_real, D)
        emb_real_np = emb_real.numpy()

    subj_to_idx = {int(s): np.where(real_subject_ids == s)[0] for s in np.unique(real_subject_ids)}

    target_subjects = sorted(reconstructions.keys())
    n_targets = len(target_subjects)
    rank1 = 0
    rank5 = 0
    sims_to_self_all: list[float] = []
    sims_to_others_all: list[float] = []
    top1_rec = 0
    top5_rec = 0
    for k in target_subjects:
        x_k = reconstructions[k].to(device)
        if x_k.ndim == 2:
            x_k = x_k.unsqueeze(0)
        with torch.no_grad():
            emb_recon = reference_embedder(x_k).cpu().numpy()[0]
        # Average cosine sim of x_k to subject j's real windows, for each j
        scores_per_subj: dict[int, float] = {}
        for j, idxs in subj_to_idx.items():
            if len(idxs) == 0:
                continue
            sims = emb_real_np[idxs] @ emb_recon  # cosine since both L2-normed
            scores_per_subj[j] = float(sims.mean())
        ranking = sorted(scores_per_subj.items(), key=lambda kv: -kv[1])
        top = [r[0] for r in ranking]
        if top[0] == k:
            rank1 += 1
            top1_rec += 1
        if k in top[:5]:
            rank5 += 1
            top5_rec += 1
        sims_to_self_all.append(scores_per_subj.get(k, float("nan")))
        sims_to_others_all.extend([v for j, v in scores_per_subj.items() if j != k])

    n_unique_real_subjects = int(len(np.unique(real_subject_ids)))
    return InversionResult(
        n_subjects=n_unique_real_subjects,
        n_reconstructions=int(n_targets),
        n_target_real=int(sum(len(idxs) for idxs in subj_to_idx.values())),
        n_other_real=int(sum(len(idxs) for idxs in subj_to_idx.values())),
        median_sim_to_target=float(np.nanmedian(sims_to_self_all)),
        median_sim_to_other=float(np.nanmedian(sims_to_others_all)),
        advantage_top1_recovery=float(rank1 / n_targets - 1.0 / n_unique_real_subjects),
        advantage_top5_recovery=float(rank5 / n_targets - 5.0 / n_unique_real_subjects),
        rank1_acc=float(rank1 / n_targets),
        rank5_acc=float(rank5 / n_targets),
    )
