"""D2 — Domain-Adversarial Neural Network (DANN) defense.

EEGNet backbone + two heads:
  - task head:    predicts motor-imagery class (4-way)
  - subject head: predicts subject ID (104-way), connected to the
                  encoder via a Gradient Reversal Layer (GRL)

Loss = L_task(task_logits, y_task) + L_subj(subj_logits, y_subj)

The Gradient Reversal Layer (GRL) multiplies the ENCODER's gradient on
the subject-loss path by -λ; the two loss terms are then summed with unit
weight. So the subject head trains at full strength while the encoder
feels adversarial pressure of exactly λ — canonical DANN. (Applying λ a
second time as a loss coefficient, as an earlier version did, makes the
encoder feel λ² and mis-scales the whole λ-sweep.) λ controls the
adversarial strength: λ=0 reduces to vanilla EEGNet (= A1 baseline);
λ ≫ 0 trades task accuracy for subject invariance.

Reference: Ganin et al. "Domain-Adversarial Training of Neural
Networks", JMLR 2016 (arXiv:1505.07818).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.autograd import Function

from models.base import VictimModel


# ---------------------------------------------------------------------------
# Gradient reversal — identity on forward, multiplies by -lambda on backward
# ---------------------------------------------------------------------------
class _GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


# ---------------------------------------------------------------------------
# DANN-EEGNet module
# ---------------------------------------------------------------------------
class _DANNNet(nn.Module):
    def __init__(
        self,
        n_chans: int,
        n_times: int,
        n_classes_task: int = 4,
        n_classes_subj: int = 104,
        input_scale: float = 1e6,
    ) -> None:
        super().__init__()
        self.input_scale = input_scale
        self.backbone = EEGNet(n_chans=n_chans, n_outputs=n_classes_task,
                               n_times=n_times)
        # Strip the original task head; we'll attach our own.
        self.backbone.final_layer = nn.Identity()
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            feat = self.backbone(dummy)
        self.feat_dim = int(feat.flatten(1).shape[1])

        self.task_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.feat_dim, n_classes_task),
        )
        self.subj_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(self.feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes_subj),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale
        return self.backbone(x).flatten(1)

    def forward(self, x: torch.Tensor, lambda_: float = 0.0):
        h = self.features(x)                       # (B, feat_dim)
        task_logits = self.task_head(h)            # (B, n_task)
        # GRL applied on the path going to the subject head only.
        subj_logits = self.subj_head(grad_reverse(h, lambda_))  # (B, n_subj)
        return task_logits, subj_logits


def _pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DANNVictim(VictimModel):
    """EEGNet trained with adversarial subject-invariance pressure.

    Looks like every other VictimModel from the outside (fit/predict/embed)
    so the existing A1 attack runs against it unchanged.
    """
    name = "eegnet_dann"

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_classes_task: int = 4,
        n_classes_subj: int = 104,
        n_epochs: int = 60,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_: float = 0.5,
        device: str = "auto",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes_task
        self.n_classes_subj = n_classes_subj
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ = lambda_
        self.device = _pick_device(device)
        self.seed = seed
        self.verbose = verbose
        self.model_: _DANNNet | None = None
        self.subject_to_idx_: dict[int, int] | None = None

    # ---- helpers ------------------------------------------------------
    def _build(self) -> _DANNNet:
        torch.manual_seed(self.seed)
        net = _DANNNet(
            n_chans=self.n_channels, n_times=self.n_times,
            n_classes_task=self.n_classes, n_classes_subj=self.n_classes_subj,
        ).to(self.device)
        return net

    def _iter_batches(self, X: np.ndarray, y_task: np.ndarray | None,
                      y_subj: np.ndarray | None, shuffle: bool, epoch: int = 0):
        idx = np.arange(len(X))
        if shuffle:
            # Advance the shuffle across epochs — seeding with a constant gave
            # the identical mini-batch order on every epoch.
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            sl = idx[start:start + self.batch_size]
            xb = torch.from_numpy(X[sl].astype(np.float32, copy=False)).to(self.device)
            yt = None if y_task is None else torch.from_numpy(y_task[sl]).long().to(self.device)
            ys = None if y_subj is None else torch.from_numpy(y_subj[sl]).long().to(self.device)
            yield xb, yt, ys

    # ---- VictimModel API ---------------------------------------------
    def fit(self, X: np.ndarray, y_task: np.ndarray,
            subject_ids: np.ndarray | None = None) -> DANNVictim:
        if subject_ids is None:
            raise ValueError(
                "DANNVictim.fit requires subject_ids — pass the subject id of every window"
            )
        # Map possibly-sparse subject ids (e.g., {1, 2, ..., 109} \ drops)
        # into a dense {0, ..., n-1} space the subject head can predict.
        unique = sorted(set(int(s) for s in subject_ids))
        self.subject_to_idx_ = {s: i for i, s in enumerate(unique)}
        # Re-instantiate with the right subject-class count
        self.n_classes_subj = len(unique)
        y_subj = np.array([self.subject_to_idx_[int(s)] for s in subject_ids],
                          dtype=np.int64)

        self.model_ = self._build()
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        ce = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.n_epochs):
            running_t = running_s = 0.0
            n = 0
            for xb, yt, ys in self._iter_batches(X, y_task, y_subj,
                                                 shuffle=True, epoch=epoch):
                opt.zero_grad()
                task_logits, subj_logits = self.model_(xb, lambda_=self.lambda_)
                lt = ce(task_logits, yt)
                ls = ce(subj_logits, ys)
                # GRL already scales the encoder's subject-path gradient by
                # lambda; sum with unit weight so the encoder feels exactly
                # lambda (not lambda^2) and the subject head trains fully.
                loss = lt + ls
                loss.backward()
                opt.step()
                running_t += lt.item() * len(xb)
                running_s += ls.item() * len(xb)
                n += len(xb)
            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"  epoch {epoch:3d}  task_loss={running_t/n:.3f}  "
                      f"subj_loss={running_s/n:.3f}", flush=True)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        self.model_.eval()
        out = []
        for xb, _, _ in self._iter_batches(X, None, None, shuffle=False):
            task_logits, _ = self.model_(xb, lambda_=0.0)
            out.append(task_logits.argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Pre-classifier features (the encoder output) — what the A1
        attack will probe for subject identity."""
        assert self.model_ is not None
        self.model_.eval()
        outs = []
        for xb, _, _ in self._iter_batches(X, None, None, shuffle=False):
            h = self.model_.features(xb)
            outs.append(h.cpu().numpy())
        emb = np.concatenate(outs, axis=0)
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        return emb.astype(np.float32, copy=False)
