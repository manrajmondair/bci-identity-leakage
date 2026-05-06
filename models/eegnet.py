"""EEGNet (Lawhern et al. 2018) wrapper.

Uses braindecode's EEGNet implementation as the backbone. The wrapper adds:
  - a clean fit/predict/embed API matching VictimModel,
  - an embedding hook that returns the features just before the final dense
    layer (used by every attack module),
  - device selection (MPS on Apple Silicon, CUDA, or CPU).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGNet

from models.base import VictimModel


def _pick_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


class EEGNetVictim(VictimModel):
    """EEGNet trained for motor-imagery classification.

    Parameters
    ----------
    n_channels, n_times : EEG window shape (channels × samples).
    n_classes : number of output classes (4 for our motor-imagery splits).
    n_epochs : training epochs.
    batch_size, lr, weight_decay : optimizer hyperparameters.
    device : 'auto', 'cpu', 'mps', or 'cuda'.
    seed : torch RNG seed (also seeds numpy in fit()).
    """
    name = "eegnet"

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_classes: int = 4,
        n_epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "auto",
        seed: int = 0,
        verbose: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = _pick_device(device)
        self.seed = seed
        self.verbose = verbose
        self.model_: EEGNet | None = None
        self._embedding_dim: int | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build(self) -> EEGNet:
        torch.manual_seed(self.seed)
        net = EEGNet(
            n_chans=self.n_channels,
            n_outputs=self.n_classes,
            n_times=self.n_times,
        )
        return net.to(self.device)

    def _iter_batches(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool):
        idx = np.arange(len(X))
        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            sl = idx[start:start + self.batch_size]
            xb = torch.from_numpy(X[sl]).to(self.device)
            yb = None if y is None else torch.from_numpy(y[sl]).long().to(self.device)
            yield xb, yb

    # ------------------------------------------------------------------
    # VictimModel API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "EEGNetVictim":
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        self.model_ = self._build()
        opt = torch.optim.AdamW(self.model_.parameters(),
                                lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        self.model_.train()
        for epoch in range(self.n_epochs):
            running = 0.0
            n = 0
            for xb, yb in self._iter_batches(X, y, shuffle=True):
                opt.zero_grad()
                logits = self.model_(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                running += loss.item() * len(xb)
                n += len(xb)
            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                print(f"  epoch {epoch:3d}  loss={running/n:.4f}")
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None, "Call fit() first"
        self.model_.eval()
        out = []
        for xb, _ in self._iter_batches(X, None, shuffle=False):
            out.append(self.model_(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Return features from just before the final classification layer.

        EEGNet's final layer is a Linear taking flattened spatiotemporal
        features → n_classes. We hook into the input of that Linear by
        running the network up to the second-to-last module and applying
        the same flatten that the final layer expects.
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None, "Call fit() first"

        # Capture the input to the final classification layer.
        final_layer = self._find_final_linear(self.model_)
        captured: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            captured.append(inputs[0].detach().cpu())

        handle = final_layer.register_forward_hook(hook)
        self.model_.eval()
        try:
            outs = []
            for xb, _ in self._iter_batches(X, None, shuffle=False):
                self.model_(xb)
                outs.append(captured.pop().numpy())
            emb = np.concatenate(outs, axis=0)
        finally:
            handle.remove()

        # Keep as 2-D (n, d). braindecode's EEGNet feeds a flattened tensor
        # into its final Linear, so emb is already (n, d). Defensive flatten:
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        self._embedding_dim = emb.shape[1]
        return emb.astype(np.float32, copy=False)

    @staticmethod
    def _find_final_linear(net: nn.Module) -> nn.Linear:
        """Return the last nn.Linear in the network's module tree.

        braindecode renames their final layer over time ('final_layer',
        'classifier', etc.); rather than hardcoding a name, we find the
        last nn.Linear in execution order, which is what we want.
        """
        last: nn.Linear | None = None
        for module in net.modules():
            if isinstance(module, nn.Linear):
                last = module
        if last is None:
            raise RuntimeError("EEGNet has no nn.Linear final layer to hook")
        return last
