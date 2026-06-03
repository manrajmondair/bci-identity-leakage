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
        input_scale: float = 1e6,
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
        # mne returns EEG in volts (~1e-5 V). EEGNet's published hyperparameters
        # were tuned for microvolts (~10-100 uV). At the volt scale, gradients
        # through the temporal conv vanish and the network cannot learn motor
        # imagery; multiplying by 1e6 brings the input to the trained-on scale.
        self.input_scale = input_scale
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

    def _iter_batches(self, X: np.ndarray, y: np.ndarray | None, shuffle: bool,
                      epoch: int = 0):
        idx = np.arange(len(X))
        if shuffle:
            # Advance the shuffle across epochs — seeding with a constant gave
            # the identical mini-batch order on every epoch.
            rng = np.random.default_rng(self.seed + epoch)
            rng.shuffle(idx)
        for start in range(0, len(idx), self.batch_size):
            sl = idx[start:start + self.batch_size]
            # Rescale volts -> microvolts on the fly so the cached arrays
            # remain in physical units for the classical baselines.
            xb = torch.from_numpy(X[sl] * self.input_scale).to(self.device)
            yb = None if y is None else torch.from_numpy(y[sl]).long().to(self.device)
            yield xb, yb

    # ------------------------------------------------------------------
    # VictimModel API
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> EEGNetVictim:
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
            for xb, yb in self._iter_batches(X, y, shuffle=True, epoch=epoch):
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
        """Pre-classifier features: input to whatever module produces logits.

        braindecode's EEGNet ends in a small Sequential (`final_layer`)
        whose input is a (B, F, 1, T') feature map; flatten gives a
        per-window vector. We hook the head module by attribute name with
        sensible fallbacks across braindecode versions.
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None, "Call fit() first"

        head = self._find_head(self.model_)
        captured: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            captured.append(inputs[0].detach().cpu())

        handle = head.register_forward_hook(hook)
        self.model_.eval()
        try:
            outs = []
            for xb, _ in self._iter_batches(X, None, shuffle=False):
                self.model_(xb)
                outs.append(captured.pop().numpy())
            emb = np.concatenate(outs, axis=0)
        finally:
            handle.remove()

        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        self._embedding_dim = emb.shape[1]
        return emb.astype(np.float32, copy=False)

    @staticmethod
    def _find_head(net: nn.Module) -> nn.Module:
        """Locate the final classification head by stable braindecode names,
        falling back to the last child that actually contains a Linear.

        The old fallback grabbed the literal last named child, which after
        Opacus's ModuleValidator.fix (BatchNorm -> GroupNorm) could be a
        norm/flatten/permute layer — silently hooking the wrong layer and
        mis-defining the embedding. Require a Linear so the hook captures the
        true pre-classifier features, and raise loudly otherwise.
        """
        for candidate in ("final_layer", "classifier", "fc", "head"):
            if hasattr(net, candidate):
                return getattr(net, candidate)
        named = list(net.named_children())
        if not named:
            raise RuntimeError("EEGNet has no children; can't locate head")
        for _name, mod in reversed(named):
            if isinstance(mod, nn.Linear):
                return mod
            if any(isinstance(m, nn.Linear) for m in mod.modules()):
                return mod
        raise RuntimeError(
            "EEGNet: could not locate a classifier head (no Linear child found)."
        )

