"""D3 — Differentially private EEGNet via Opacus DP-SGD.

Trains EEGNet under (ε, δ)-differential privacy using Opacus's
PrivacyEngine.make_private_with_epsilon API. ModuleValidator.fix is run
first to swap any Opacus-incompatible layers (e.g. BatchNorm → GroupNorm).

The defender's claim under DP-SGD is mechanism-level and rigorous: no
adversary, even with arbitrary side information and unbounded compute,
can determine the contribution of any single training sample with
likelihood-ratio better than e^ε. ε ≤ 1 is "strong" privacy; ε ∈ [3, 10]
is "loose"; ε → ∞ recovers vanilla SGD.

A1 closed-set re-ID is then run against the DP-trained encoder's
embeddings to measure how much identity *empirically* survives the
privacy mechanism.

Caveat when comparing to the A1 headline baseline: this victim is
GroupNorm + plain SGD (Opacus requires GroupNorm, and DP-SGD uses SGD),
whereas the A1 baseline is AdamW + BatchNorm. That architecture swap
alone lowers re-ID a lot, independent of any privacy noise. The clean
"how much is the privacy vs the architecture" decomposition is the
target_epsilon=None arm here and the dedicated ablation in
experiments/19_dp_sgd_arch_ablation.py — do not read the gap between this
no-DP baseline and the AdamW+BatchNorm A1 baseline as a privacy effect.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet
from torch.utils.data import DataLoader, TensorDataset

from models.base import VictimModel


def _pick_device(prefer: str = "auto") -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and prefer != "cuda":
        return torch.device("mps")
    return torch.device("cpu")


class DPSGDVictim(VictimModel):
    """EEGNet trained with Opacus DP-SGD at a target (ε, δ).

    Pass `target_epsilon=None` to train without DP — used as the
    "no-defense" baseline in the same code path.
    """
    name = "eegnet_dpsgd"

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_classes: int = 4,
        n_epochs: int = 40,
        batch_size: int = 256,
        lr: float = 1.0,
        target_epsilon: float | None = 3.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        device: str = "auto",
        seed: int = 0,
        input_scale: float = 1e6,
        verbose: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.device = _pick_device(device)
        self.seed = seed
        self.input_scale = input_scale
        self.verbose = verbose
        self.model_: nn.Module | None = None
        self.final_epsilon_: float | None = None

    def _build(self) -> nn.Module:
        torch.manual_seed(self.seed)
        net = EEGNet(n_chans=self.n_channels, n_outputs=self.n_classes,
                     n_times=self.n_times)
        # braindecode's EEGNet wraps the spatial conv's weight in
        # nn.utils.parametrize (a max-norm constraint). Parametrized modules
        # don't pickle, and Opacus's ModuleValidator.fix needs to pickle to
        # clone the module. Strip parametrizations and bake the (now-constant)
        # parametrized weights into the underlying parameters before handing
        # to Opacus. Loses the max-norm constraint, but DP-SGD's per-sample
        # gradient clipping is itself a strong regularizer, so this is fine.
        import torch.nn.utils.parametrize as parametrize
        for module in net.modules():
            if hasattr(module, "parametrizations"):
                for param_name in list(module.parametrizations.keys()):
                    parametrize.remove_parametrizations(
                        module, param_name, leave_parametrized=True,
                    )
        # Now safe to swap incompatible layers (BN -> GN) and clone for DP.
        from opacus.validators import ModuleValidator
        net = ModuleValidator.fix(net)
        ModuleValidator.validate(net, strict=True)
        return net.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> DPSGDVictim:
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        net = self._build()
        # Build DataLoader for Opacus (it needs the loader to inject the
        # noisy sampler).
        ds = TensorDataset(
            torch.from_numpy(X * self.input_scale),
            torch.from_numpy(y).long(),
        )
        # Opacus prefers a regular sampler; it will replace it.
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                            drop_last=False)

        # DP-SGD typically uses SGD with high lr (Opacus default suggestion);
        # AdamW with DP is unstable.
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.0)

        if self.target_epsilon is not None:
            from opacus import PrivacyEngine
            engine = PrivacyEngine(secure_mode=False)
            net, optimizer, loader = engine.make_private_with_epsilon(
                module=net, optimizer=optimizer, data_loader=loader,
                epochs=self.n_epochs,
                target_epsilon=float(self.target_epsilon),
                target_delta=float(self.target_delta),
                max_grad_norm=self.max_grad_norm,
                poisson_sampling=True,
            )
            self._engine = engine
        else:
            self._engine = None

        ce = nn.CrossEntropyLoss()
        net.train()
        for epoch in range(self.n_epochs):
            running = 0.0
            n = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = ce(logits, yb)
                loss.backward()
                optimizer.step()
                running += loss.item() * len(xb)
                n += len(xb)
            if self.verbose and (epoch % 10 == 0 or epoch == self.n_epochs - 1):
                if self._engine is not None:
                    eps = self._engine.get_epsilon(self.target_delta)
                    print(f"  epoch {epoch:3d}  loss={running/n:.3f}  ε={eps:.2f}",
                          flush=True)
                else:
                    print(f"  epoch {epoch:3d}  loss={running/n:.3f}", flush=True)

        self.model_ = net
        if self._engine is not None:
            self.final_epsilon_ = float(self._engine.get_epsilon(self.target_delta))
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None
        self.model_.eval()
        out = []
        for start in range(0, len(X), 256):
            xb = torch.from_numpy(X[start:start + 256] * self.input_scale).to(self.device)
            out.append(self.model_(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        """Pre-classifier features. Hooks the head module and captures its
        input — same approach as plain EEGNetVictim."""
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None
        # Locate the head; ModuleValidator.fix may have wrapped/renamed layers,
        # so the fallback must find a layer that actually contains a Linear
        # rather than blindly taking the last child (which could be a GroupNorm
        # and silently mis-define the embedding).
        head = None
        for candidate in ("final_layer", "classifier", "fc", "head"):
            if hasattr(self.model_, candidate):
                head = getattr(self.model_, candidate)
                break
        if head is None:
            for _name, mod in reversed(list(self.model_.named_children())):
                if isinstance(mod, nn.Linear) or any(
                    isinstance(m, nn.Linear) for m in mod.modules()
                ):
                    head = mod
                    break
        if head is None:
            raise RuntimeError("DP-EEGNet: cannot locate classifier head for embedding hook")

        captured: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            captured.append(inputs[0].detach().cpu())

        handle = head.register_forward_hook(hook)
        self.model_.eval()
        try:
            outs = []
            for start in range(0, len(X), 256):
                xb = torch.from_numpy(X[start:start + 256] * self.input_scale).to(self.device)
                self.model_(xb)
                outs.append(captured.pop().numpy())
            emb = np.concatenate(outs, axis=0)
        finally:
            handle.remove()
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        return emb.astype(np.float32, copy=False)
