"""D4 — Federated DP-SGD on EEGNet (central-DP FedAvg).

The threat model targets on-device BCI inference. Centralised DP-SGD
trains a single model on a pooled cohort; in a deployment world that
pooling step is where users' EEG actually leaves the device. Federated
learning inverts the data-flow: each user trains locally on their own
device, the server only sees model updates.

This victim implements the canonical "FedAvg with central DP" protocol
(Geyer et al. 2017; McMahan et al. 2017): each round samples a random
subset of clients, each client does K local SGD epochs on its own data,
the server receives the (w_local - w_global) update from each client,
clips per-client update norm, sums, adds Gaussian noise, and averages.

The (ε, δ) bound is at the CLIENT level (per-user privacy), not the
per-sample level. This matches the BCI deployment scenario: each user
contributes once, on their own data, and the privacy claim is about
whether the trained-model API leaks the existence or characteristics
of that user.

We expose the same VictimModel API as the other defenses so the rest
of the attack code paths consume it unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import EEGNet

from models.base import VictimModel


def _pick_device(prefer: str = "auto") -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and prefer != "cuda":
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class FedRoundLog:
    round_idx: int
    n_participants: int
    pre_clip_norms_mean: float
    pre_clip_norms_max: float
    post_clip_norms_mean: float
    server_noise_sigma: float
    server_lr: float


class FederatedDPVictim(VictimModel):
    """EEGNet trained via FedAvg with central per-client DP noise.

    The simulation faithfully implements the noise mechanism (per-client
    clip to C, sum, add N(0, (σC)^2), average). The defense should be read
    as EMPIRICAL: the reported (ε, δ) is an informal annotation, NOT an
    enforced guarantee.

    Why informal: with the default `poisson_clients=False` the per-round
    cohort is sampled to a FIXED size without replacement, whereas the RDP
    accountant in `rdp_epsilon()` assumes Poisson subsampling (each client
    included independently w.p. q). The two have different privacy curves,
    so the fixed-size ε is a heuristic, not a sound bound. Set
    `poisson_clients=True` to sample clients via independent Bernoulli(q),
    which matches the accountant's assumption and makes `rdp_epsilon()` a
    valid participant-level bound (at some utility cost from variable
    cohort size). The empirical re-ID / fine-tune protection is the
    substantive result either way.
    """
    name = "eegnet_federated_dp"

    def __init__(
        self,
        *,
        n_channels: int,
        n_times: int,
        n_classes: int = 4,
        n_rounds: int = 30,
        participant_fraction: float = 0.5,
        local_epochs: int = 3,
        local_lr: float = 5e-3,
        local_batch_size: int = 32,
        server_lr: float = 1.0,
        clip_norm: float = 1.0,
        noise_sigma: float = 0.4,
        poisson_clients: bool = False,
        device: str = "auto",
        seed: int = 0,
        input_scale: float = 1e6,
        verbose: bool = False,
    ) -> None:
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.n_rounds = n_rounds
        self.participant_fraction = participant_fraction
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.local_batch_size = local_batch_size
        self.server_lr = server_lr
        self.clip_norm = clip_norm
        self.noise_sigma = noise_sigma
        self.poisson_clients = poisson_clients
        self.device = _pick_device(device)
        self.seed = seed
        self.input_scale = input_scale
        self.verbose = verbose
        self.model_: nn.Module | None = None
        self.round_log_: list[FedRoundLog] = []

    def _build(self) -> nn.Module:
        torch.manual_seed(self.seed)
        # GroupNorm so the architectural side-effect matches D3 — this
        # makes federated vs centralised DP directly comparable.
        net = EEGNet(n_chans=self.n_channels, n_outputs=self.n_classes,
                     n_times=self.n_times)
        import torch.nn.utils.parametrize as parametrize
        for module in net.modules():
            if hasattr(module, "parametrizations"):
                for param_name in list(module.parametrizations.keys()):
                    parametrize.remove_parametrizations(
                        module, param_name, leave_parametrized=True,
                    )
        from opacus.validators import ModuleValidator
        net = ModuleValidator.fix(net)
        return net.to(self.device)

    def _train_one_client_local(self, client_X, client_y, *,
                                rng: np.random.Generator) -> dict[str, torch.Tensor]:
        """Run K local SGD epochs on a single client; return the delta."""
        client_model = self._build()
        # Load current global weights
        client_model.load_state_dict(self.model_.state_dict())
        client_model.train()
        opt = torch.optim.SGD(client_model.parameters(), lr=self.local_lr,
                              momentum=0.0)
        ce = nn.CrossEntropyLoss()
        n = len(client_X)
        Xt = torch.from_numpy(client_X.astype(np.float32, copy=False))
        yt = torch.from_numpy(client_y).long()
        for _ in range(self.local_epochs):
            idx = rng.permutation(n)
            for i in range(0, n, self.local_batch_size):
                sl = idx[i:i + self.local_batch_size]
                if len(sl) < 2:
                    continue
                xb = (Xt[sl] * self.input_scale).to(self.device)
                yb = yt[sl].to(self.device)
                opt.zero_grad()
                logits = client_model(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()
        # Compute delta (client_local_weights - global_weights)
        global_sd = self.model_.state_dict()
        client_sd = client_model.state_dict()
        delta = {k: client_sd[k] - global_sd[k] for k in global_sd}
        return delta

    @staticmethod
    def _flatten_delta(delta: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([d.reshape(-1) for d in delta.values()])

    def _clip_in_place(self, delta: dict[str, torch.Tensor]) -> tuple[float, float]:
        flat = self._flatten_delta(delta)
        nrm = float(torch.linalg.vector_norm(flat).item())
        scale = min(1.0, self.clip_norm / max(nrm, 1e-12))
        if scale < 1.0:
            for k in delta:
                delta[k] = delta[k] * scale
        post = nrm * scale
        return nrm, post

    def fit(self, X: np.ndarray, y: np.ndarray,
            client_ids: np.ndarray) -> FederatedDPVictim:
        """Fit via FedAvg with central per-client DP noise.

        `client_ids` is a per-window array assigning each window to a
        client (e.g. subject_id from the WindowedDataset).
        """
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert client_ids.shape[0] == X.shape[0]

        # Initialise global
        self.model_ = self._build()

        clients = np.asarray(sorted(int(c) for c in np.unique(client_ids)))
        n_clients = len(clients)
        n_per_round = max(1, int(round(self.participant_fraction * n_clients)))

        rng = np.random.default_rng(self.seed)
        global_sd = {k: v.clone() for k, v in self.model_.state_dict().items()}

        for r in range(self.n_rounds):
            if self.poisson_clients:
                # Independent Bernoulli(q) inclusion — matches the Poisson
                # subsampling the RDP accountant assumes, so rdp_epsilon() is
                # a valid bound under this mode.
                incl = rng.random(n_clients) < self.participant_fraction
                chosen = sorted(int(c) for c in clients[incl])
                if not chosen:
                    continue
            else:
                # Fixed-size sampling without replacement (default). Note this
                # does NOT satisfy the accountant's Poisson assumption, so the
                # reported epsilon is an informal annotation only.
                chosen = sorted(int(c) for c in rng.choice(clients, size=n_per_round,
                                                          replace=False))
            # Run each client locally
            deltas: list[dict[str, torch.Tensor]] = []
            pre_norms = []
            post_norms = []
            for c in chosen:
                mask = client_ids == c
                if int(mask.sum()) < 2:
                    continue
                delta = self._train_one_client_local(
                    X[mask], y[mask],
                    rng=np.random.default_rng(self.seed + r * 1000 + c),
                )
                pre, post = self._clip_in_place(delta)
                pre_norms.append(pre)
                post_norms.append(post)
                deltas.append(delta)
            if not deltas:
                continue
            # Aggregate: sum deltas, add Gaussian noise per parameter,
            # divide by participant count, scale by server_lr.
            agg_sd = {k: torch.zeros_like(global_sd[k]) for k in global_sd}
            for d in deltas:
                for k in agg_sd:
                    agg_sd[k] = agg_sd[k] + d[k]
            sigma_eff = self.noise_sigma * self.clip_norm
            for k in agg_sd:
                # Skip BatchNorm/GroupNorm running stats (Long buffers
                # that don't carry gradient — noise injection breaks them).
                if agg_sd[k].dtype not in (torch.float32, torch.float16,
                                           torch.float64, torch.bfloat16):
                    continue
                noise = torch.randn_like(agg_sd[k]) * sigma_eff
                agg_sd[k] = (agg_sd[k] + noise) / len(deltas)
            # Update global
            for k in global_sd:
                global_sd[k] = global_sd[k] + self.server_lr * agg_sd[k]
            self.model_.load_state_dict(global_sd, strict=False)

            self.round_log_.append(FedRoundLog(
                round_idx=int(r), n_participants=int(len(deltas)),
                pre_clip_norms_mean=float(np.mean(pre_norms)) if pre_norms else 0.0,
                pre_clip_norms_max=float(np.max(pre_norms)) if pre_norms else 0.0,
                post_clip_norms_mean=float(np.mean(post_norms)) if post_norms else 0.0,
                server_noise_sigma=float(sigma_eff),
                server_lr=float(self.server_lr),
            ))
            if self.verbose and (r % 5 == 0 or r == self.n_rounds - 1):
                print(f"  round {r:3d}  n_clients={len(deltas):3d}  "
                      f"mean_norm_pre={np.mean(pre_norms):.3f}  "
                      f"mean_norm_post={np.mean(post_norms):.3f}",
                      flush=True)
        return self

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None
        self.model_.eval()
        out = []
        for i in range(0, len(X), 256):
            xb = torch.from_numpy(X[i:i + 256] * self.input_scale).to(self.device)
            out.append(self.model_(xb).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def embed(self, X: np.ndarray) -> np.ndarray:
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        assert self.model_ is not None
        head = None
        for candidate in ("final_layer", "classifier", "fc", "head"):
            if hasattr(self.model_, candidate):
                head = getattr(self.model_, candidate)
                break
        if head is None:
            # Require a layer that contains a Linear rather than blindly taking
            # the last child (post-ModuleValidator.fix that could be GroupNorm).
            for _name, mod in reversed(list(self.model_.named_children())):
                if isinstance(mod, nn.Linear) or any(
                    isinstance(m, nn.Linear) for m in mod.modules()
                ):
                    head = mod
                    break
        if head is None:
            raise RuntimeError("Federated-EEGNet: cannot locate classifier head")
        captured: list[torch.Tensor] = []

        def hook(_module, inputs, _output):
            captured.append(inputs[0].detach().cpu())
        handle = head.register_forward_hook(hook)
        self.model_.eval()
        try:
            outs = []
            for i in range(0, len(X), 256):
                xb = torch.from_numpy(X[i:i + 256] * self.input_scale).to(self.device)
                self.model_(xb)
                outs.append(captured.pop().numpy())
            emb = np.concatenate(outs, axis=0)
        finally:
            handle.remove()
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        return emb.astype(np.float32, copy=False)

    def rdp_epsilon(self, delta: float = 1e-5) -> float:
        """Participant-level (epsilon, delta) annotation via Opacus's RDP accountant.

        Composes, over n_rounds, the Renyi-DP curve of a sub-sampled Gaussian
        mechanism with noise_multiplier=noise_sigma and sample_rate=q, then
        converts to (epsilon, delta) (Mironov 2017 / Opacus.RDPAccountant).

        IMPORTANT — read as informal unless `poisson_clients=True`. The RDP
        subsampled-Gaussian curve assumes each client is included independently
        with probability q (Poisson sampling). With the default fixed-size
        cohort (`poisson_clients=False`) the sampling distribution differs, so
        this number is a heuristic annotation, NOT an enforced bound. Under
        `poisson_clients=True` the assumption holds and the returned epsilon is
        a valid participant-level bound.
        """
        try:
            from opacus.accountants import RDPAccountant
        except Exception:
            # Fallback to the analytic composition bound if opacus is missing.
            R = max(self.n_rounds, 1)
            q = self.participant_fraction
            sigma = max(self.noise_sigma, 1e-6)
            return math.sqrt(2.0 * R * q * q * math.log(1.0 / delta)) / sigma

        accountant = RDPAccountant()
        for _ in range(self.n_rounds):
            accountant.step(noise_multiplier=float(self.noise_sigma),
                            sample_rate=float(self.participant_fraction))
        return float(accountant.get_epsilon(delta=delta))

    def informal_epsilon_estimate(self, delta: float = 1e-5) -> float:
        """Coarse analytic Gaussian-composition estimate: sqrt(2 R q^2 ln(1/δ))/σ.

        Looser than `rdp_epsilon` and, like it, only a heuristic annotation —
        the fixed-size (non-Poisson) default sampling means neither number is
        an enforced (epsilon, delta) guarantee. Kept distinct (it previously
        just aliased `rdp_epsilon`, which was misleading) so the reported
        'simple Gaussian bound' is actually the simple bound.
        """
        R = max(self.n_rounds, 1)
        q = self.participant_fraction
        sigma = max(self.noise_sigma, 1e-6)
        return math.sqrt(2.0 * R * q * q * math.log(1.0 / delta)) / sigma
