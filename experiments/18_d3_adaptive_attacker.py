"""D3 adaptive attacker — does DP-SGD ε=3 hold against a strong attacker?

Mirror of experiments/15 for the DP-SGD victim. The earlier D3 result
(ε=3: re-ID top-1 ≈ 2%) was measured against a generic logreg probe.
DANN looked the same — and collapsed under fine-tune. Formal DP should
NOT collapse under fine-tune, because the (ε, δ) bound is attacker-
agnostic by construction. This experiment empirically tests that.

Three attacks against the same DP-SGD-trained EEGNet (ε=3):
  attack 1 — logreg probe        (frozen-encoder baseline)
  attack 2 — deep MLP probe      (frozen encoder, higher-capacity)
  attack 3 — encoder fine-tune   (warm-start from the DP weights, train
                                  end-to-end on subject-id; same protocol
                                  as experiment 15)

Expected behaviour: if formal DP holds empirically, all three attacks
stay near the no-defense-but-GN baseline; if attack 3 jumps back up
toward the no-DP-no-GN baseline (~41%), there's a leak the formal
guarantee doesn't capture (it shouldn't, but we test).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import top_k_accuracy_score

from attacks.closed_set import closed_set_reid
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.dp_sgd import DPSGDVictim
from eval.bootstrap import grouped_bootstrap_ci
from preprocess.windows import windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


# ---------------------------------------------------------------------------
# Attack 2: deep MLP probe on the FROZEN encoder
# ---------------------------------------------------------------------------
class _DeepMLPProbe(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256),    nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def _train_deep_mlp(Z_train, y_train, *, n_classes, device, n_epochs=60, lr=1e-3,
                    batch_size=256, seed=0):
    torch.manual_seed(seed)
    n, d = Z_train.shape
    sid_to_idx = {int(s): i for i, s in enumerate(sorted(set(int(x) for x in y_train)))}
    y_idx = np.array([sid_to_idx[int(s)] for s in y_train], dtype=np.int64)
    Z = torch.from_numpy(Z_train).float()
    y = torch.from_numpy(y_idx).long()
    model = _DeepMLPProbe(d, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    model.train()
    for _epoch in range(n_epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch_size):
            sl = idx[i:i + batch_size]
            zb = Z[sl].to(device)
            yb = y[sl].to(device)
            opt.zero_grad()
            logits = model(zb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
    return model, sid_to_idx


@torch.no_grad()
def _deep_mlp_predict_with_proba(model, Z_test, sid_to_idx, *, device, n_classes):
    inv = {v: k for k, v in sid_to_idx.items()}
    model.eval()
    Z = torch.from_numpy(Z_test).float().to(device)
    logits = model(Z)
    proba = F.softmax(logits, dim=1).cpu().numpy()
    pred_idx = logits.argmax(dim=1).cpu().numpy()
    sorted_classes = np.array([inv[i] for i in range(n_classes)])
    preds = np.array([inv[int(i)] for i in pred_idx])
    return preds, proba, sorted_classes


# ---------------------------------------------------------------------------
# Attack 3: encoder fine-tune on the DP-trained backbone
# ---------------------------------------------------------------------------
def _strip_opacus_hooks(module: nn.Module) -> None:
    """Recursively clear all forward / backward hooks on a module tree.

    Opacus's `GradSampleModule` registers per-sample-gradient hooks on the
    inner module. When we deepcopy that inner module to fine-tune from its
    weights, the hooks come along — and they reference per-parameter
    state (`_forward_counter`) that exists only on the GradSampleModule-
    wrapped parameters. Any forward pass through the copy then crashes
    with `'Parameter' object has no attribute '_forward_counter'`.

    We never need those hooks during the adaptive attack — we want a
    clean autograd graph for end-to-end backprop on subject-id labels.
    Strip them before training.
    """
    for m in module.modules():
        m._forward_hooks.clear()
        m._backward_hooks.clear()
        if hasattr(m, "_forward_pre_hooks"):
            m._forward_pre_hooks.clear()
        if hasattr(m, "_full_backward_hooks"):
            m._full_backward_hooks.clear()
        if hasattr(m, "_full_backward_pre_hooks"):
            m._full_backward_pre_hooks.clear()


def _clean_groupnorm_eegnet_from_victim(victim: DPSGDVictim) -> nn.Module:
    """Rebuild a fresh GroupNorm-EEGNet (architecture-equivalent to the
    one DP-SGD trained) and load the trained weights into it.

    This sidesteps Opacus's hook-bookkeeping entirely: we don't deepcopy
    the wrapped module, we construct a clean one and copy weights. Fixes
    the `_forward_counter` AttributeError under torch 2.10 + opacus.
    """
    # _build() returns a fresh GroupNorm-EEGNet (after parametrize-strip
    # and ModuleValidator.fix), already on victim.device.
    fresh = victim._build()
    # Trained model_'s state dict — GradSampleModule wrapping prefixes
    # everything with `_module.`; strip it.
    sd = victim.model_.state_dict()
    cleaned: dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith("_module."):
            cleaned[k[len("_module."):]] = v
        else:
            cleaned[k] = v
    # Drop Opacus's `_loss_reduction` / similar non-parameter entries
    # by loading non-strict.
    missing, unexpected = fresh.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"  [fine-tune-init] missing keys (not present in DP weights): "
              f"{missing[:3]}{'...' if len(missing) > 3 else ''}", flush=True)
    if unexpected:
        # Filter out Opacus bookkeeping that we expect to be unexpected.
        relevant = [k for k in unexpected if not k.startswith(("_loss",))]
        if relevant:
            print(f"  [fine-tune-init] unexpected keys (Opacus extras, harmless): "
                  f"{relevant[:3]}{'...' if len(relevant) > 3 else ''}", flush=True)
    # Belt-and-braces: clear any hooks that snuck through (none expected
    # since fresh was just built, but cheap insurance).
    _strip_opacus_hooks(fresh)
    return fresh


class _FineTunedDPReID(nn.Module):
    """GroupNorm-EEGNet (warm-started from DP-SGD weights) + linear re-ID head.

    Grabs the head module that braindecode's EEGNet exposes (`final_layer`
    fallback to last named child), strips it, and attaches a new
    n_subjects-way classifier. The remaining backbone is fine-tuned end-
    to-end alongside the head.
    """

    def __init__(self, source_module: nn.Module, n_subjects: int,
                 *, input_scale: float = 1e6) -> None:
        super().__init__()
        self.input_scale = input_scale
        # Locate head by attribute (mirrors DPSGDVictim.embed)
        head_attr = None
        for candidate in ("final_layer", "classifier", "fc", "head"):
            if hasattr(source_module, candidate):
                head_attr = candidate
                break
        if head_attr is None:
            raise RuntimeError("Cannot locate classifier head on DP-EEGNet")
        # source_module is a *clean* rebuild (not a deepcopy of the
        # GradSampleModule's inner) so we can use it directly.
        self.backbone = source_module
        setattr(self.backbone, head_attr, nn.Identity())
        # Belt-and-braces clear in case any caller passes a raw inner
        # module by accident.
        _strip_opacus_hooks(self.backbone)
        self._head_attr = head_attr
        self._n_subjects = n_subjects
        self.head: nn.Module | None = None  # built lazily on first forward

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


def _fine_tune_dp_encoder(victim: DPSGDVictim, X_train, y_train,
                          *, device, n_epochs=20, lr=5e-4, batch_size=64,
                          seed=0):
    torch.manual_seed(seed)
    sid_to_idx = {int(s): i for i, s in enumerate(sorted(set(int(x) for x in y_train)))}
    y_idx = np.array([sid_to_idx[int(s)] for s in y_train], dtype=np.int64)
    n_classes = len(sid_to_idx)

    # Clean rebuild + state-dict load. Avoids Opacus hook contamination
    # that breaks deepcopy on torch 2.10+.
    src = _clean_groupnorm_eegnet_from_victim(victim)
    model = _FineTunedDPReID(src, n_classes, input_scale=victim.input_scale).to(device)
    # Warm up the lazy head with one dummy forward so the optimizer sees
    # all parameters from epoch 0.
    with torch.no_grad():
        dummy = torch.zeros(2, victim.n_channels, victim.n_times,
                            device=device, dtype=torch.float32)
        # GroupNorm needs train()-mode forward to be safe with small batches
        model.eval(); model(dummy); model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    Xt = torch.from_numpy(X_train.astype(np.float32, copy=False))
    yt = torch.from_numpy(y_idx)
    for _epoch in range(n_epochs):
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


@torch.no_grad()
def _fine_tune_predict(model, X_test, sid_to_idx, *, device, n_classes,
                       batch_size=256):
    inv = {v: k for k, v in sid_to_idx.items()}
    model.eval()
    chunks = []
    for i in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[i:i + batch_size].astype(np.float32, copy=False)).to(device)
        chunks.append(F.softmax(model(xb), dim=1).cpu().numpy())
    proba = np.concatenate(chunks, axis=0)
    pred_idx = proba.argmax(axis=1)
    sorted_classes = np.array([inv[i] for i in range(n_classes)])
    preds = np.array([inv[int(i)] for i in pred_idx])
    return preds, proba, sorted_classes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveResult:
    attack: str
    top1: float
    top1_ci_low: float
    top1_ci_high: float
    top5: float
    top10: float
    n_test_windows: int
    n_subjects: int
    chance_top1: float


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--target-epsilon", type=float, default=3.0,
                   help="DP-SGD target epsilon for the victim.")
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--dp-batch-size", type=int, default=256)
    p.add_argument("--dp-lr", type=float, default=1.0)
    p.add_argument("--dp-epochs", type=int, default=40)
    p.add_argument("--mlp-epochs", type=int, default=60)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.dp_epochs = 10
        args.mlp_epochs = 20
        args.ft_epochs = 5
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"DP-SGD target ε={args.target_epsilon}  δ={args.delta:.0e}  "
          f"epochs={args.dp_epochs}  max_grad_norm={args.max_grad_norm}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  train={train.n_windows}  test={test.n_windows}  "
          f"chans={train.n_channels}\n", flush=True)

    # ---- Step 1: train the DP-SGD victim being attacked ----
    print(f"=== training DP-SGD victim (ε={args.target_epsilon}) ===", flush=True)
    t0 = time.time()
    victim = DPSGDVictim(
        n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
        n_epochs=args.dp_epochs, batch_size=args.dp_batch_size, lr=args.dp_lr,
        target_epsilon=args.target_epsilon, target_delta=args.delta,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed, verbose=False,
    )
    victim.fit(train.X, train.y)
    task_acc = victim.score(test.X, test.y)
    final_eps = victim.final_epsilon_
    print(f"  trained in {time.time() - t0:.1f}s  task_acc(test)={task_acc:.3f}  "
          f"final_ε={final_eps:.2f}", flush=True)

    Z_train = victim.embed(train.X)
    Z_test = victim.embed(test.X)
    n_subj = len(np.unique(train.subject_ids))

    results: list[AdaptiveResult] = []

    # ---- Attack 1: standard logreg ----
    print("\n=== attack 1: logreg probe (frozen DP encoder) ===", flush=True)
    t0 = time.time()
    a1 = closed_set_reid(victim, train, test, probes=("logreg",),
                         bootstrap_n=1000, seed=args.seed)[0]
    results.append(AdaptiveResult(
        attack="logreg_probe",
        top1=a1.top1, top1_ci_low=a1.top1_ci_low, top1_ci_high=a1.top1_ci_high,
        top5=a1.top5, top10=a1.top10,
        n_test_windows=a1.n_test_windows, n_subjects=a1.n_subjects,
        chance_top1=a1.chance_top1,
    ))
    print(f"  top1 = {a1.top1:.3f} [{a1.top1_ci_low:.3f}, {a1.top1_ci_high:.3f}]  "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- Attack 2: deep MLP probe ----
    print("\n=== attack 2: deep MLP probe (frozen DP encoder) ===", flush=True)
    t0 = time.time()
    mlp, sid_to_idx = _train_deep_mlp(
        Z_train, train.subject_ids, n_classes=n_subj,
        device=device, n_epochs=args.mlp_epochs, seed=args.seed,
    )
    preds, proba, sorted_classes = _deep_mlp_predict_with_proba(
        mlp, Z_test, sid_to_idx, device=device, n_classes=n_subj,
    )
    correct = (preds == test.subject_ids).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=test.trial_ids, statistic=np.mean,
                              n_resamples=1000, seed=args.seed)
    top5 = float(top_k_accuracy_score(test.subject_ids, proba, k=5,
                                       labels=sorted_classes))
    top10 = float(top_k_accuracy_score(test.subject_ids, proba, k=10,
                                        labels=sorted_classes))
    results.append(AdaptiveResult(
        attack="deep_mlp_probe", top1=ci.point,
        top1_ci_low=ci.low, top1_ci_high=ci.high, top5=top5, top10=top10,
        n_test_windows=int(test.n_windows), n_subjects=int(n_subj),
        chance_top1=1.0 / n_subj,
    ))
    print(f"  top1 = {ci.point:.3f} [{ci.low:.3f}, {ci.high:.3f}]  "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- Attack 3: encoder fine-tune ----
    print("\n=== attack 3: encoder fine-tune (end-to-end on DP weights) ===",
          flush=True)
    t0 = time.time()
    ft_model, sid_to_idx = _fine_tune_dp_encoder(
        victim, train.X, train.subject_ids,
        device=device, n_epochs=args.ft_epochs, seed=args.seed,
    )
    preds, proba, sorted_classes = _fine_tune_predict(
        ft_model, test.X, sid_to_idx, device=device, n_classes=n_subj,
    )
    correct = (preds == test.subject_ids).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=test.trial_ids, statistic=np.mean,
                              n_resamples=1000, seed=args.seed)
    top5 = float(top_k_accuracy_score(test.subject_ids, proba, k=5,
                                       labels=sorted_classes))
    top10 = float(top_k_accuracy_score(test.subject_ids, proba, k=10,
                                        labels=sorted_classes))
    results.append(AdaptiveResult(
        attack="encoder_finetune", top1=ci.point,
        top1_ci_low=ci.low, top1_ci_high=ci.high, top5=top5, top10=top10,
        n_test_windows=int(test.n_windows), n_subjects=int(n_subj),
        chance_top1=1.0 / n_subj,
    ))
    print(f"  top1 = {ci.point:.3f} [{ci.low:.3f}, {ci.high:.3f}]  "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- Persist ----
    out = {
        "victim": f"DPSGD_eps={args.target_epsilon}",
        "target_epsilon": float(args.target_epsilon),
        "final_epsilon": float(final_eps) if final_eps is not None else None,
        "delta": float(args.delta),
        "max_grad_norm": float(args.max_grad_norm),
        "task_acc": float(task_acc),
        "attacks": [r.__dict__ for r in results],
    }
    out_path = RESULTS_DIR / "18_d3_adaptive_attacker.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}\n")

    print("| Attack | Top-1 (95% CI) | Top-5 | Top-10 |")
    print("|---|---|---|---|")
    for r in results:
        ci_s = f"{r.top1:.3f} [{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]"
        print(f"| {r.attack} | {ci_s} | {r.top5:.3f} | {r.top10:.3f} |")


if __name__ == "__main__":
    main()

