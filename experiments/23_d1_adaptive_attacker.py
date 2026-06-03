"""D1 adaptive attacker — does ad-hoc input transformation hold under
encoder fine-tune?

Symmetry experiment to 15 (D2 adaptive) and 18 (D3 adaptive). The D1
results in experiments/07 (PCA), 11 (noise + channel-drop) were
measured against a generic logreg probe. We re-run the same end-to-end
encoder-fine-tune attack against one representative defense point per
D1 family:

    - PCA k=8                (07_d1_pca: top1 generic ≈ 0.36)
    - additive noise σ=1.0   (11_d1_noise: top1 generic ≈ 0.34)
    - channel-drop k=8       (11_d1_channel_drop: top1 generic ≈ 0.30)

Note: D1 is "input transform applied before the victim sees the
windows" — so the threat model under adaptive attack is "the attacker
knows the transform, has access to the same training data, and can
re-train EEGNet on the transformed inputs end-to-end on subject-id".
That's exactly the encoder-fine-tune protocol from experiment 15
applied to a victim trained on transformed inputs.

Pipeline per defense point:
    1. Apply D1 transform to imagery train/test windows.
    2. Train an EEGNet (4-class motor-imagery) victim on transformed train.
    3. Run attack 1 (logreg on frozen EEGNet embeddings) — sanity check
       against the generic-attacker baseline reported in 07/11.
    4. Run attack 3 (encoder fine-tune from the trained EEGNet weights,
       same protocol as experiment 15).

Expected behaviour: D1's ad-hoc transforms reduce generic-attacker
top-1 modestly. Under encoder fine-tune the encoder sees richly-
informative subject-specific features anyway and recovers leakage.
The per-defense delta tells us how much of D1's privacy was real
vs an artifact of the generic attacker.
"""
from __future__ import annotations

import argparse
import copy
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
from defenses.adhoc import ChannelDrop, ChannelGaussianNoise, ChannelPCA
from eval.bootstrap import grouped_bootstrap_ci
from models.eegnet import EEGNetVictim
from preprocess.windows import WindowedDataset, windowed_subjects

VICTIM_TRAIN_RUNS = (4, 6, 8, 10)
VICTIM_TEST_RUNS = (12, 14)


# ---------------------------------------------------------------------------
# Encoder fine-tune attack on a vanilla EEGNet (no DANN backbone)
# ---------------------------------------------------------------------------
class _FineTunedReID(nn.Module):
    """Vanilla EEGNet (warm-started from victim weights) + linear re-ID head.

    Mirrors the structure used in experiments/15 and 18, but for the
    plain `EEGNetVictim` that backs the D1 ad-hoc victims (after the
    input transform has been applied to the data).
    """

    def __init__(self, source_module: nn.Module, n_subjects: int,
                 *, input_scale: float = 1e6) -> None:
        super().__init__()
        self.input_scale = input_scale
        head_attr = None
        for candidate in ("final_layer", "classifier", "fc", "head"):
            if hasattr(source_module, candidate):
                head_attr = candidate
                break
        if head_attr is None:
            raise RuntimeError("Cannot locate classifier head on EEGNet")
        self.backbone = copy.deepcopy(source_module)
        setattr(self.backbone, head_attr, nn.Identity())
        self._head_attr = head_attr
        self._n_subjects = n_subjects
        self.head: nn.Module | None = None  # built lazily

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


def _fine_tune(victim: EEGNetVictim, X_train, y_train,
               *, device, n_epochs=20, lr=5e-4, batch_size=64, seed=0):
    torch.manual_seed(seed)
    sid_to_idx = {int(s): i for i, s in enumerate(sorted(set(int(x) for x in y_train)))}
    y_idx = np.array([sid_to_idx[int(s)] for s in y_train], dtype=np.int64)
    n_classes = len(sid_to_idx)
    src = victim.model_
    model = _FineTunedReID(src, n_classes,
                           input_scale=victim.input_scale).to(device)
    # Warm up the lazy head with one dummy forward
    with torch.no_grad():
        dummy = torch.zeros(2, victim.n_channels, victim.n_times,
                            device=device, dtype=torch.float32)
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
# D1 transforms
# ---------------------------------------------------------------------------
def _apply_d1(transform_name: str, X_train: np.ndarray, X_test: np.ndarray,
              *, seed: int):
    if transform_name == "pca_k8":
        d = ChannelPCA(k=8); d.fit(X_train); return d.transform(X_train), d.transform(X_test)
    if transform_name == "noise_sigma1.0":
        d = ChannelGaussianNoise(sigma=1.0, seed=seed); d.fit(X_train)
        return d.transform(X_train), d.transform(X_test)
    if transform_name == "channel_drop_k8":
        d = ChannelDrop(k=8); d.fit(X_train); return d.transform(X_train), d.transform(X_test)
    raise ValueError(transform_name)


def _datasets_with_transformed_X(orig: WindowedDataset, X_new: np.ndarray) -> WindowedDataset:
    return WindowedDataset(
        X=X_new, y=orig.y, subject_ids=orig.subject_ids,
        trial_ids=orig.trial_ids, run_ids=orig.run_ids,
        sfreq=orig.sfreq, channel_names=orig.channel_names,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveResult:
    defense: str
    attack: str
    top1: float
    top1_ci_low: float
    top1_ci_high: float
    top5: float
    top10: float
    n_test_windows: int
    n_subjects: int
    chance_top1: float
    task_acc: float


DEFENSES = ("pca_k8", "noise_sigma1.0", "channel_drop_k8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--defenses", nargs="+", default=list(DEFENSES))
    p.add_argument("--eegnet-epochs", type=int, default=80)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.eegnet_epochs = 30
        args.ft_epochs = 5
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"D1 defenses: {args.defenses}", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    full = windowed_subjects(subjects, runs="imagery")
    train_orig = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test_orig = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  train={train_orig.n_windows}  test={test_orig.n_windows}  "
          f"chans={train_orig.n_channels}\n", flush=True)

    all_results: list[AdaptiveResult] = []
    for defense in args.defenses:
        print(f"=== {defense} ===", flush=True)
        Xtr, Xte = _apply_d1(defense, train_orig.X, test_orig.X, seed=args.seed)
        train = _datasets_with_transformed_X(train_orig, Xtr)
        test = _datasets_with_transformed_X(test_orig, Xte)
        print(f"  transformed shapes: train={train.X.shape}  test={test.X.shape}",
              flush=True)

        # Train EEGNet on transformed inputs
        t0 = time.time()
        victim = EEGNetVictim(
            n_channels=train.n_channels, n_times=train.n_times, n_classes=4,
            n_epochs=args.eegnet_epochs, seed=args.seed, verbose=False,
        )
        victim.fit(train.X, train.y)
        task_acc = float(victim.score(test.X, test.y))
        print(f"  victim trained in {time.time()-t0:.1f}s  "
              f"task_acc={task_acc:.3f}", flush=True)

        n_subj = len(np.unique(train.subject_ids))

        # Attack 1: logreg probe (sanity vs generic baseline)
        t0 = time.time()
        a1 = closed_set_reid(victim, train, test, probes=("logreg",),
                             bootstrap_n=1000, seed=args.seed)[0]
        all_results.append(AdaptiveResult(
            defense=defense, attack="logreg_probe",
            top1=a1.top1, top1_ci_low=a1.top1_ci_low, top1_ci_high=a1.top1_ci_high,
            top5=a1.top5, top10=a1.top10,
            n_test_windows=a1.n_test_windows, n_subjects=a1.n_subjects,
            chance_top1=a1.chance_top1, task_acc=task_acc,
        ))
        print(f"  logreg : top1={a1.top1:.3f} "
              f"[{a1.top1_ci_low:.3f}, {a1.top1_ci_high:.3f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)

        # Attack 3: encoder fine-tune
        t0 = time.time()
        ft_model, sid_to_idx = _fine_tune(
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
        all_results.append(AdaptiveResult(
            defense=defense, attack="encoder_finetune",
            top1=ci.point, top1_ci_low=ci.low, top1_ci_high=ci.high,
            top5=top5, top10=top10,
            n_test_windows=int(test.n_windows), n_subjects=int(n_subj),
            chance_top1=1.0/n_subj, task_acc=task_acc,
        ))
        print(f"  fine-tune: top1={ci.point:.3f} "
              f"[{ci.low:.3f}, {ci.high:.3f}]  "
              f"({time.time()-t0:.0f}s)", flush=True)
        print()

    out = {"defenses": [r.__dict__ for r in all_results]}
    out_path = RESULTS_DIR / "23_d1_adaptive_attacker.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Results written to {out_path}\n")

    print("| Defense | Attack | Top-1 (95% CI) | Task acc |")
    print("|---|---|---|---|")
    for r in all_results:
        ci_s = f"{r.top1:.3f} [{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]"
        print(f"| {r.defense} | {r.attack} | {ci_s} | {r.task_acc:.3f} |")


if __name__ == "__main__":
    main()

