"""D2 adaptive attacker — does the DANN defense survive an attacker who
*knows* the defense exists?

The original D2 result (λ=0.2: re-ID drops to 21.5%) used a single L2-
regularized logistic regression probe on the encoder's penultimate
features. That's a generic attacker — it doesn't know about DANN. We
test three increasingly strong adaptive attackers, all white-box on the
trained DANN encoder:

  attack 1 — logreg probe         (= the original, baseline)
  attack 2 — deep MLP probe       on the FROZEN encoder; same input
                                  features but a much higher-capacity
                                  classifier (3 hidden layers, BatchNorm,
                                  dropout, ReLU)
  attack 3 — encoder fine-tune    encoder is initialized from the DANN
                                  weights and fine-tuned end-to-end on
                                  subject-id classification. This is the
                                  strongest realistic adaptive threat.

If attack 3 collapses leakage back toward the no-defense baseline (~41%),
DANN's privacy claim was illusory under adaptive threat. If leakage
stays in the ~22% range across all three attacks, the defense is robust.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import top_k_accuracy_score

from attacks.closed_set import closed_set_reid
from config import RESULTS_DIR
from data.physionet_loader import valid_subjects
from defenses.dann import DANNVictim, _DANNNet
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
    for epoch in range(n_epochs):
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
def _deep_mlp_predict(model, Z_test, sid_to_idx, *, device):
    inv = {v: k for k, v in sid_to_idx.items()}
    model.eval()
    Z = torch.from_numpy(Z_test).float().to(device)
    logits = model(Z)
    pred_idx = logits.argmax(dim=1).cpu().numpy()
    return np.array([inv[int(i)] for i in pred_idx])


# ---------------------------------------------------------------------------
# Attack 3: encoder fine-tune
# ---------------------------------------------------------------------------
class _FineTunedReID(nn.Module):
    """DANN backbone (initialized from the trained encoder) + linear re-ID head.
    All weights trainable end-to-end."""
    def __init__(self, source_net: _DANNNet, n_subjects: int):
        super().__init__()
        self.input_scale = source_net.input_scale
        # Copy the backbone — this gets fine-tuned end-to-end
        self.backbone = source_net.backbone
        self.feat_dim = source_net.feat_dim
        self.head = nn.Linear(self.feat_dim, n_subjects)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.input_scale
        h = self.backbone(x).flatten(1)
        return self.head(h)


def _fine_tune_encoder(source_victim: DANNVictim, X_train, y_train,
                       *, device, n_epochs=20, lr=5e-4, batch_size=64, seed=0):
    torch.manual_seed(seed)
    sid_to_idx = {int(s): i for i, s in enumerate(sorted(set(int(x) for x in y_train)))}
    y_idx = np.array([sid_to_idx[int(s)] for s in y_train], dtype=np.int64)
    n_classes = len(sid_to_idx)

    src_net = source_victim.model_
    model = _FineTunedReID(src_net, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    n = len(X_train)
    Xt = torch.from_numpy(X_train.astype(np.float32, copy=False))
    yt = torch.from_numpy(y_idx)
    model.train()
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


@torch.no_grad()
def _fine_tune_predict(model, X_test, sid_to_idx, *, device, batch_size=256):
    inv = {v: k for k, v in sid_to_idx.items()}
    model.eval()
    out_idx = []
    for i in range(0, len(X_test), batch_size):
        xb = torch.from_numpy(X_test[i:i + batch_size].astype(np.float32, copy=False)).to(device)
        out_idx.append(model(xb).argmax(dim=1).cpu().numpy())
    pred_idx = np.concatenate(out_idx)
    return np.array([inv[int(i)] for i in pred_idx])


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


def _bootstrap_correctness(preds, truth, trial_ids, *, seed=0, n_resamples=1000):
    correct = (preds == truth).astype(np.float64)
    return grouped_bootstrap_ci(correct, groups=trial_ids, statistic=np.mean,
                                n_resamples=n_resamples, seed=seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--lambda-", dest="lambda_", type=float, default=0.2,
                   help="Adversary weight for the DANN victim being attacked.")
    p.add_argument("--dann-epochs", type=int, default=50)
    p.add_argument("--mlp-epochs", type=int, default=60)
    p.add_argument("--ft-epochs", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.smoke:
        subjects = valid_subjects()[:10]
        args.dann_epochs = 15
        args.mlp_epochs = 20
        args.ft_epochs = 5
    elif args.all:
        subjects = valid_subjects()
    else:
        p.error("Provide --smoke or --all")

    np.random.seed(args.seed)
    print(f"Subjects: {len(subjects)} (chance top-1 = {100/len(subjects):.2f}%)")
    print(f"DANN λ = {args.lambda_}  ({args.dann_epochs} epochs)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n", flush=True)

    print("Loading windowed data ...", flush=True)
    full = windowed_subjects(subjects, runs="imagery")
    train = full.filter_runs(list(VICTIM_TRAIN_RUNS))
    test = full.filter_runs(list(VICTIM_TEST_RUNS))
    print(f"  train={train.n_windows}  test={test.n_windows}  "
          f"chans={train.n_channels}\n", flush=True)

    # ---- Step 1: train the DANN victim being attacked ----
    print(f"=== training DANN victim (λ={args.lambda_}) ===", flush=True)
    t0 = time.time()
    victim = DANNVictim(
        n_channels=train.n_channels, n_times=train.n_times,
        n_classes_task=4, n_epochs=args.dann_epochs,
        lambda_=float(args.lambda_), seed=args.seed, verbose=False,
    )
    victim.fit(train.X, train.y, subject_ids=train.subject_ids)
    task_acc = victim.score(test.X, test.y)
    print(f"  trained in {time.time() - t0:.1f}s  task_acc(test)={task_acc:.3f}",
          flush=True)

    # Embeddings (frozen encoder) for attacks 1 and 2
    Z_train = victim.embed(train.X)
    Z_test = victim.embed(test.X)
    n_subj = len(np.unique(train.subject_ids))

    results: list[AdaptiveResult] = []

    # ---- Attack 1: standard logreg (= D2 baseline) ----
    print(f"\n=== attack 1: logreg probe (frozen encoder) ===", flush=True)
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

    # ---- Attack 2: deep MLP probe on frozen encoder ----
    print(f"\n=== attack 2: deep MLP probe (frozen encoder) ===", flush=True)
    t0 = time.time()
    mlp, sid_to_idx = _train_deep_mlp(
        Z_train, train.subject_ids, n_classes=n_subj,
        device=device, n_epochs=args.mlp_epochs, seed=args.seed,
    )
    preds = _deep_mlp_predict(mlp, Z_test, sid_to_idx, device=device)
    correct = (preds == test.subject_ids).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=test.trial_ids, statistic=np.mean,
                              n_resamples=1000, seed=args.seed)
    # top-k via softmax
    mlp.eval()
    with torch.no_grad():
        all_logits = mlp(torch.from_numpy(Z_test).float().to(device))
        proba = F.softmax(all_logits, dim=1).cpu().numpy()
    inv = {v: k for k, v in sid_to_idx.items()}
    sorted_classes = np.array([inv[i] for i in range(n_subj)])
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

    # ---- Attack 3: fine-tune the encoder for re-ID ----
    print(f"\n=== attack 3: encoder fine-tune (end-to-end) ===", flush=True)
    t0 = time.time()
    ft_model, sid_to_idx = _fine_tune_encoder(
        victim, train.X, train.subject_ids,
        device=device, n_epochs=args.ft_epochs, seed=args.seed,
    )
    preds = _fine_tune_predict(ft_model, test.X, sid_to_idx, device=device)
    correct = (preds == test.subject_ids).astype(np.float64)
    ci = grouped_bootstrap_ci(correct, groups=test.trial_ids, statistic=np.mean,
                              n_resamples=1000, seed=args.seed)
    # top-k via softmax over batches (memory friendly)
    ft_model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(test.X), 256):
            xb = torch.from_numpy(test.X[i:i + 256].astype(np.float32, copy=False)).to(device)
            chunks.append(F.softmax(ft_model(xb), dim=1).cpu().numpy())
    proba = np.concatenate(chunks, axis=0)
    inv = {v: k for k, v in sid_to_idx.items()}
    sorted_classes = np.array([inv[i] for i in range(n_subj)])
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
        "victim": "DANN λ=" + str(args.lambda_),
        "task_acc": float(task_acc),
        "attacks": [r.__dict__ for r in results],
    }
    out_path = RESULTS_DIR / "15_d2_adaptive_attacker.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults written to {out_path}\n")

    print("| Attack | Top-1 (95% CI) | Top-5 | Top-10 |")
    print("|---|---|---|---|")
    for r in results:
        ci = f"{r.top1:.3f} [{r.top1_ci_low:.3f}, {r.top1_ci_high:.3f}]"
        print(f"| {r.attack} | {ci} | {r.top5:.3f} | {r.top10:.3f} |")


if __name__ == "__main__":
    main()
