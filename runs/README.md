# `runs/` — execution provenance

One subdirectory per experiment run, named with a sortable timestamp + experiment tag + git short SHA. Each contains a small `meta.json` capturing what was run, on what hardware, against which commit, and when. Treated as the audit trail for every claim made from `results/`.

```
runs/
├── 20260507T0140Z_a1_cross_3733900/
│   └── meta.json
├── 20260520T0930Z_a4_openset_<sha>/
│   └── meta.json
└── ...
```

`meta.json` schema (one per run):

```json
{
  "run_id": "20260507T0140Z_a1_cross_3733900",
  "experiment": "experiments.02_closed_set_reid",
  "args": ["--all"],
  "git_sha": "3733900...",
  "hardware": "Colab NVIDIA L4",
  "platform": "Linux-...-x86_64-with-glibc...",
  "torch_version": "2.5.1+cu121",
  "completed_at_utc": "2026-05-07T01:40:00Z",
  "outputs": ["results/02_closed_set_reid.json", "figures/02_closed_set_reid.pdf"]
}
```

Notebooks under `colab/` write this file at the end of every run; the result JSON and meta JSON are committed to canonical paths so every reported number traces back to a specific commit + hardware + timestamp.

`python -m tools.audit` writes its report to `runs/<timestamp>_audit_<sha>/audit.{md,json}` when invoked, so an audit can be archived next to the run provenance it checks. The audit enforces the result-file invariants — confidence-interval brackets, unit-interval ranges, train/test split disjointness, effect-size sanity against the literature, and a shuffled-label negative control — over the canonical `results/`. The latest run on the refreshed results returns **273 OK / 0 WARN / 0 FAIL**.

## 2026-05 refresh

After a code-hardening pass (A4 verification pairing, DANN λ application, federated-ε accounting, and embed-hook robustness — see the git history), every canonical result was re-generated. The refresh-run provenance is the `2026-05-30` / `2026-05-31` entries; the original development runs (`2026-05-07`) are retained for history. Most numbers reproduced exactly — the A4 PhysioNet open-set headline (AUC 0.925) and the closed-set re-ID baselines (100% Riemann / 89% FBCSP / 41% EEGNet) were unchanged.
