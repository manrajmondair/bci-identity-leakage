# `runs/` — execution provenance

One subdirectory per experiment run, named with a sortable timestamp + experiment tag + git short SHA. Each contains a small `meta.json` capturing what was run, on what hardware, against which commit, and when. Treated as the audit trail for the experimental claims in the report.

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

Notebooks under `colab/` write this file at the end of every run; the user sends it back along with the canonical result JSON. Both get committed to the repo so any number in the milestone or final report can be traced to a specific commit + hardware + timestamp.

Mac-local runs (e.g. classical baseline reruns) follow the same pattern; the helper that produces `meta.json` lives in `eval/run_meta.py` (TODO: add a thin wrapper for Mac runs so the provenance trail is symmetric).
