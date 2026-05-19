"""Tier-1 multi-seed replication for the headline new results.

The milestone-time A4 PhysioNet result had a 5-seed extension
(experiment 14, AUC = 0.934 +/- 0.020). The Tier 1+2 headline numbers
were all reported at seed 0. This script re-runs the load-bearing ones
across 5 seeds and reports mean +/- std across seeds in addition to
the per-seed bootstrap CIs.

Covered:
    A3 Lee 2019 cross-session re-ID (experiment 20)
    A4 Lee 2019 open-set verification, within-session (experiment 24)
    A4 cross-dataset symmetric, all four directions (experiment 26)

Each call is just experiment X with a different --seed; the outputs
are aggregated here per-run and written as a single multi-seed JSON.

Usage
-----
    python -m experiments.34_tier1_multi_seed --quick   # 3 seeds, all targets
    python -m experiments.34_tier1_multi_seed --full    # 5 seeds, all targets
    python -m experiments.34_tier1_multi_seed --quick --target a4_lee2019
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from config import RESULTS_DIR

TARGETS: dict[str, dict] = {
    "a3_lee2019": {
        "script": "experiments.20_a3_lee2019",
        "argv": ["--all"],
        "result_json": "20_a3_lee2019.json",
        "metric_paths": [
            ("eegnet_logreg_top1", lambda data: next(
                r["top1"] for r in data
                if r["victim"] == "eegnet" and r["probe"] == "logreg")),
            ("fbcsp_logreg_top1", lambda data: next(
                r["top1"] for r in data
                if r["victim"] == "fbcsp_lda" and r["probe"] == "logreg")),
            ("riemann_logreg_top1", lambda data: next(
                r["top1"] for r in data
                if r["victim"] == "riemann_ts_lr" and r["probe"] == "logreg")),
        ],
    },
    "a4_lee2019": {
        "script": "experiments.24_a4_lee2019",
        "argv": ["--all"],
        "result_json": "24_a4_lee2019_within_session.json",
        "metric_paths": [
            ("auc_within_session", lambda data: data["auc"]),
            ("eer_within_session", lambda data: data["eer"]),
        ],
    },
    "xds_iv2a_to_physionet": {
        "script": "experiments.26_a4_cross_dataset_symmetric",
        "argv": ["--direction", "iv2a_to_physionet", "--all"],
        "result_json": "26_a4_xds_iv2a_to_physionet.json",
        "metric_paths": [("auc", lambda data: data["auc"])],
    },
    "xds_physionet_to_lee2019": {
        "script": "experiments.26_a4_cross_dataset_symmetric",
        "argv": ["--direction", "physionet_to_lee2019", "--all"],
        "result_json": "26_a4_xds_physionet_to_lee2019.json",
        "metric_paths": [("auc", lambda data: data["auc"])],
    },
    "xds_lee2019_to_physionet": {
        "script": "experiments.26_a4_cross_dataset_symmetric",
        "argv": ["--direction", "lee2019_to_physionet", "--all"],
        "result_json": "26_a4_xds_lee2019_to_physionet.json",
        "metric_paths": [("auc", lambda data: data["auc"])],
    },
    "xds_iv2a_to_lee2019": {
        "script": "experiments.26_a4_cross_dataset_symmetric",
        "argv": ["--direction", "iv2a_to_lee2019", "--all"],
        "result_json": "26_a4_xds_iv2a_to_lee2019.json",
        "metric_paths": [("auc", lambda data: data["auc"])],
    },
}


def _run_one(target: str, seed: int) -> dict:
    cfg = TARGETS[target]
    cmd = ["python", "-u", "-m", cfg["script"], *cfg["argv"], "--seed", str(seed)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if proc.returncode != 0:
        return {"seed": seed, "ok": False, "wall_seconds": dt,
                "stderr_tail": proc.stderr[-2000:]}
    result_path = Path(RESULTS_DIR) / cfg["result_json"]
    data = json.loads(result_path.read_text())
    row: dict = {"seed": seed, "ok": True, "wall_seconds": dt}
    for name, getter in cfg["metric_paths"]:
        try:
            row[name] = float(getter(data))
        except Exception as exc:
            row[name] = None
            row[f"{name}_error"] = str(exc)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="3 seeds")
    p.add_argument("--full",  action="store_true", help="5 seeds")
    p.add_argument("--target", nargs="*", default=None,
                   help="Subset of target keys; default = all")
    args = p.parse_args()

    if args.quick:
        seeds = [0, 1, 2]
    elif args.full:
        seeds = [0, 1, 2, 3, 4]
    else:
        p.error("Provide --quick or --full")

    targets = list(args.target) if args.target else list(TARGETS.keys())
    print(f"Multi-seed sweep over {len(targets)} target(s) x {len(seeds)} seed(s)")
    for t in targets:
        if t not in TARGETS:
            print(f"  !! unknown target: {t} -- skipping", flush=True)

    summary: dict = {"seeds": seeds, "targets": targets, "rows": {}}
    for t in targets:
        if t not in TARGETS:
            continue
        rows = []
        for s in seeds:
            print(f"\n=== {t} | seed = {s} ===", flush=True)
            row = _run_one(t, s)
            print(json.dumps(row, indent=2)[:600], flush=True)
            rows.append(row)
        # Compute mean +/- std on the numeric metrics across seeds.
        agg: dict = {}
        for name, _ in TARGETS[t]["metric_paths"]:
            vals = [r[name] for r in rows if r.get("ok") and isinstance(r.get(name), (int, float))]
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
                agg[name] = {"mean": mean, "std": var ** 0.5, "n": len(vals),
                             "values": vals}
        summary["rows"][t] = {"rows": rows, "aggregated": agg}

    out_path = Path(RESULTS_DIR) / "34_tier1_multi_seed.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
