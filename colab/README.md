# Colab notebooks

Heavy compute (anything that trains EEGNet, DANN variants, DP-SGD, or shadow models for membership inference) runs here. Mac is for development, classical baselines, and attack code.

## How the workflow works

1. **Open a notebook in Colab** — File → Upload Notebook, or just open the `.ipynb` directly from the GitHub repo URL.
2. **Set runtime to L4 GPU** — Runtime → Change runtime type → L4. Default; only switch to A100 if a notebook explicitly says so.
3. **Run all cells.** Each notebook is self-contained: it clones `main`, installs deps, prefetches data, runs the experiment, prints results.
4. **Paste the result block back to me.** The last cell prints the JSON contents inline. Copy that whole block into the chat.
5. **I commit the result JSON + figure** back to the repo so the canonical state stays in `main`.

The notebooks intentionally don't push back to the repo themselves — that would require putting a GitHub PAT in Colab. Easier to keep that boundary clean and copy/paste the small JSONs.

## Notebooks

| Notebook | What it runs | Wall time | Output |
|---|---|---|---|
| `A1_eegnet_rerun.ipynb` | `experiments/02_closed_set_reid --all` (rebuilds A1 with the volts → microvolts EEGNet fix) | ~25 min | `results/02_closed_set_reid.json`, `figures/02_closed_set_reid.pdf` |

More to come (A2 cross-task, A4 contrastive, A5 shadow models, D2 DANN, D3 DP-SGD).

## Hardware notes

- **L4 default.** EEGNet is ~5K params; data transfer dominates over compute. L4 ≈ A100 for our workloads, costs fewer Pro compute units, and starts faster from cold.
- **A100 only if a notebook explicitly requests it.** Reserved for if/when we add a transformer-scale EEG model (we don't have one in scope right now).
- **Disk:** PhysioNet imagery cache is 1.7 GB, windowed-array cache is 2.3 GB; well under Colab's free disk.
