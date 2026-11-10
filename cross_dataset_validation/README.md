# Cross-dataset validation

This folder tests whether adaptive pattern sampling for GPM holds beyond a single model and dataset. Adaptive sampling starts each node with a few random-walk patterns and adds more only while the model is unsure.

The earlier results came from one trained checkpoint on Amazon Computers. This folder repeats the evaluation on:

- **Two datasets:** Amazon Computers and WikiCS.
- **Three checkpoints per dataset,** each trained with a different seed.
- **Ten sampling repetitions per checkpoint,** each drawing new random walks.

It reuses the repository's adaptive inference (`GPM/final_inference.py`) and accuracy-efficiency evaluation (`accuracy_efficiency_eval/`) without changing them. This folder adds checkpoint training, a checkpoint registry, per-dataset diagnostics, timing and result tables.

## Questions

1. **Accuracy:** Does adaptive inference stay within 0.5 points of fixed K = 128 on every checkpoint, at lower cost?
2. **Allocation:** Does choosing which nodes get more patterns beat spreading the same budget evenly, or at random?
3. **Room to gain:** On a new dataset, do more patterns help at all, and does confidence point to the nodes they help?
4. **Uncertainty scores:** Is maximum confidence still a good cheap stopping signal, compared with entropy and two-run disagreement?
5. **Time:** Do the pattern savings turn into real time savings?

## Definitions

| Term | Meaning |
|---|---|
| **Pattern** | One random walk of length 8 from a node. GPM classifies a node by attending over K of its patterns. |
| **Checkpoint** | A trained, frozen set of GPM weights. Each one is specific to its dataset, because the input and output sizes depend on the features and classes. |
| **Repetition** | One inference pass over all nodes with new random walks. GPM is random at inference, so results are averaged over repetitions. |
| **Nested sampling** | Each node draws 128 patterns once. K = 16 uses the first 16, K = 32 the first 32, and so on. |
| **Nested four-stage rule** | Predict with 16 patterns. If top-class confidence is below the threshold, try 32, then 64, then 128. |
| **Pattern-passes** | Cost: the patterns sent through GPM per node, counting every pass. Earlier stages are recomputed, so stopping at 16, 32, 64 or 128 costs 16, 48, 112 or 240. Fixed K costs K. |

## Design

| Choice | Setting |
|---|---|
| Split | Split 0 for every checkpoint. Computers uses GPM's random split (fixed by seed 42). WikiCS uses its public split. |
| Checkpoints | Training seeds 0, 1 and 2. Checkpoint ID is seed + 1. |
| Training | Same settings as `GPM/main.py --use_params`. Training stops by early stopping on validation accuracy. |
| Epoch kept | The epoch with the best validation accuracy. Test accuracy is recorded but never used to choose. |
| Stopping threshold | Chosen from validation nodes only: the strictest threshold whose mean cost is at most 48 pattern-passes. It is frozen before test nodes are used. |
| Results | Test nodes only, 10 repetitions. Gaps have 95% intervals from a paired node bootstrap. |
| Sampling seeds | Predictions 71000 + r, live run 171000 + r, timing 271000. The same for every checkpoint. |
| Uncertainty comparison | Validation nodes. Scores come from repetitions 0–1 and measured benefit from repetitions 2–9, so the two never share random walks. |
| Variation | Reported two ways: between checkpoints (training) and between repetitions (sampling). |

## Folder structure

```
cross_dataset_validation/
├── README.md
├── checkpoints.csv              registry of every checkpoint used
│
├── common.py                    shared paths, registry helpers, environment info
├── run_inference.py             runs GPM/final_inference.py from this folder
├── download_data.py             downloads a dataset and prints its split
├── train_checkpoint.py          trains one seed on split 0 and registers it
├── register_checkpoint.py       registers a checkpoint trained elsewhere
├── run_checkpoint.py            the full pipeline for one checkpoint
├── diagnostic.py                K = 16 against K = 128 on validation nodes
├── uncertainty_comparison.py    cheap uncertainty scores across checkpoints
├── time_methods.py              matched timing of fixed, repeated and adaptive
├── summarize_results.py         result tables per checkpoint and per dataset
├── export_handoff.py            packages selected runs for further analysis
├── test_cross_dataset.py        unit tests
│
├── runs/<dataset>/ckpt<id>/     outputs for one checkpoint
├── results/                     tables across checkpoints
├── runs_smoke/                  small pipeline checks, never used in results
└── handoff/                     packaged runs, from export_handoff.py
```

### Scripts

| Script | What it does | GPU |
|---|---|---|
| `download_data.py` | Loads a dataset through GPM's loader, which downloads it on first use. Prints node, edge, feature and class counts and the split sizes. | No |
| `train_checkpoint.py` | Trains GPM on split 0 with a chosen seed. Follows `GPM/main.py` step by step, but on a fixed split and keeping the best-validation weights. Saves the checkpoint and a training log, and registers it. | Yes |
| `register_checkpoint.py` | Adds a checkpoint trained outside this folder to the registry, with its hash. | No |
| `run_checkpoint.py` | Checks the checkpoint's hash, then runs four steps: nested predictions, live adaptive run, evaluation and plot. Finished steps are skipped on rerun. A smoke mode uses 32 nodes and 2 repetitions. | Yes |
| `diagnostic.py` | Compares K = 16 with K = 128 on validation nodes: accuracy change, errors corrected and correct answers broken, nodes that gain or lose, nodes wrong in every repetition, and how well confidence points to nodes that gain. | No |
| `uncertainty_comparison.py` | Compares confidence, entropy and two-run disagreement as predictors of benefit from more patterns, using Spearman correlation, AUROC and the top-20% benefit. | No |
| `time_methods.py` | Times fixed K = 16, 32, 64 and 128, repeated sampling at K = 16, and the adaptive rule in one session. The order is shuffled over 3 rounds, and speedup is measured within each round. Also records peak GPU memory. | Yes |
| `summarize_results.py` | Collects every finished run into the two result tables. Rejects smoke runs. | No |
| `export_handoff.py` | Checks each run's predictions and decisions (full run, nested, correct checkpoint, matching test nodes) and copies them into labelled folders with a manifest. | No |
| `test_cross_dataset.py` | Tests the parts that run without a GPU: scores, diagnostic, registry, pipeline commands, summary, export and training settings. | No |

Every script lists its options with `--help`. Scripts refuse to overwrite outputs; delete an output file to redo a step.

### Order of use

1. Download each dataset.
2. Train three checkpoints per dataset.
3. Run the pipeline for each checkpoint, as a smoke test first.
4. Run the diagnostic and the uncertainty comparison. These use validation nodes only.
5. Time the methods.
6. Build the result tables and, if needed, export runs.

### `checkpoints.csv`

One row per checkpoint. Every later step checks a checkpoint's hash against this file, so a run can't silently use different weights.

| Column | Meaning |
|---|---|
| `dataset`, `checkpoint_id`, `seed`, `split_index` | Which checkpoint, and how it was trained |
| `path`, `sha256` | Where the weights are and their hash |
| `best_epoch`, `val_acc`, `test_acc` | Epoch kept and its accuracies |
| `source`, `notes` | The script that produced it, or where it came from |
| `torch`, `gpu`, `git_commit` | Environment it was trained in |

### `runs/<dataset>/ckpt<id>/`

| File | Contents |
|---|---|
| `final_predictions.npz` | Class probabilities `[K, repetition, node, class]` at K = 16, 32, 64 and 128, for validation and test nodes, with labels, node IDs, split and metadata |
| `adaptive_decisions.npz` | Live adaptive run on test nodes: final K, prediction, pattern-passes and sampled patterns per node and repetition, plus the frozen threshold and run time |
| `eval/` | Output of `accuracy_efficiency_eval/evaluate.py`: `summary.json`, `comparisons.csv`, `curve.csv` and the `accuracy_vs_cost` plot |
| `diagnostic_val.json` | Output of `diagnostic.py` |
| `timing.json` | Output of `time_methods.py`, per round and summarised |

The `.npz` files are not tracked by git.

### `results/`

| File | Contents |
|---|---|
| `per_checkpoint.csv` | One row per checkpoint: accuracy of fixed K = 128 and adaptive, gap with interval, gains over fixed and random at the same cost, cost, threshold, stop shares, whether the 0.5-point target is met, and timing if measured |
| `per_dataset.csv` | Means over checkpoints, with spread between checkpoints and between repetitions, and how many checkpoints meet the target |
| `uncertainty/per_checkpoint_val.csv`, `uncertainty/summary_val.csv` | Uncertainty score comparison per checkpoint and per dataset |
| `combined/` | All runs merged in the format of `accuracy_efficiency_eval/aggregate_runs.py` |

### `handoff/`

One folder per exported run, named `<dataset>_ckpt<id>`, holding `final_predictions.npz` and `adaptive_decisions.npz`. Any folder can be evaluated directly with `accuracy_efficiency_eval/evaluate.py`. `MANIFEST.json` records each folder's checkpoint, seed, node counts, threshold, cost and file hashes. The data files and `handoff.zip` are not tracked by git.

## Results

Test nodes, mean of three checkpoints per dataset. Gaps and gains are in percentage points. Full tables are in `results/`.

| | Computers | WikiCS |
|---|---|---|
| Fixed K = 128 accuracy | 91.15% | 80.61% |
| Adaptive accuracy | 91.05% | 80.34% |
| Gap vs fixed K = 128 | −0.11 | −0.27 |
| Gain vs fixed K at the same cost | +0.28 | +0.13 |
| Gain vs random allocation at the same cost | +0.79 | +0.78 |
| Pattern-passes per node | 46.0 (36% of K = 128) | 49.5 (39% of K = 128) |
| Checkpoints within 0.5 points of K = 128 | 3 of 3 | 3 of 3 |

- **Success target:** Every checkpoint on both datasets stays within 0.5 points of fixed K = 128, with the whole interval inside the target, at under 40% of its pattern-passes.
- **Computers:** Adaptive is close to level with fixed K = 128, and clearly beats spreading the same budget evenly on all three checkpoints.
- **WikiCS:** Adaptive is slightly but measurably below fixed K = 128. Its gain over an even budget is smaller and is clear on one checkpoint of three.
- **Confidence as a signal:** Adaptive beats random allocation by about 0.8 points on every checkpoint, on both datasets.
- **Time:** On checkpoint 1 of each dataset, adaptive ran 2.2–2.4 times faster than fixed K = 128 on an RTX 5060 Laptop GPU.

## Requirements and setup

- **Environment:** The same Python environment as `GPM/`, with PyTorch and PyTorch Geometric. A CUDA GPU is needed for training, inference and timing. The results above used PyTorch 2.7.1 with CUDA 12.8.
- **Data:** Datasets download automatically into `GPM/data/` on first use. The walk cache is saved in `GPM/patterns/`. Neither is tracked by git.
- **Checkpoints:** `train_checkpoint.py` saves weights to `GPM/model/<dataset>/seed<seed>/best.pt`, which is not tracked by git. The registry records each file's hash.
- **Windows:** The scripts run on Windows. They never import `GPM/main.py`, which needs the Unix-only `resource` module. If `dgl` is missing, `common.py` installs a stub, because GPM imports it but never calls it.
- **Paths:** Scripts are run from the repository root.

## Notes

- **Test nodes:** Do not inspect test results until every threshold is frozen. The diagnostic and the uncertainty comparison use validation nodes by default.
- **Environment:** Accuracies depend on the environment. The same checkpoint has given K = 128 accuracies about 1 point apart on different GPUs and library versions. Compare methods within one environment, and do not compare these numbers with results produced elsewhere.
- **Live and offline results:** Evaluation reports both the live adaptive run and an offline replay on saved predictions. They use different random walks, so they differ slightly.
