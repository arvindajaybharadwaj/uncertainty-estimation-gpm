# Encoder caching

Nested four-stage adaptive inference (K = 16 → 32 → 64 → 128) that encodes only the new walks at each stage, instead of re-encoding all of them.

Two things change from `cross_dataset_validation`, and nothing else:

1. **Layout.** A walk keeps its multiscale length for good. The walks added at each stage are split into one equal chunk per length, and the walks are placed length group by length group, so GPM's own position mask assigns the lengths. K = 16 is the same as the original layout. See `layout.py`.
2. **Caching.** Each stage encodes only its new walks with GPM's `PatternEncoder`. Their vectors join each length group's cache, and the pattern Transformer, CLS token and head rerun on all of them. See `cached_gpm.py`.

Unchanged from the original runs:

- **Checkpoints and nodes:** the same checkpoints, with their SHA-256 hashes, and the same validation and test nodes.
- **Seeds:** 71000 + r for the prediction stage, 171000 + r for live inference and 271000 for timing.
- **Repetitions and batch size:** 10 repetitions. Batch size 256, or 32 for CoraFull, as in the original runs; 32 for timing.
- **Stopping rule:** confidence (top-class probability) against a threshold from `choose_threshold` in `GPM/final_inference.py`, on validation nodes, at a target of 48 pattern-passes.
- **Cost:** pattern-passes, counted every time a walk goes through the model (16 / 48 / 112 / 240). Caching shows up in measured time, not in this cost.

GPM's files and everything in `cross_dataset_validation/` are only read. Every output in this folder refuses to overwrite an existing file.

## Built-in checks

- **Stage 1:** K = 16 is identical in both layouts, so `predictions.py` stops without saving if it differs from the original `final_predictions.npz` by more than 1e-4.
- **Caching:** `live_cached.py --check` runs a fresh forward pass on the same walks at every stage, and stops if the cached probabilities differ by more than 1e-4. Smoke runs always use it.

On the WikiCS checkpoint 1 smoke run, stage 1 matched exactly (difference 0.0), and cached matched fresh to within 2.1e-7.

## Files

| File | Purpose |
|---|---|
| `layout.py` | The per-length layout: which walks each budget uses, grouped by length |
| `cached_gpm.py` | Model setup, the original forward pass, `encode` (new walks only), `classify`, and the cached adaptive rule |
| `predictions.py` | Prediction stage in the new layout: K = 16, 32, 64, 128 and fixed K = 48, for validation and test nodes |
| `live_cached.py` | Threshold from the new-layout validation predictions, then cached adaptive inference on test nodes |
| `evaluate_cached.py` | Paired comparisons for one checkpoint; `--summarize` builds the tables |
| `run_checkpoint.py` | Predictions → live → `evaluate.py` → comparisons, for one checkpoint |
| `timing_cached.py` | Matched timing of the original methods, fixed K = 48 and cached adaptive |
| `plot_results.py` | The three figures in `results/`, drawn from the outputs above |
| `plot_sampling_benefit.py` | Fig. 1 of the ESANN paper (K = 16 vs 128 on validation nodes) from the new-layout predictions, with the original values printed beside the new |
| `paths.py` | Shared paths and settings |
| `test_encoder_caching.py` | Layout and pipeline tests (no GPU needed) |

Outputs go to `runs/<dataset>/ckpt<id>/` (`predictions.npz`, `decisions.npz`, `eval/`, `comparison.json`, `timing.json`) and `results/` (`summary.csv`, `summary_by_dataset.csv`). Smoke runs go to `runs_smoke/`.

## Comparisons

On test nodes, with a paired node bootstrap: 2,000 draws, seed 20261105 + 9.

| Name | Comparison | Question |
|---|---|---|
| `cached_vs_new_fixed_{32,48,128}` | Cached adaptive − new-layout fixed K | Main result |
| `cached_vs_original_adaptive` | Cached adaptive − original adaptive | Does the new method change accuracy? |
| `layout_fixed_{32,48,64,128}` | New-layout fixed K − original fixed K | Does the layout alone change accuracy? |
| `replay_vs_cached` | Offline replay at the cached threshold − cached live | Does replay still describe the live method? |

The original files are `final_predictions.npz`, `fixed_budgets.npz` (fixed K = 48) and `adaptive_decisions.npz` in `cross_dataset_validation/runs`. Walks differ between the runs being compared, so the pairs are by node, not by walk.

## Running

Use the `GPM` conda environment. The default Python lacks `torch_sparse`. Run from the repository root.

1. Smoke test, about a minute:

   ```
   python encoder_caching/run_checkpoint.py --dataset wikics --checkpoint-id 1 --smoke
   ```

2. All nine checkpoints:

   ```
   for %d in (computers wikics cora_full) do @for %c in (1 2 3) do @python encoder_caching\run_checkpoint.py --dataset %d --checkpoint-id %c
   ```

3. Summary tables:

   ```
   python encoder_caching/evaluate_cached.py --summarize
   ```

4. Timing, checkpoint 1 of each dataset, after step 2:

   ```
   for %d in (computers wikics cora_full) do @python encoder_caching\timing_cached.py --dataset %d --checkpoint-id 1
   ```

`run_checkpoint.py` skips any step whose output already exists. To redo a step, move its output elsewhere first.
