# Repository inspection

Inspected repository: `/Users/arvind/Documents/research/experiments/uncertainty-estimation-gpm/`.

- Primary cache `sampling_experiment_results/computers_sampling_full.npz`: complete (6,50,2751,10) probabilities, labels, node IDs, budgets, checkpoint path, repetition count and final completed-budget marker.
- Cache budgets: 4,8,16,32,64,128. No K=256 array found. `sampling_smoke.npz` is a separate truncated smoke experiment, deliberately excluded.
- Existing producer: `GPM/GPM/sampling_experiment.py`; existing execution wrapper: `GPM/modal_gpm.py`; previous analysis: `experiments.ipynb`.
- Checkpoint identified in cache and execution wrapper: `/workspace/GPM/model/computers/epoch_380.pt`, stored in the original remote Modal model volume. No local `.pt`/`.pth` checkpoint or processed dataset found; `GPM/data/` contains only its README. The remote volume was not accessed or changed.
- Source loads test IDs once from the first split, assigns a shared node axis across K and repetitions, freezes model parameters and uses eval mode. Fresh random walks use seed 10000+1000*K+r. Hence there is no cross-K pairing justified by common random numbers.
- Tuned Computers configuration overrides defaults: high split, p=q=1, Laplacian PE dimension 8, GRU PE encoder, PE weight .5, CLS token enabled, attention fusion disabled. Original sampling loader uses this tuned configuration. No tokenizer or sampling-distribution change was made.
- Existing working tree already had changes to `GPM/main.py` and untracked sampling/Modal scripts. These were preserved. No baseline bug requiring a change was identified during this inspection.
- Mask membership, original graph labels, frozen state hash, exact training run seed, dependency versions of original inference and exact positional-encoding realization cannot be independently verified from the saved cache. Source-based identity is explicitly qualified in config/report.
- No inference is needed for these diagnostics. Exact regeneration would require restoring the original frozen checkpoint and dataset/environment. No training, downloads of research data, remote GPU jobs or Adaptive GPM implementation were performed.
