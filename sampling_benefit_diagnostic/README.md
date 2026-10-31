# Sampling uncertainty and benefit diagnostic

This separate module reuses the original Computers cache. It does not modify the GPM, run training, or implement adaptive sampling. Read `results/REPORT.md` for measured findings and provenance limitations.

From this folder, with Python 3.10+ and numpy, pandas, scipy, scikit-learn, matplotlib installed:

```sh
python -m unittest -v test_experiment.py
python experiment.py --repository /Users/arvind/Documents/research/experiments/uncertainty-estimation-gpm --cache /Users/arvind/Documents/research/experiments/uncertainty-estimation-gpm/sampling_experiment_results/computers_sampling_full.npz --output results --seed 20261029 --bootstrap 2000 --random-allocations 2000
```

Dependencies are analysis-only; use the existing scientific Python environment. No GPU is needed. `requirements.txt` lists direct dependencies. Outputs overwrite only files in the explicitly selected output directory; choose another directory to preserve earlier diagnostic runs.

The original cache already has all required endpoint probabilities. No additional inference generator is necessary. If a replacement cache is absent, incomplete, nonfinite or malformed, the script fails explicitly rather than synthesizing data. To regenerate inference, the original `GPM/GPM/sampling_experiment.py` is the existing implementation, but the exact checkpoint, dataset and GPM environment must first be restored; none are downloaded or trained here.

A = first floor(R/2) repetitions; B = remainder; fewer than 4 repetitions fail. A/B IDs and actual count are recorded. Source-asserted split/checkpoint identity is distinguished from independently verified metadata in the report/config. The script accepts the original probability schema only; it deliberately refuses silently interpreting arbitrary logits as probabilities.

No policies are tuned on labels. The fixed top-25% rule and high-error threshold e16>=0.5 are specified in the module before analysis. Oracle uses labels only as an explicit non-deployable bound. Bootstrap draws resample nodes, preserving each node's endpoint contrasts; scores, bins and selected memberships are fixed. Undefined constant-vector correlations and single-class detection metrics are JSON null. Empty quintiles are represented explicitly. Ties use node ID. Cross-seed aggregation is not performed because only one checkpoint cache exists.
