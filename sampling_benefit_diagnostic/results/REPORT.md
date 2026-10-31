# Does sampling uncertainty predict benefit from more samples?

## A. Research question and motivation
Predictive association between label-free sampling variability at K=16 and node-level error reduction from K=16 to K=128. This is exploratory and conditional on one frozen checkpoint; no Adaptive GPM implementation or causal claim.

## B. Experimental setup and reproducibility
Computers, 2751 test nodes, 10 classes, checkpoint `/workspace/GPM/model/computers/epoch_380.pt`. Training seed 0 is inferred from the first training-run code, not authenticated by checkpoint metadata. Existing probability cache reused without inference or training. K=[4, 8, 16, 32, 64, 128], 50 repetitions each. K=256 unavailable.

A repetitions [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] construct all scores; B repetitions [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] construct all outcomes, including fixed baselines. Repetition IDs across K use different seeds (script default 10000+1000*K+r), so no common-random-number pairing is assumed. Comparisons remain paired by node. Trace of empirical covariance uses ddof=1 and sums class variances. Alternatives: entropy of A-mean probabilities, negative A-mean maximum probability, negative A-mean top-two margin, and variation ratio. None use labels. These measure sampling variability, not Bayesian epistemic uncertainty or posterior variance.

Split inferred from source: seed 42, `high` (60/20/20), first test split. Preprocessing: NormalizeFeatures, RemoveSelfLoops, ToUndirected, AddLaplacianEigenvectorPE(8). Cache has one shared node/label axis for every K and repetition; finite probabilities, normalization, class bounds, unique IDs and A/B disjointness passed. Original dataset/masks and checkpoint files are absent locally; original labels, held-out membership, positional encodings, exact frozen checkpoint identity and model seed cannot be independently authenticated. `config.json` hashes the input and relevant source files. There is no per-run seed manifest in the original cache. Existing scripts/results are preserved.

Seed 20261029; 2000 node bootstrap resamples, percentile 95% intervals. Nodes, not repeated predictions, are statistical units. Intervals condition on this graph/checkpoint and realized A/B predictions, and omit cross-node graph dependence, finite-repetition uncertainty, and training-seed variability. No threshold tuning or test-label policy construction, except the explicitly non-deployable oracle.

## C. Uncertainty versus improvement
Primary Spearman: 0.315360 [95% CI 0.265192, 0.366380].
Secondary Pearson: 0.335292 [95% CI 0.238709, 0.428437].
Mean error K=16: 0.102857 [95% CI 0.093275, 0.112439]; K=128: 0.090934 [95% CI 0.080930, 0.100967].
Mean error reduction: 0.011923 [95% CI 0.009174, 0.014744].
NLL reduction: 0.032130 [95% CI 0.028860, 0.035370]; Brier reduction: 0.016567 [95% CI 0.014484, 0.018668]. Both proper scores are averaged per repetition, not evaluated only on mean probabilities. NLL floor is 1e-15.
Positive/zero/negative estimated error improvement fractions: 0.1327 / 0.8252 / 0.0422.
Top-minus-bottom quintile improvement: 0.055419 [95% CI 0.042982, 0.067784].
Benefit detection AUROC: 0.946821 [95% CI 0.937969, 0.955225]; AP: 0.646058 [95% CI 0.595466, 0.703029]; positive prevalence 0.1327. The target is estimated Delta>0, not a verified latent true benefit.

| Quintile | Nodes | Error16 | Error128 | Improvement (95% CI) |
|---|---:|---:|---:|---|
| 1 | 551 | 0.059093 | 0.058802 | 0.000290 [95% CI 0.000000, 0.000871] |
| 2 | 550 | 0.023636 | 0.023636 | 0.000000 [95% CI 0.000000, 0.000000] |
| 3 | 550 | 0.023636 | 0.023636 | 0.000000 [95% CI 0.000000, 0.000000] |
| 4 | 550 | 0.092073 | 0.088436 | 0.003636 [95% CI 0.002036, 0.005456] |
| 5 | 550 | 0.315927 | 0.260218 | 0.055709 [95% CI 0.042618, 0.068655] |

Quintiles have approximately equal node counts. Ties break by node ID, so equal scores can cross bin boundaries; bins are descriptive, not calibrated thresholds. Negative improvements are retained. `associations.csv` compares every score against all three improvement outcomes and baseline error, with confidence intervals.

## D. Uncertainty versus error
U versus baseline stochastic error, Spearman: 0.558481 [95% CI 0.523217, 0.591306].
Among the bottom uncertainty quintile, consistently incorrect in all 25 K=128 outcome runs: 0.058076 [95% CI 0.039927, 0.078040]. Among the top quintile, estimated error decreases for fraction 0.583636 [95% CI 0.543636, 0.625455].
High error is defined in advance as e(16)>=0.5, using B. Detection AUROC: 0.733512 [95% CI 0.690743, 0.772231]; AP: 0.261845 [95% CI 0.221126, 0.315872]; prevalence 0.0923.
Exploratory residual rank correlation controlling linearly for ranked e(16): 0.152937. This is descriptive: e(16) appears in Delta, and adjustment does not establish causation or independent benefit. Difficulty and additional-sampling benefit must be assessed separately using the correlations, quintiles and random-allocation contrast.

## E. Retrospective budget-allocation diagnostic
Select floor(0.25*N) nodes (687 here), ties by node ID, based only on A scores. Nominal theoretical mean K is 44; integer rounding gives 43.969466. K=32 and K=64 bracket the budget; neither is exactly matched. Random allocation uses the same selected count; its analytic expectation and node-bootstrap CI are reported, plus 2000 realized random allocations and their allocation-distribution interval (not a confidence interval). Oracle uses B Delta and is optimistically evaluated on the same outcomes: an upper bound for realized estimates, not a deployable or unbiased performance estimate.

| Policy | Nominal K | Expected error (95% CI) |
|---|---:|---|
| sampling_variance | 43.9695 | 0.091269 [95% CI 0.081221, 0.101302] |
| entropy | 43.9695 | 0.091908 [95% CI 0.081861, 0.101842] |
| negative_confidence | 43.9695 | 0.091748 [95% CI 0.081745, 0.101767] |
| negative_margin | 43.9695 | 0.091821 [95% CI 0.081861, 0.101840] |
| variation_ratio | 43.9695 | 0.092345 [95% CI 0.082370, 0.102278] |
| oracle_delta_non_deployable | 43.9695 | 0.084551 [95% CI 0.075069, 0.094003] |
| random_25_percent | 43.9695 | 0.099880 [95% CI 0.090177, 0.109601] |
| fixed_16 | 16.0000 | 0.102857 [95% CI 0.093275, 0.112439] |
| fixed_32 | 32.0000 | 0.096081 [95% CI 0.086281, 0.105838] |
| fixed_64 | 64.0000 | 0.092475 [95% CI 0.082790, 0.102465] |
| fixed_128 | 128.0000 | 0.090934 [95% CI 0.080930, 0.100967] |

Variance allocation error reduction versus expected random allocation: 0.008611 [95% CI 0.006546, 0.010706]. Positive is favorable. Paired node bootstrap preserves within-node contrasts with policy memberships fixed. This conditions on learned rankings; it does not include uncertainty from re-estimating scores or selecting nodes on a new dataset.

**Uncertainty estimation costs 25 forward passes at K=16 (400 patterns/node).** These costs are excluded from nominal budgets. These retrospective/oracle-style diagnostics measure predictive value, not practical efficiency. No measured timing is available. No speedup claim is supported.

## F. Limitations and potential confounders
One checkpoint only; no independently trained seed hierarchy available. No original masks/checkpoint/dataset locally; held-out identity is source-supported, not independently verified. Finite 25-run outcomes can flip Delta's sign, and zero improvement is quantized in steps of 1/25. Consistently wrong means all observed B runs, not certain population error. Scores may reflect class difficulty, calibration, node degree, random-walk coverage, or checkpoint-specific failures; graph covariates are unavailable locally. Node bootstrap ignores graph dependence. Secondary score comparisons are exploratory with no post-hoc significance threshold or multiplicity-adjusted claim. Fixed full-compute policies and diagnostic allocation use different nominal budgets and exclude overhead. No predictions were fabricated or extrapolated. Arrays are derived exclusively from the validated original cache.

## G. Implications for Adaptive GPM
Use effect sizes and intervals, not an imposed significance cutoff. Strong positive benefit association and a meaningful paired gain over random allocation would support studying a validation-calibrated stopping rule next (e.g., add batches of walks until successive probability means stabilize). This rule is a proposal only. If higher K stabilizes predictions with negligible accuracy benefit, study accuracy-cost tradeoffs. If benefit prediction is weak or negative, investigate confidently incorrect nodes, coverage limitations and uncertainty that sampling cannot resolve before implementing Adaptive GPM. The measured evidence and checkpoint limitations should guide that decision; a correlation alone is insufficient to justify an efficient adaptive method.

Artifacts: `per_node.csv`, `predictions.npz`, `metrics.json`, `aggregate_means.csv`, `associations.csv`, `policies.csv`, `random_allocations.csv`, and three PNG/PDF figure pairs. `predictions.npz` preserves class probabilities and predicted classes, node IDs/labels, K/repetition IDs, group membership, split and checkpoint metadata. See `../README.md` for execution.

### Measured interpretation and score comparison

| Score | Spearman with error improvement | Spearman with baseline error | Benefit AUROC | Benefit AP | High-error AUROC | High-error AP |
|---|---:|---:|---:|---:|---:|---:|
| sampling_variance | 0.315360 | 0.558481 | 0.946821 | 0.646058 | 0.733512 | 0.261845 |
| entropy | 0.285764 | 0.621940 | 0.913214 | 0.481262 | 0.846231 | 0.367089 |
| negative_confidence | 0.282671 | 0.630694 | 0.915500 | 0.480435 | 0.853661 | 0.412459 |
| negative_margin | 0.281423 | 0.633266 | 0.916245 | 0.490147 | 0.855780 | 0.402243 |
| variation_ratio | 0.352823 | 0.722300 | 0.862737 | 0.506350 | 0.726342 | 0.294012 |

Mean covariance trace at K=16: 0.015267 [95% CI 0.014055, 0.016522]; K=128: 0.002179 [95% CI 0.001937, 0.002441].

For this cache, uncertainty predicts both baseline difficulty (rank correlation 0.558) and improvement (0.315). The positive allocation gain over random is 0.861 percentage points, with paired 95% interval [0.655, 1.071]. Most improvement concentrates in the top quintile. Low variability does not imply correctness: 5.81% of the bottom quintile remain wrong in all observed K=128 outcome runs.

The observed positive benefit association and improvement over random are consistent with decision case A for this checkpoint. They support investigating a simple stopping-rule experiment next after restoring and verifying original data/checkpoint provenance, ideally with independent trained seeds. They do not establish the value of a practical adaptive policy: entropy/confidence/margin perform similarly in retrospective allocation, and the cost of repeated uncertainty-estimation passes has not been recovered. Case B (only stabilization, barely any accuracy change) is a less complete description of these results because measured error also falls; resolving the practical size of that gain requires validation and compute measurements. Case C remains a relevant failure-mode analysis for confidently incorrect low-variability nodes. No Adaptive GPM has been implemented.
