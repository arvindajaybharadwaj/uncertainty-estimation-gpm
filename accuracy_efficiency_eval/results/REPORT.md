# Accuracy-efficiency pilot: Amazon Computers

*Superseded by `../final/REPORT.md`, which uses live runs and validation-tuned thresholds. Pilot accuracies come from an earlier run and are not directly comparable.*

**Adaptive replay matched fixed K = 128 accuracy at about a third of its nominal work, and beat fixed, random, and repeated sampling at similar nominal cost. This is a pilot: it does not yet prove a speedup, because no real adaptive run or adaptive timing exists.**

## Setup

- **Data:** Saved predictions from one Computers checkpoint, with 2,751 test nodes and 50 repetitions at each K.
- **Split:** 1,375 calibration and 1,376 evaluation nodes, chosen at random.
- **Adaptive rule:** Stop at the first of K = 16, 32, 64, 128 where maximum confidence reaches a threshold.
- **Threshold choice:** For cost targets, thresholds use calibration nodes and no labels.
- **Cost:** Patterns sent through GPM per node, with every pass counted and no reuse. Reaching K = 128 costs 240.
- **Intervals:** 95% paired bootstrap over evaluation nodes, with 2,000 draws.

## Results

| Method | Cost | Accuracy |
|---|---:|---:|
| Fixed K = 16 | 16 | 90.12% |
| Fixed K = 32 | 32 | 90.74% |
| Fixed K = 64 | 64 | 90.97% |
| Fixed K = 128 | 128 | 91.15% |
| Repeated, 2 passes at K = 16 | 32 | 90.60% |
| Repeated, 4 passes at K = 16 | 64 | 90.98% |
| Repeated, 8 passes at K = 16 | 128 | 91.05% |
| Random, 14.3% at K = 128 | 32 | 90.27% |
| Random, 28.6% at K = 128 | 48 | 90.42% |
| Random, 42.9% at K = 128 | 64 | 90.57% |
| Random, 71.4% at K = 128 | 96 | 90.86% |
| Adaptive, threshold 0.697 | 29.9 | 91.00% |
| Adaptive, threshold 0.818 | 44.1 | 91.15% |
| Adaptive, threshold 0.882 | 58.6 | 91.14% |
| Adaptive, threshold 0.933 | 89.5 | 91.15% |

At a cost of 44.1, adaptive replay compared as follows:

| Comparison | Adaptive gain, percentage points | 95% interval |
|---|---:|---:|
| Fixed K = 128 | -0.01 | -0.05 to +0.04 |
| Fixed budgets, interpolated to the same cost | +0.32 | +0.15 to +0.49 |
| Random allocation at the same cost | +0.76 | +0.52 to +0.99 |

**The 0.5-point target:** Using calibration labels, the cheapest threshold within 0.5 points of fixed K = 128 was 0.601. On evaluation nodes, it cost 23.1 and scored 90.76%, a drop of 0.39 points (interval 0.26 to 0.53). The upper end slightly exceeds 0.5, and 93.6% of decisions stopped at K = 16.

**Split sensitivity:** Across 20 other random splits at a target of 64, the adaptive gap to fixed K = 128 ranged from -0.03 to +0.01 points, and cost ranged from 57.5 to 69.0.

**Repeated sampling:** It matched fixed K at a cost of 64, but fell 0.11 points behind at 128 with 8 passes at K = 16.

## Follow-up analyses

These use the same split and are also exploratory. Details are in `incremental_batches.csv`, `two_stage.csv`, and `score_comparison.csv`.

**Choice of score:** Confidence, margin, and entropy performed almost identically, within 0.05 points at each target. Maximum confidence is the simplest choice to carry forward, subject to validation.

**Incremental batches:** This rule adds independent K = 16 passes, so no earlier work is redone and its nested structure is closer to real inference. At a cost of 44.7, it reached 91.05%, 0.11 points below fixed K = 128 (interval -0.32 to +0.10). Its gain over fixed budgets at the same cost was +0.22 points, and that interval includes zero (-0.05 to +0.48). This smaller gain supports the concern that the fresh-walk replay is optimistic, though this rule is a different method and uses only 6 independent groups. Its accuracy is also capped near 91.05%, the result of averaging 8 passes.

**Two-stage rule:** Few nodes stopped at the middle steps, so this rule uses K = 16 and sends uncertain nodes straight to a fresh K = 128 pass. At a cost of 44.9, with 22.6% of nodes escalated, it reached 91.13%, 0.03 points below fixed K = 128 (interval -0.06 to 0.00). This is close to the four-step replay.

This rule matters for two reasons. It is the only tested rule whose replay matches the real procedure, because a two-stage run uses independent walks, as the saved runs do. Limitation 1 below therefore does not apply to it, and neither does the multiscale issue, because no encodings are reused. Its drawback is that the first 16 patterns are not reused, which departs from the proposal, and it still needs validation-based tuning and real timing.

## Why this is not yet a final result

1. **Fresh walks, not nested ones.** The saved runs sample new walks at each K, so an unlucky K = 16 sample does not carry into K = 32. Real nested inference keeps those patterns, so this replay may be optimistic.
2. **Test nodes.** These nodes were already examined in earlier analysis. Final thresholds must use validation nodes.
3. **One checkpoint and one dataset.** WikiCS and the other checkpoints are still untested.
4. **Nominal cost is not time.** The transformer attends across patterns, so time may grow faster than K. Escalated nodes also form smaller batches, which can use the GPU less efficiently. Overheads such as grouping nodes are not included.
5. **Uncertainty score.** Only maximum confidence is reported here. The choice of score should be made on validation data.

## Issue for the adaptive implementation

Encoding reuse is not exact with a plain first-K slice. In `encoder.py`, a pattern's multiscale mask depends on its position relative to the total number of patterns. With K = 16, position 4 is in the second scale group. With K = 32, it is in the first. Reused encodings would therefore differ from a fresh forward pass, unless patterns are kept in a separate pool for each scale. Sampling 128 walks and slicing is still valid for nesting without reuse. `reuse_cost` in `curve.csv` should not be reported until this is resolved.

## Checks

- 21 automated tests pass, including a hand-worked stopping example.
- Recomputing the 44.1-cost result with plain loops gave the same accuracy and cost.
- Fixed K accuracy across all 50 runs reproduces the slide figures: 89.73% at K = 16 and 90.88% at K = 128.
- `measure_timing.py` has only had its calculations tested. It has not run on GPM.

## Next steps

1. Run `measure_timing.py` with the checkpoint to replace nominal cost with real time.
2. Rerun this evaluation on Arvind's nested validation and test predictions.
3. Add his live adaptive decisions and measured time with `--adaptive-decisions`.
