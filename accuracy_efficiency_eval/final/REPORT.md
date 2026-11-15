# Final accuracy-efficiency results: Amazon Computers, one checkpoint

**All three live adaptive rules stayed within 0.5 points of fixed K = 128 while using about 47 patterns' worth of forward passes per node instead of 128. They also beat fixed budgets and random allocation at the same pattern cost. In one matched timing round on a T4, they ran 2.3 to 2.7 times faster than fixed K = 128.**

## Setup

- **Runs:** Arvind's live runs, with checkpoint `epoch_380.pt`, 50 repetitions, and 2,751 test nodes.
- **Thresholds:** Maximum confidence, frozen from 2,750 validation nodes at a target of 48 pattern-passes.
- **Cost:** Patterns sent through GPM per node, with every pass counted. No encodings were reused.
- **Intervals:** 95% node bootstrap, paired by node. Walks differ between runs, so these are not same-walk pairs.

## Live results

Each gap is the rule's accuracy minus the baseline's, in percentage points. Fixed K = 128 was 91.68% in the nested export and 91.72% in the fresh export.

| Rule | Accuracy | Pattern-passes | Live time, 50 repetitions | Vs fixed K = 128 | Vs fixed at same cost | Vs random at same cost |
|---|---:|---:|---:|---|---|---|
| Nested four-stage | 91.65% | 46.9 | 12.32 s | -0.03 (-0.10 to +0.04) | +0.32 (+0.22 to +0.43) | +0.89 (+0.74 to +1.06) |
| Fresh four-stage | 91.66% | 47.0 | 12.13 s | -0.05 (-0.12 to +0.01) | +0.39 (+0.28 to +0.51) | +0.87 (+0.69 to +1.04) |
| Fresh two-stage | 91.63% | 46.7 | 11.43 s | -0.09 (-0.16 to -0.03) | +0.35 (+0.24 to +0.47) | +0.83 (+0.66 to +1.00) |

Fixed at the same cost is interpolated between K = 32 and K = 64. It is a mixture of those two budgets, not a measured run.

## What this means

- **Success target:** On pattern cost, all three rules meet it. Accuracy stayed within 0.5 points, at about 37% of fixed K = 128's pattern-passes.
- **Nested four-stage:** It is the proposal's method, and its gap to fixed K = 128 is not distinguishable from zero. It also samples the fewest patterns, at 31.4 per node.
- **Two-stage:** It has a small but measurable accuracy loss compared with fixed K = 128, and the shortest live time of the three. Differences this small need matched timing before choosing between rules.
- **Replay was reliable:** Replaying the frozen thresholds on exported predictions landed within 0.05 points and 0.3 pattern-passes of the live runs.
- **Nesting concern:** The pilot's worry that fresh replay was optimistic did not hold here. Nested and fresh rules performed almost the same.
- **Repeated sampling:** Offline, it did not match adaptive inference at lower cost. At 64 pattern-passes it scored 91.45% to 91.46%.
- **Aggressive thresholds:** The cheapest threshold within 0.5 points on validation nodes cost 20 to 23 pattern-passes, but missed the target on test nodes, with drops of 0.60 and 0.79 points. It should not be used. Target 48 is the safer operating point.

## Matched timing on Colab

All nine methods ran in one session on a Tesla T4 (torch 2.11), over all 2,751 test nodes and 50 repetitions. Thresholds were the frozen ones from Arvind's runs. **Only one round completed**, so there is no estimate of timing variability yet.

| Method | Accuracy | Pattern-passes | ms per node | Speedup vs fixed K = 128 | Peak GPU MB |
|---|---:|---:|---:|---:|---:|
| Fixed K = 16 | 89.50% | 16.0 | 0.174 | 7.80 | 496 |
| Fixed K = 32 | 90.21% | 32.0 | 0.332 | 4.08 | 857 |
| **Fresh two-stage** | **90.72%** | **52.5** | **0.508** | **2.67** | 1,252 |
| **Nested four-stage** | **90.68%** | **53.4** | **0.583** | **2.32** | 812 |
| **Fresh four-stage** | **90.66%** | **53.8** | **0.589** | **2.30** | 801 |
| Fixed K = 64 | 90.57% | 64.0 | 0.667 | 2.03 | 1,580 |
| Repeated, 4 passes at K = 16 | 90.54% | 64.0 | 0.693 | 1.95 | 496 |
| Fixed K = 128 | 90.73% | 128.0 | 1.354 | 1.00 | 3,026 |
| Repeated, 8 passes at K = 16 | 90.74% | 128.0 | 1.387 | 0.98 | 496 |

- **Speedup:** All three adaptive rules ran 2.3 to 2.7 times faster than fixed K = 128, while scoring 0.01 to 0.07 points below it. That meets the 0.5-point target in measured time.
- **Better than cheaper fixed budgets:** Every adaptive rule was both faster and more accurate than fixed K = 64.
- **Two-stage was fastest:** It had the best accuracy and speed of the three rules here, but used more peak memory than the four-stage rules.
- **Memory:** Every adaptive rule used less than half the peak GPU memory of fixed K = 128.
- **Time tracks pattern-passes closely:** Fixed K time was nearly proportional to K.

**Environment effect:** Fixed K = 128 scored 90.73% here, against 91.72% in Arvind's L40S runs and 90.88% in the slide runs, with the same checkpoint and test nodes. Adaptive rules also used about 53 pattern-passes here instead of 47, because confidence values shifted while thresholds stayed frozen. This supports an environment or positional-encoding cause for the accuracy difference. Comparisons within this run remain fair, because every method shared one environment, but accuracies should not be compared across runs.

## Unresolved issue

These runs are about 0.8 points more accurate than the earlier saved runs behind the slides. On the same 2,751 test nodes, fixed K = 128 scored 90.88% before and 91.7% now.

- Test node IDs and labels match exactly, and validation nodes do not overlap with test nodes.
- The change is about 10 times larger than the difference between two groups of repetitions. Across all 50 runs, the earlier and new K = 128 predictions agree on 97.6% of nodes.
- The cause is unknown. The Laplacian positional encoding is a possible cause, because GPM recomputes it each time data loads, but this has not been tested. An environment change is also possible.

Comparisons inside the new runs are consistent with each other, so the results above stand on their own. They should not be mixed with the slide figures until this is explained.

## Limits

- One checkpoint on one dataset. WikiCS and the other checkpoints are pending.
- Validation-tuned thresholds are frozen, but test nodes were examined in earlier exploratory work.
- Matched timing has one round, so its variability is unknown.

## Checks

- All file hashes match `SHA256.json`, and the checkpoint matches the hash recorded in earlier work.
- Test nodes, decisions, and pattern-pass accounting are consistent for all three rules.
- All three frozen thresholds were reproduced exactly from validation data.
- `evaluate.py` now uses recorded pattern-passes, which fixes the two-stage cost issue Arvind reported.
- 40 automated tests pass. `matched_timing.py` ran successfully on Colab for one round.
