# Final accuracy-efficiency results

**All three live adaptive rules stayed within 0.5 points of fixed K = 128 while using about 47 patterns' worth of forward passes per node instead of 128. They also beat fixed budgets and random allocation at the same pattern cost. In three matched timing rounds on a T4, they ran 2.3 to 2.4 times faster than fixed K = 128.**

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

All nine methods ran in one session on a Tesla T4 (torch 2.11), over all 2,751 test nodes and 50 repetitions. They ran in three rounds, in a shuffled order each round. Each speedup is fixed K = 128's time divided by the method's time within the same round. The table shows the median, with the range across rounds. Data and chart are in `timing/`.

| Method | Accuracy | Pattern-passes | ms per node | Speedup vs fixed K = 128 (range) | Peak GPU MB |
|---|---:|---:|---:|---:|---:|
| Fixed K = 16 | 89.49% | 16.0 | 0.174 | 7.81 (7.79 to 7.85) | 496 |
| Fixed K = 32 | 90.24% | 32.0 | 0.333 | 4.08 (4.07 to 4.11) | 857 |
| **Fresh two-stage** | **90.72%** | **52.3** | **0.558** | **2.44 (2.43 to 2.75)** | 1,252 |
| **Nested four-stage** | **90.68%** | **53.4** | **0.587** | **2.32 (2.31 to 2.32)** | 812 |
| **Fresh four-stage** | **90.70%** | **53.9** | **0.591** | **2.30 (2.30 to 2.35)** | 812 |
| Fixed K = 64 | 90.58% | 64.0 | 0.671 | 2.03 (2.03 to 2.03) | 1,580 |
| Repeated, 4 passes at K = 16 | 90.55% | 64.0 | 0.694 | 1.96 (1.96 to 1.96) | 496 |
| Fixed K = 128 | 90.73% | 128.0 | 1.360 | 1.00 | 3,026 |
| Repeated, 8 passes at K = 16 | 90.73% | 128.0 | 1.390 | 0.98 (0.98 to 0.98) | 496 |

- **Speedup:** All three adaptive rules ran 2.3 to 2.4 times faster than fixed K = 128 at the median, while scoring 0.01 to 0.05 points below it. That meets the 0.5-point target in measured time.
- **Stable timing:** Nested four-stage varied by less than 1% across rounds. Two-stage was faster in its first round, at 2.75, than in the other two, at 2.43 and 2.44, so its median is the safer figure.
- **Better than cheaper fixed budgets:** Every adaptive rule was both faster and more accurate than fixed K = 64.
- **Repeated sampling:** It was slower than fixed K at the same pattern cost and never matched adaptive inference.
- **Memory:** Every adaptive rule used less than half the peak GPU memory of fixed K = 128. Repeated sampling used the least, because it runs small batches.
- **Time tracks pattern-passes closely:** Fixed K time was nearly proportional to K.
- **Accuracy is a single sample per method:** Rounds reuse the same sampling seeds, so accuracies barely change across rounds. Accuracy differences of a few hundredths of a point here should not be read as real.

### Timing on a second GPU and on WikiCS

Ahana repeated the matched timing for Computers checkpoint 1 on an RTX 5060 laptop GPU, with torch 2.7.1, 10 repetitions, and three rounds. Her threshold was 0.775, recalibrated on validation nodes in her environment. Arvind's was 0.835. The file is `timing/ahana_rtx5060_computers_ckpt1.json`, and its summary matches its per-round data.

| Method | Accuracy | ms per node | Speedup vs fixed K = 128 (range) | Peak GPU MB |
|---|---:|---:|---:|---:|
| Fixed K = 32 | 90.02% | 0.407 | 3.60 (3.57 to 3.62) | 856 |
| Nested four-stage | 90.52% | 0.599 | 2.45 (2.40 to 2.45) | 653 |
| Fixed K = 64 | 90.37% | 0.739 | 1.98 (1.98 to 1.99) | 1,579 |
| Repeated, 4 passes at K = 16 | 90.33% | 0.817 | 1.80 (1.79 to 1.80) | 495 |
| Fixed K = 128 | 90.60% | 1.465 | 1.00 | 3,025 |

On a different GPU and library version, nested four-stage was again about 2.4 times faster than fixed K = 128, 0.08 points less accurate, and both faster and more accurate than fixed K = 64. The speedup therefore does not depend on one machine.

She also timed WikiCS checkpoint 1 on the same GPU, using its frozen threshold of 0.784. The file is `timing/ahana_rtx5060_wikics_ckpt1.json`, and its summary also matches its per-round data.

| Method | Accuracy | ms per node | Speedup vs fixed K = 128 (range) | Peak GPU MB |
|---|---:|---:|---:|---:|
| Fixed K = 32 | 79.93% | 0.365 | 3.57 (3.57 to 3.59) | 828 |
| Nested four-stage | 80.39% | 0.585 | 2.23 (2.22 to 2.24) | 671 |
| Fixed K = 64 | 80.34% | 0.656 | 1.99 (1.98 to 1.99) | 1,551 |
| Repeated, 4 passes at K = 16 | 80.32% | 0.734 | 1.78 (1.78 to 1.78) | 468 |
| Fixed K = 128 | 80.62% | 1.304 | 1.00 | 2,997 |

On WikiCS, nested four-stage was 2.23 times faster than fixed K = 128 and 0.23 points less accurate, within the 0.5-point target. It was faster than fixed K = 64 and only 0.05 points more accurate, a negligible difference. This matches the accuracy results: on WikiCS, adaptive inference mainly buys speed, with little accuracy advantage over a cheaper fixed budget.

**Environment effect:** Fixed K = 128 scored 90.73% here, against 91.72% in Arvind's L40S runs and 90.88% in the slide runs, with the same checkpoint and test nodes. Adaptive rules also used about 53 pattern-passes here instead of 47, because confidence values shifted while thresholds stayed frozen. This supports an environment or positional-encoding cause for the accuracy difference. Comparisons within this run remain fair, because every method shared one environment, but accuracies should not be compared across runs.

## Across datasets and checkpoints

Ahana ran nested four-stage adaptive inference on two more Computers checkpoints and three WikiCS checkpoints. Computers checkpoint 1 is Arvind's run. Each threshold was frozen from that checkpoint's validation nodes at a target of 48 pattern-passes. Data are in `checkpoints/` and `combined/`.

| Dataset, checkpoint | Accuracy | Pattern-passes | Vs fixed K = 128 | Vs fixed at same cost | Vs random at same cost |
|---|---:|---:|---|---|---|
| Computers 1 | 91.65% | 46.9 | -0.03 (-0.10 to +0.04) | +0.32 (+0.22 to +0.43) | +0.89 (+0.74 to +1.06) |
| Computers 2 | 90.99% | 45.7 | 0.00 (-0.16 to +0.16) | +0.31 (+0.14 to +0.49) | +0.72 (+0.49 to +0.95) |
| Computers 3 | 91.65% | 46.7 | -0.12 (-0.28 to +0.05) | +0.32 (+0.12 to +0.53) | +0.90 (+0.65 to +1.16) |
| WikiCS 1 | 80.35% | 49.9 | -0.27 (-0.41 to -0.11) | +0.16 (-0.01 to +0.32) | +0.77 (+0.57 to +0.96) |
| WikiCS 2 | 80.14% | 49.4 | -0.34 (-0.50 to -0.18) | +0.07 (-0.10 to +0.22) | +0.79 (+0.60 to +0.97) |
| WikiCS 3 | 80.52% | 49.2 | -0.21 (-0.36 to -0.07) | +0.18 (+0.03 to +0.33) | +0.79 (+0.59 to +0.99) |

- **The 0.5-point target held everywhere:** All six checkpoints stayed within 0.5 points of fixed K = 128. WikiCS 2 came closest, with an interval reaching -0.50.
- **WikiCS has a real accuracy cost:** On Computers, the gap to fixed K = 128 is not distinguishable from zero. On WikiCS, every checkpoint loses 0.21 to 0.34 points, and each interval excludes zero.
- **The gain over fixed budgets is smaller on WikiCS:** It averaged +0.13 points, against +0.31 on Computers. For two of three WikiCS checkpoints, the interval includes zero.
- **Random allocation is clearly worse everywhere:** Adaptive gained 0.72 to 0.90 points over it at the same cost.
- **Repeated sampling never matched adaptive inference at lower cost.**

### Why WikiCS gains less

Using validation nodes, half of each run's repetitions measured which nodes improved from K = 16 to K = 128. The other half supplied each node's K = 16 confidence.

| Measure | Computers | WikiCS |
|---|---:|---:|
| AUROC: low confidence finds nodes that improve | 0.89 to 0.91 | 0.85 |
| Nodes wrong at K = 128 in every repetition | 7.0% to 8.4% | 14.5% to 15.1% |
| Nodes wrong at both K = 16 and K = 128 in every repetition | 5.4% to 6.0% | 10.1% to 10.9% |

Confidence identifies nodes that benefit less well on WikiCS, and WikiCS has about twice as many nodes that more patterns never fix. Such nodes tend to look uncertain, so adaptive inference spends extra patterns on them without gain. This evidence is consistent with the smaller WikiCS gain, but it does not prove the cause.

The repetition count differs between runs: 50 for Computers 1 and 10 for the others. Per-node counts therefore have different precision across checkpoints.

### Not yet verified

- **Environments:** Computers 1 ran on an L40S, and the other checkpoints on an RTX 5060 laptop GPU with torch 2.7.1. Comparisons within each checkpoint are fair, but absolute accuracies are not comparable across environments.

## Unresolved issue

These runs are about 0.8 points more accurate than the earlier saved runs behind the slides. On the same 2,751 test nodes, fixed K = 128 scored 90.88% before and 91.7% now.

- Test node IDs and labels match exactly, and validation nodes do not overlap with test nodes.
- The change is about 10 times larger than the difference between two groups of repetitions. Across all 50 runs, the earlier and new K = 128 predictions agree on 97.6% of nodes.
- The cause is unknown. The Laplacian positional encoding is a possible cause, because GPM recomputes it each time data loads, but this has not been tested. An environment change is also possible.

Comparisons inside the new runs are consistent with each other, so the results above stand on their own. They should not be mixed with the slide figures until this is explained.

## Limits

- Validation-tuned thresholds are frozen, but test nodes were examined in earlier exploratory work.
- Matched timing covers one checkpoint per dataset: Computers on a Tesla T4 and an RTX 5060 laptop GPU, and WikiCS on the RTX 5060.

## Checks

- All file hashes match `SHA256.json`, and the checkpoint matches the hash recorded in earlier work.
- Test nodes, decisions, and pattern-pass accounting are consistent for all three rules.
- All three frozen thresholds were reproduced exactly from validation data.
- `evaluate.py` now uses recorded pattern-passes, which fixes the two-stage cost issue Arvind reported.
- 40 automated tests pass. `matched_timing.py` ran successfully on Colab for three rounds.
