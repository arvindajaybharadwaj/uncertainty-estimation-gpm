"""Build the cross-dataset results table from every finished checkpoint run.

Reads, for each runs/<dataset>/ckpt<id>/ folder:
  final_predictions.npz   fixed K=128 accuracy on test nodes, per repetition
  adaptive_decisions.npz  live adaptive predictions, cost and time
  eval/summary.json       evaluate.py's paired comparisons and intervals
  timing.json             optional, from time_methods.py

Writes results/per_checkpoint.csv (one row per checkpoint) and
results/per_dataset.csv (mean over checkpoints). Variation is reported two ways:
  checkpoint_sd  spread of the checkpoint means, from training
  sampling_sd    spread over sampling repetitions within a checkpoint, averaged
Accuracies are percentages and gaps are percentage points.

Example, from the repository root:
    python cross_dataset_validation/summarize_results.py
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, RUNS

RULE = 'nested_four_stage'
SUCCESS_GAP = 0.5  # percentage points below fixed K=128


def per_repetition_accuracy(predicted, labels):
    return 100 * (predicted == labels).mean(1)


def checkpoint_row(folder):
    with np.load(folder / 'final_predictions.npz', allow_pickle=False) as z:
        test = z['node_split'].astype(str) == 'test'
        k128 = list(z['k_values']).index(128)
        labels, node_ids = z['labels'][test], z['node_ids'][test]
        fixed = per_repetition_accuracy(z['probabilities'][k128][:, test].argmax(-1), labels)
    with np.load(folder / 'adaptive_decisions.npz', allow_pickle=False) as z:
        if not np.array_equal(z['node_ids'], node_ids):
            raise ValueError(f'{folder}: decision nodes differ from the test nodes in the predictions.')
        adaptive = per_repetition_accuracy(z['predicted'], labels)
        final_k = z['final_k']
        live = dict(pattern_passes=float(z['pattern_passes'].mean()), sampled_patterns=float(z['sampled_patterns'].mean()),
                    seconds_total=float(z['seconds_total']), threshold=float(z['threshold']),
                    smoke_test=bool(z['smoke_test']))
    if live['smoke_test']:
        raise ValueError(f'{folder} is a smoke test and cannot be included in final results.')
    with open(folder / 'eval' / 'summary.json') as f:
        comparisons = json.load(f)['live_adaptive'][RULE]['comparisons']
    gap = comparisons['fixed_128']
    interpolated = next(v for k, v in comparisons.items() if k.startswith('fixed_') and k != 'fixed_128')
    row = dict(
        fixed_128_acc=fixed.mean(), fixed_128_sampling_sd=fixed.std(ddof=1) if len(fixed) > 1 else np.nan,
        adaptive_acc=adaptive.mean(), adaptive_sampling_sd=adaptive.std(ddof=1) if len(adaptive) > 1 else np.nan,
        gap_vs_fixed_128_pp=100 * gap['diff'], gap_ci_low_pp=100 * gap['ci_low'], gap_ci_high_pp=100 * gap['ci_high'],
        gain_vs_fixed_same_cost_pp=100 * interpolated['diff'],
        gain_vs_random_same_cost_pp=100 * comparisons['random_same_cost']['diff'],
        pattern_passes=live['pattern_passes'], sampled_patterns=live['sampled_patterns'],
        cost_vs_fixed_128=live['pattern_passes'] / 128, threshold=live['threshold'],
        live_ms_per_node=1000 * live['seconds_total'] / final_k.size,
        **{f'stop_at_{k}_pct': 100 * float((final_k == k).mean()) for k in (16, 32, 64, 128)})
    row['within_target'] = row['gap_vs_fixed_128_pp'] >= -SUCCESS_GAP
    row['within_target_whole_ci'] = row['gap_ci_low_pp'] >= -SUCCESS_GAP
    timing = folder / 'timing.json'
    if timing.exists():
        with open(timing) as f:
            summary = {r['method']: r for r in json.load(f)['summary']}
        row.update(time_vs_fixed_128=1 / summary[RULE]['speedup_vs_fixed_128'],
                   speedup_vs_fixed_128=summary[RULE]['speedup_vs_fixed_128'],
                   timed_ms_per_node=summary[RULE]['ms_per_node'],
                   fixed_128_timed_ms_per_node=summary['fixed_128']['ms_per_node'],
                   peak_gpu_mb=summary[RULE]['peak_gpu_mb'], timing_rounds=summary[RULE]['rounds'])
    return row


def per_dataset(table):
    columns = [c for c in ('fixed_128_acc', 'adaptive_acc', 'gap_vs_fixed_128_pp', 'gain_vs_fixed_same_cost_pp',
                           'gain_vs_random_same_cost_pp', 'pattern_passes', 'cost_vs_fixed_128',
                           'speedup_vs_fixed_128', 'time_vs_fixed_128') if c in table]
    grouped = table.groupby('dataset', sort=False)
    out = grouped[columns].mean()
    for column in ('fixed_128_acc', 'adaptive_acc', 'gap_vs_fixed_128_pp'):
        out[f'{column}_checkpoint_sd'] = grouped[column].std()
    out['adaptive_acc_sampling_sd'] = grouped['adaptive_sampling_sd'].mean()
    out['checkpoints'] = grouped.size()
    out['checkpoints_within_target'] = grouped['within_target'].sum()
    return out.reset_index()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--runs', default=str(RUNS))
    a.add_argument('--output', default=str(HERE / 'results'))
    args = a.parse_args()
    folders = sorted(p.parent.parent for p in Path(args.runs).glob('*/ckpt*/eval/summary.json'))
    if not folders:
        raise SystemExit('No finished runs found. Run run_checkpoint.py first.')
    rows = [dict(dataset=f.parent.name, checkpoint=f.name.removeprefix('ckpt'), **checkpoint_row(f)) for f in folders]
    table = pd.DataFrame(rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'per_checkpoint.csv', index=False)
    summary = per_dataset(table)
    summary.to_csv(out / 'per_dataset.csv', index=False)
    pd.set_option('display.width', 200)
    print(table.round(3).to_string(index=False))
    print()
    print(summary.round(3).to_string(index=False))
    print('Saved', out / 'per_checkpoint.csv', 'and', out / 'per_dataset.csv')


if __name__ == '__main__':
    main()
