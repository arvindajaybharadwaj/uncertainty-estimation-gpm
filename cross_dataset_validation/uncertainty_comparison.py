"""Repeat the cheap uncertainty comparison on validation nodes, for every checkpoint.

This is Table 1 of Cheap_Uncertainty.ipynb, computed the same way:
  - Maximum confidence: 1 - max probability of one K=16 run.
  - Predictive entropy of that same run.
  - Sampling disagreement: total variation between two independent K=16 runs.
Benefit is accuracy at K=128 minus accuracy at K=16, per node, averaged over the
remaining runs. Scores use runs 0 and 1 and benefit uses runs 2 onwards, so
they never share a run. With 10 repetitions benefit has 8 runs, against 25 in
the notebook, so it is noisier.

The notebook's 25-pass reference is not repeated: it needs 25 runs for scores alone.

Example, from the repository root:
    python cross_dataset_validation/uncertainty_comparison.py
    python cross_dataset_validation/uncertainty_comparison.py wikics:1=cross_dataset_validation/runs/wikics/ckpt1/final_predictions.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from common import HERE, RUNS

METHODS = ('Maximum confidence', 'Predictive entropy', 'Sampling disagreement')
PASSES = dict(zip(METHODS, (1, 1, 2)))
SCORE_RUNS = 2
TOP_FRACTION = 0.20


def load(path, split='val'):
    """Probabilities [K, run, node, class] for one node split of a final predictions file."""
    with np.load(path, allow_pickle=False) as z:
        if 'node_split' not in z.files:
            raise ValueError(f'{path} has no node_split, so it is not a final predictions file.')
        keep = z['node_split'].astype(str) == split
        if not keep.any():
            raise ValueError(f'{path} has no {split} nodes.')
        return dict(p=z['probabilities'][:, :, keep].astype(np.float64), y=z['labels'][keep],
                    node_ids=z['node_ids'][keep], ks=[int(k) for k in z['k_values']])


def cheap_scores(p1, p2):
    """Higher means more uncertain. Same definitions as Cheap_Uncertainty.ipynb cell 7."""
    return {
        'Maximum confidence': 1 - p1.max(-1),
        'Predictive entropy': -(p1 * np.log(np.clip(p1, 1e-12, 1))).sum(-1),
        'Sampling disagreement': 0.5 * np.abs(p1 - p2).sum(-1),
    }


def split_runs(data):
    runs = data['p'].shape[1]
    if runs < SCORE_RUNS + 2:
        raise ValueError(f'Need at least {SCORE_RUNS + 2} repetitions; found {runs}.')
    return np.arange(SCORE_RUNS), np.arange(SCORE_RUNS, runs)


def accuracy(p, y):
    """Per-node accuracy over runs, for p of shape [run, node, class]."""
    return (p.argmax(-1) == y).mean(0)


def benefit(data, runs):
    i16, i128 = data['ks'].index(16), data['ks'].index(128)
    acc16 = accuracy(data['p'][i16][runs], data['y'])
    acc128 = accuracy(data['p'][i128][runs], data['y'])
    return acc128 - acc16, 1 - acc16, 1 - acc128


def compare(data):
    """One row per cheap score, as in Table 1."""
    score_runs, benefit_runs = split_runs(data)
    p16 = data['p'][data['ks'].index(16)]
    scores = cheap_scores(p16[score_runs[0]], p16[score_runs[1]])
    gain, error16, error128 = benefit(data, benefit_runs)
    n = int(np.floor(TOP_FRACTION * len(gain)))
    rows = []
    for name in METHODS:
        score = scores[name]
        top = np.lexsort((data['node_ids'], -score))[:n]  # most uncertain first, ties by node ID
        rho = spearmanr(score, gain).statistic if np.ptp(score) > 0 and np.ptp(gain) > 0 else np.nan
        auroc = roc_auc_score(gain > 0, score) if len(np.unique(gain > 0)) == 2 else np.nan
        rows.append(dict(method=name, passes=PASSES[name], patterns_per_node=16 * PASSES[name],
                         spearman=float(rho), auroc=float(auroc),
                         top20_benefit_pp=100 * float(gain[top].mean()) if n else np.nan,
                         top20_error16_pct=100 * float(error16[top].mean()) if n else np.nan,
                         top20_error128_pct=100 * float(error128[top].mean()) if n else np.nan,
                         all_benefit_pp=100 * float(gain.mean()), nodes=len(gain), top_nodes=n,
                         benefit_runs=len(benefit_runs)))
    return pd.DataFrame(rows)


def parse_runs(items):
    """'dataset:checkpoint=path' arguments, or every final_predictions.npz under runs/."""
    if not items:
        found = sorted(RUNS.glob('*/ckpt*/final_predictions.npz'))
        return [(p.parent.parent.name, p.parent.name.removeprefix('ckpt'), p) for p in found]
    runs = []
    for item in items:
        name, path = item.split('=', 1)
        dataset, checkpoint = name.split(':', 1)
        runs.append((dataset, checkpoint, Path(path)))
    return runs


def summarise(table):
    """Mean and spread across checkpoints, per dataset and score."""
    columns = ['spearman', 'auroc', 'top20_benefit_pp', 'all_benefit_pp']
    grouped = table.groupby(['dataset', 'method'], sort=False)[columns]
    out = grouped.mean().add_suffix('_mean').join(grouped.std().add_suffix('_sd'))
    out['checkpoints'] = grouped.size()
    return out.reset_index()


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('runs', nargs='*', help='dataset:checkpoint=final_predictions.npz; default: all under runs/')
    a.add_argument('--split', default='val', choices=['val', 'test'],
                   help='val for model selection; test only after thresholds are frozen')
    a.add_argument('--output', default=str(HERE / 'results' / 'uncertainty'))
    args = a.parse_args()
    runs = parse_runs(args.runs)
    if not runs:
        raise SystemExit('No predictions files found. Run run_checkpoint.py first.')
    tables = []
    for dataset, checkpoint, path in runs:
        table = compare(load(path, args.split))
        table.insert(0, 'checkpoint', checkpoint)
        table.insert(0, 'dataset', dataset)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f'per_checkpoint_{args.split}.csv', index=False)
    summary = summarise(table)
    summary.to_csv(out / f'summary_{args.split}.csv', index=False)
    pd.set_option('display.width', 200)
    print(table.round(4).to_string(index=False))
    print()
    print(summary.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
