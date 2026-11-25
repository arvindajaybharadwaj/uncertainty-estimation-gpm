"""Paired comparisons for one checkpoint, and the summary table across checkpoints.

On test nodes, with evaluate.py's paired node bootstrap (2,000 draws, seed
20261105 + 9). "New" results come from this folder's predictions.npz and
decisions.npz; "original" ones from cross_dataset_validation/runs (read only):
final_predictions.npz, fixed_budgets.npz (fixed K=48) and adaptive_decisions.npz.

  1. Main result     cached adaptive  - new fixed K=32, 48, 128
  2. Caching         cached adaptive  - original adaptive
  3. Layout          new fixed K      - original fixed K, for K=32, 48, 64, 128
  4. Replay check    replay of the new predictions at the cached threshold - cached live

Walks differ between the runs being compared, so pairs are by node, not walk.
Writes <folder>/comparison.json and refuses to overwrite it. --summarize writes
results/summary.csv and results/summary_by_dataset.csv, refusing to overwrite
them unless --overwrite is given.

Example, from the repository root:
    python encoder_caching/evaluate_cached.py --dataset wikics --checkpoint-id 1
    python encoder_caching/evaluate_cached.py --summarize
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd

import paths

sys.path.insert(0, str(paths.EVAL_DIR))
import evaluate as ev  # noqa: E402

FIXED = (32, 48, 64, 128)


def fixed_48(path, node_ids, labels, key='fixed_48'):
    """Per-node accuracy of a stored fixed K=48 array, aligned to node_ids."""
    with np.load(path, allow_pickle=False) as z:
        if key == 'fixed_48':
            p, ids = z['fixed_48'], z['node_ids']
        else:  # fixed_budgets.npz from cross_dataset_validation
            p, ids = z['probabilities'][list(z['k_values']).index(48)], z['node_ids']
    lookup = {int(n): i for i, n in enumerate(ids)}
    positions = np.array([lookup[int(n)] for n in node_ids])
    return ev.correct(p[:, positions], labels).mean(0)


def aligned(data, node_ids):
    lookup = {int(n): i for i, n in enumerate(data['node_ids'])}
    return np.array([lookup[int(n)] for n in node_ids])


def compare(dataset, checkpoint_id, smoke):
    new_dir = paths.folder(dataset, checkpoint_id, smoke)
    old_dir = paths.original_folder(dataset, checkpoint_id, smoke)
    new = ev.load_predictions(new_dir / 'predictions.npz')
    old = ev.load_predictions(old_dir / 'final_predictions.npz')
    costs = ev.load_costs(ev.ADAPTIVE_STEPS)[0]

    cached, cached_node, positions = ev.load_adaptive_decisions(new_dir / 'decisions.npz', new['y'],
                                                                new['node_ids'], costs)
    test_ids = new['node_ids'][positions]
    original, original_node, old_positions = ev.load_adaptive_decisions(old_dir / 'adaptive_decisions.npz',
                                                                        old['y'], old['node_ids'], costs)
    if not np.array_equal(old['node_ids'][old_positions], test_ids):
        raise ValueError('The original and new runs have different test nodes.')
    y = new['y'][positions]
    if not np.array_equal(old['y'][old_positions], y):
        raise ValueError('Labels differ between the original and new runs.')

    node = {'cached_adaptive': cached_node, 'original_adaptive': original_node}
    for k in (32, 64, 128):
        node[f'new_fixed_{k}'] = ev.fixed(new, k, positions)
        node[f'original_fixed_{k}'] = ev.fixed(old, k, old_positions)
    node['new_fixed_48'] = fixed_48(new_dir / 'predictions.npz', test_ids, y)
    budgets = old_dir / 'fixed_budgets.npz'
    missing = []
    if budgets.exists():
        node['original_fixed_48'] = fixed_48(budgets, test_ids, y, key='budgets')
    elif smoke:
        missing.append('original_fixed_48')
    else:
        raise FileNotFoundError(f'{budgets} is needed for the original fixed K=48. '
                                'Run cross_dataset_validation/fixed_budget_baseline.py first.')
    scores = ev.adaptive_scores(new, positions, 'confidence')
    node['cached_replay'], step = ev.adaptive(new, positions, scores, cached['threshold'])

    pairs = {'cached_vs_new_fixed_32': ('cached_adaptive', 'new_fixed_32'),
             'cached_vs_new_fixed_48': ('cached_adaptive', 'new_fixed_48'),
             'cached_vs_new_fixed_128': ('cached_adaptive', 'new_fixed_128'),
             'cached_vs_original_adaptive': ('cached_adaptive', 'original_adaptive'),
             **{f'layout_fixed_{k}': (f'new_fixed_{k}', f'original_fixed_{k}') for k in FIXED},
             'replay_vs_cached': ('cached_replay', 'cached_adaptive')}
    comparisons = {}
    for name, (a, b) in pairs.items():
        if a in node and b in node:
            d, low, high = ev.bootstrap_diff(node[a], node[b], paths.BOOTSTRAP, paths.BOOTSTRAP_SEED)
            comparisons[name] = dict(diff=100 * d, ci_low=100 * low, ci_high=100 * high)
    return dict(dataset=dataset, checkpoint_id=checkpoint_id, smoke_test=smoke, test_nodes=len(positions),
                threshold=dict(cached=cached['threshold'], original=original['threshold']),
                cost=dict(cached=cached['rerun_cost'], original=original['rerun_cost'],
                          cached_replay=float(ev.stop_costs(costs)[0][step].mean()),
                          cached_sampled=cached['sampled_patterns'], original_sampled=original['sampled_patterns']),
                stop_share=dict(cached=cached['stop_share'], original=original['stop_share']),
                accuracy={name: 100 * float(v.mean()) for name, v in node.items()},
                comparisons=comparisons, missing=missing)


def run(args):
    output = paths.folder(args.dataset, args.checkpoint_id, args.smoke) / 'comparison.json'
    paths.refuse_existing(output)
    result = compare(args.dataset, args.checkpoint_id, args.smoke)
    with open(output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nTest nodes: {result['test_nodes']:,}. Thresholds: cached {result['threshold']['cached']:.4f}, "
          f"original {result['threshold']['original']:.4f}. Cost: cached {result['cost']['cached']:.1f}, "
          f"original {result['cost']['original']:.1f} pattern-passes")
    for name, value in result['accuracy'].items():
        print(f'  {name:<22} {value:6.2f}%')
    for name, c in result['comparisons'].items():
        print(f"  {name:<30} {c['diff']:+.2f}  ({c['ci_low']:+.2f}, {c['ci_high']:+.2f})")
    if result['missing']:
        print('  Not compared (smoke run, original file absent):', ', '.join(result['missing']))
    print('Saved', output)


def verdict(c):
    if c['ci_low'] > 0:
        return 'higher'
    if c['ci_high'] < 0:
        return 'lower'
    return 'no clear difference'


def summarize(args):
    outputs = paths.RESULTS / 'summary.csv', paths.RESULTS / 'summary_by_dataset.csv'
    if not args.overwrite:
        paths.refuse_existing(*outputs)
    rows = []
    for path in sorted(paths.RUNS.glob('*/ckpt*/comparison.json')):
        with open(path) as f:
            s = json.load(f)
        row = dict(dataset=s['dataset'], checkpoint=s['checkpoint_id'],
                   cached_threshold=s['threshold']['cached'], original_threshold=s['threshold']['original'],
                   cached_cost=s['cost']['cached'], original_cost=s['cost']['original'],
                   **{f'{k}_acc': v for k, v in s['accuracy'].items()})
        for name, c in s['comparisons'].items():
            row.update({f'{name}_pp': c['diff'], f'{name}_ci_low': c['ci_low'], f'{name}_ci_high': c['ci_high'],
                        f'{name}_verdict': verdict(c)})
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No comparison.json files under {paths.RUNS}. Run a checkpoint first.')
    order = {d: i for i, d in enumerate(paths.DATASETS)}
    table = pd.DataFrame(rows).sort_values(['dataset', 'checkpoint'],
                                           key=lambda c: c.map(order) if c.name == 'dataset' else c)
    paths.RESULTS.mkdir(exist_ok=True)
    table.to_csv(outputs[0], index=False)
    numeric = table.drop(columns=['checkpoint'] + [c for c in table if c.endswith('_verdict')])
    means = numeric.groupby('dataset', sort=False).mean().join(table.groupby('dataset').size().rename('checkpoints'))
    means.reset_index().to_csv(outputs[1], index=False)
    print(table[['dataset', 'checkpoint', 'cached_cost', 'cached_adaptive_acc', 'new_fixed_48_acc',
                 'cached_vs_new_fixed_48_pp', 'cached_vs_original_adaptive_pp']].round(2).to_string(index=False))
    print('Saved', outputs[0], 'and', outputs[1])


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', choices=paths.DATASETS)
    a.add_argument('--checkpoint-id', type=int)
    a.add_argument('--smoke', action='store_true', help='compare the smoke runs')
    a.add_argument('--summarize', action='store_true', help='build the results tables from finished runs')
    a.add_argument('--overwrite', action='store_true', help='with --summarize, replace existing tables')
    args = a.parse_args()
    if args.summarize:
        return summarize(args)
    if args.dataset is None or args.checkpoint_id is None:
        a.error('--dataset and --checkpoint-id are required unless --summarize is given')
    run(args)


if __name__ == '__main__':
    main()
