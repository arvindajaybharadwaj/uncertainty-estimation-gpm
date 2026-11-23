"""Measure fixed K=32 and K=48 directly, and compare them with adaptive inference.

The paper's equal-cost baseline is computed: a mix of fixed K=32 and K=64,
interpolated to the adaptive cost. Accuracy flattens as K grows, so a measured
uniform budget at the same cost may beat that straight line. This script runs
fixed K=48 (about the adaptive cost) and fixed K=32 (slightly cheaper).

For one checkpoint it:
  1. Redraws the walks of final_inference.py's prediction step: same split,
     node order, batch size and seeds (71000 + r), one 128-walk pool per batch.
     K=32 and K=48 are the first 32 and 48 walks of that pool, so they share
     walks with the stored K=16, 32, 64 and 128 predictions.
  2. Checks that the recomputed K=32 matches final_predictions.npz, and stops
     if it does not (different walks or environment).
  3. Compares on test nodes, with evaluate.py's paired node bootstrap:
       adaptive (live)   - fixed K=48, fixed K=32, fixed K=128
       adaptive (replay) - fixed K=48, the same rule on the same walks
       fixed K=48        - the computed K=32/64 mix at cost 48
Writes runs/<dataset>/ckpt<id>/fixed_budgets.npz and fixed_budgets.json and
refuses to overwrite them. --summarize then builds
results/fixed_budget_baseline.csv and fixed_budget_baseline_by_dataset.csv.

Example, from the repository root:
    python cross_dataset_validation/fixed_budget_baseline.py --dataset wikics --checkpoint-id 1
    python cross_dataset_validation/fixed_budget_baseline.py --summarize
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from common import (DATASETS, EVAL_DIR, HERE, RUNS, checkpoint_file, environment, find_checkpoint,
                    run_folder, sha256, use_gpm)
from run_checkpoint import PREDICTION_SEED

sys.path.insert(0, str(EVAL_DIR))
import evaluate as ev  # noqa: E402

BUDGETS = (32, 48)
POOL = 128
BATCH_SIZE = {'cora_full': 32}  # batch size each dataset's predictions were made with; 256 otherwise
SMOKE_NODES = 32                # run_checkpoint.py --smoke
BOOTSTRAP = 2000
SEED = 20261105 + 9             # the seed evaluate.py uses for live comparisons
CHECK_TOLERANCE = 1e-4


def predict_fixed(dataset, checkpoint, node_ids, repeats, batch_size, smoke):
    """Class probabilities at K=32 and K=48, shape [budget, repetition, node, class]."""
    use_gpm()
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    # Setup as in GPM/final_inference.py, so the walks are the same.
    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = load_baseline_params(dataset)
    params['device'] = device
    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params.get('num_tasks') or int(graph.y.max()) + 1
    ids = [mask2idx(splits[0][s]).cpu() for s in ('val', 'test')]
    if smoke:
        ids = [x[:SMOKE_NODES] for x in ids]
    nodes = torch.cat(ids)
    if not np.array_equal(nodes.numpy(), node_ids):
        raise ValueError('Node order differs from final_predictions.npz, so the split has changed.')
    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    def sample(batch, count):
        return get_patterns(graph, dict(params, pre_sample_pattern_num=count), nodes=batch)

    def predict(batch, pool):
        run = dict(params, pattern_set={'pattern': pool[0].contiguous(), 'eid': pool[1].contiguous()})
        logits = model(graph, torch.arange(len(batch)), run, mode='eval')[0]
        # The model drops the batch dimension for a single node; keep it.
        return logits.softmax(-1).cpu().numpy().reshape(len(batch), -1)

    out = np.empty((len(BUDGETS), repeats, len(nodes), params['output_dim']), dtype=np.float32)
    start = time.perf_counter()
    with torch.inference_mode():
        for r in range(repeats):
            set_sampling_seed(PREDICTION_SEED + r)
            for offset in range(0, len(nodes), batch_size):
                batch = nodes[offset:offset + batch_size]
                pool = sample(batch, POOL)  # one pool per batch, as in the prediction step
                for i, k in enumerate(BUDGETS):
                    out[i, r, offset:offset + len(batch)] = predict(batch, tuple(x[:k] for x in pool))
            print(f'Repetition {r + 1}/{repeats} complete', flush=True)
    return out, time.perf_counter() - start


def check_reproduction(recomputed, stored):
    """The recomputed K=32 must match the stored K=32, or the walks differ."""
    diff = float(np.abs(recomputed - stored).max())
    agreement = float((recomputed.argmax(-1) == stored.argmax(-1)).mean())
    if diff > CHECK_TOLERANCE:
        raise ValueError(f'Recomputed K=32 differs from final_predictions.npz (max difference {diff:.2e}, '
                         f'{100 * agreement:.2f}% same class). The walks or environment have changed.')
    return dict(max_abs_difference=diff, same_prediction_pct=100 * agreement)


def compare(data, fixed, decisions):
    """Per-checkpoint accuracies and paired comparisons on test nodes. Accuracies in %, gaps in points."""
    costs = ev.load_costs(data['ks'])[0]
    live, live_node, positions = ev.load_adaptive_decisions(decisions, data['y'], data['node_ids'], costs)
    y = data['y'][positions]
    node = {f'fixed_{k}': ev.correct(fixed[i][:, positions], y).mean(0) for i, k in enumerate(BUDGETS)}
    for k in (64, 128):
        node[f'fixed_{k}'] = ev.fixed(data, k, positions)
    node['adaptive_live'] = live_node
    scores = ev.adaptive_scores(data, positions, 'confidence')
    node['adaptive_replay'], step = ev.adaptive(data, positions, scores, live['threshold'])
    replay_cost = float(ev.stop_costs(costs)[0][step].mean())
    node['mix_32_64_at_48'] = .5 * node['fixed_32'] + .5 * node['fixed_64']
    w = float(np.clip((live['rerun_cost'] - 32) / 32, 0, 1))
    node['mix_32_64_at_adaptive_cost'] = (1 - w) * node['fixed_32'] + w * node['fixed_64']

    pairs = {'adaptive_vs_fixed_48': ('adaptive_live', 'fixed_48'),
             'adaptive_vs_fixed_32': ('adaptive_live', 'fixed_32'),
             'adaptive_vs_fixed_128': ('adaptive_live', 'fixed_128'),
             'replay_vs_fixed_48': ('adaptive_replay', 'fixed_48'),
             'fixed_48_vs_mix_at_48': ('fixed_48', 'mix_32_64_at_48'),
             'adaptive_vs_mix_at_adaptive_cost': ('adaptive_live', 'mix_32_64_at_adaptive_cost')}
    comparisons = {}
    for name, (a, b) in pairs.items():
        d, low, high = ev.bootstrap_diff(node[a], node[b], BOOTSTRAP, SEED)
        comparisons[name] = dict(diff=100 * d, ci_low=100 * low, ci_high=100 * high)
    return dict(test_nodes=len(positions), threshold=live['threshold'],
                accuracy={name: 100 * float(v.mean()) for name, v in node.items()},
                cost=dict(fixed_32=32., fixed_48=48., adaptive_live=live['rerun_cost'],
                          adaptive_replay=replay_cost),
                comparisons=comparisons)


def run(args):
    row = find_checkpoint(args.dataset, args.checkpoint_id)
    checkpoint = checkpoint_file(row)
    if sha256(checkpoint) != row['sha256']:
        raise ValueError(f'{checkpoint} does not match the hash in checkpoints.csv.')
    folder = run_folder(args.dataset, args.checkpoint_id, args.smoke)
    outputs = folder / 'fixed_budgets.npz', folder / 'fixed_budgets.json'
    existing = [str(p) for p in outputs if p.exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite {", ".join(existing)}')
    data = ev.load_predictions(folder / 'final_predictions.npz')
    with np.load(folder / 'final_predictions.npz', allow_pickle=False) as z:
        if str(z['checkpoint_sha256']) != row['sha256'] or not bool(z['nested']):
            raise ValueError('final_predictions.npz is not a nested run of this checkpoint.')
        if int(z['seed']) != PREDICTION_SEED:
            raise ValueError(f"final_predictions.npz used seed {int(z['seed'])}, not {PREDICTION_SEED}.")
    batch_size = args.batch_size or BATCH_SIZE.get(args.dataset, 256)
    repeats = data['p'].shape[1]

    fixed, seconds = predict_fixed(args.dataset, checkpoint, data['node_ids'], repeats, batch_size, args.smoke)
    check = check_reproduction(fixed[BUDGETS.index(32)], data['p'][data['ks'].index(32)])
    result = compare(data, fixed, folder / 'adaptive_decisions.npz')

    test = data['node_split'] == 'test'
    np.savez_compressed(outputs[0], probabilities=fixed[:, :, test], k_values=np.array(BUDGETS),
                        labels=data['y'][test], node_ids=data['node_ids'][test], dataset=args.dataset,
                        checkpoint_sha256=row['sha256'], seed=PREDICTION_SEED, batch_size=batch_size,
                        nested=True, smoke_test=args.smoke)
    summary = dict(dataset=args.dataset, checkpoint_id=args.checkpoint_id, batch_size=batch_size,
                   repetitions=repeats, seconds=seconds, reproduction_check=check, **result, **environment())
    with open(outputs[1], 'w') as f:
        json.dump(summary, f, indent=2)

    acc, cmp = result['accuracy'], result['comparisons']
    print(f"\nTest nodes: {result['test_nodes']:,}. K=32 reproduced: max difference {check['max_abs_difference']:.1e}")
    for name in ('fixed_32', 'fixed_48', 'fixed_64', 'fixed_128', 'adaptive_live', 'adaptive_replay'):
        print(f'  {name:<16} {acc[name]:6.2f}%')
    for name, c in cmp.items():
        print(f"  {name:<34} {c['diff']:+.2f}  ({c['ci_low']:+.2f}, {c['ci_high']:+.2f})")
    print('Saved', outputs[1])


def verdict(c):
    if c['ci_low'] > 0:
        return 'adaptive better'
    if c['ci_high'] < 0:
        return 'adaptive worse'
    return 'no clear difference'


def summarize(args):
    rows = []
    for path in sorted(RUNS.glob('*/ckpt*/fixed_budgets.json')):
        with open(path) as f:
            s = json.load(f)
        row = dict(dataset=s['dataset'], checkpoint=s['checkpoint_id'], batch_size=s['batch_size'],
                   adaptive_cost=s['cost']['adaptive_live'],
                   **{f'{k}_acc': v for k, v in s['accuracy'].items()})
        for name, c in s['comparisons'].items():
            row.update({f'{name}_pp': c['diff'], f'{name}_ci_low': c['ci_low'], f'{name}_ci_high': c['ci_high']})
        row['verdict_vs_fixed_48'] = verdict(s['comparisons']['adaptive_vs_fixed_48'])
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f'No fixed_budgets.json files under {RUNS}. Run a checkpoint first.')
    order = {d: i for i, d in enumerate(DATASETS)}
    table = pd.DataFrame(rows).sort_values(['dataset', 'checkpoint'], key=lambda c: c.map(order) if c.name == 'dataset' else c)
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    table.to_csv(out / 'fixed_budget_baseline.csv', index=False)
    numeric = table.drop(columns=['checkpoint', 'batch_size', 'verdict_vs_fixed_48']).groupby('dataset', sort=False)
    means = numeric.mean().join(table.groupby('dataset').size().rename('checkpoints'))
    means.reset_index().to_csv(out / 'fixed_budget_baseline_by_dataset.csv', index=False)
    print(table[['dataset', 'checkpoint', 'adaptive_cost', 'fixed_32_acc', 'fixed_48_acc', 'adaptive_live_acc',
                 'adaptive_vs_fixed_48_pp', 'verdict_vs_fixed_48']].round(2).to_string(index=False))
    print('Saved', out / 'fixed_budget_baseline.csv', 'and', out / 'fixed_budget_baseline_by_dataset.csv')


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', choices=DATASETS)
    a.add_argument('--checkpoint-id', type=int)
    a.add_argument('--batch-size', type=int,
                   help='must match the prediction run: 32 for cora_full, 256 otherwise (the default)')
    a.add_argument('--smoke', action='store_true', help='use runs_smoke/, made by run_checkpoint.py --smoke')
    a.add_argument('--summarize', action='store_true', help='build the results tables from finished runs')
    args = a.parse_args()
    if args.summarize:
        return summarize(args)
    if args.dataset is None or args.checkpoint_id is None:
        a.error('--dataset and --checkpoint-id are required unless --summarize is given')
    run(args)


if __name__ == '__main__':
    main()
