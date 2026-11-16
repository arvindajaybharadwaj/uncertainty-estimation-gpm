"""Time fixed, repeated and adaptive inference in one session on one GPU.

Adaptive live times cannot be compared with fixed K=128 unless both run on the
same hardware under the same conditions. This script runs every method over
all test nodes, repeats that in shuffled rounds, and reports speedups measured
within each round. Thresholds are read from the frozen decision files, so no
tuning happens here.

The timer matches final_inference.py: sampling, inference, transfers and loop
overhead are counted; loading and warm-up are not.
"""
import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from measure_timing import install_dgl_stub_if_missing

BASELINE = 'fixed_128'


def method_table(thresholds):
    """thresholds: nested_four, fresh_four, fresh_two."""
    return [dict(name='fixed_16', kind='fixed', k=16), dict(name='fixed_32', kind='fixed', k=32),
            dict(name='fixed_64', kind='fixed', k=64), dict(name=BASELINE, kind='fixed', k=128),
            dict(name='repeated_4x16', kind='repeated', k=16, m=4), dict(name='repeated_8x16', kind='repeated', k=16, m=8),
            dict(name='nested_four_stage', kind='adaptive', nested=True, steps=(16, 32, 64, 128), threshold=thresholds['nested_four']),
            dict(name='fresh_four_stage', kind='adaptive', nested=False, steps=(16, 32, 64, 128), threshold=thresholds['fresh_four']),
            dict(name='fresh_two_stage', kind='adaptive', nested=False, steps=(16, 128), threshold=thresholds['fresh_two'])]


def round_order(methods, round_index, seed):
    order = np.random.default_rng(seed + round_index).permutation(len(methods))
    return [methods[i] for i in order]


def summarise(runs, nodes, repeats):
    """Reduce per-round runs. Speedups are computed within a round, then pooled."""
    by_method = defaultdict(dict)
    for run in runs:
        by_method[run['method']][run['round']] = run
    if BASELINE not in by_method:
        raise ValueError(f'{BASELINE} is required as the timing baseline.')
    base = by_method[BASELINE]
    rows = []
    for name, rounds in by_method.items():
        missing = set(base) - set(rounds)
        if missing:
            raise ValueError(f'{name} is missing rounds {sorted(missing)}.')
        seconds = [rounds[r]['seconds'] for r in sorted(rounds)]
        speedups = [base[r]['seconds'] / rounds[r]['seconds'] for r in sorted(rounds)]
        rows.append(dict(method=name, rounds=len(rounds), median_seconds=statistics.median(seconds),
                         min_seconds=min(seconds), max_seconds=max(seconds),
                         ms_per_node=statistics.median(seconds) / (nodes * repeats) * 1000,
                         speedup_vs_fixed_128=statistics.median(speedups), speedup_min=min(speedups),
                         speedup_max=max(speedups), accuracy=statistics.mean(rounds[r]['accuracy'] for r in rounds),
                         pattern_passes=statistics.mean(rounds[r]['pattern_passes'] for r in rounds),
                         peak_gpu_mb=max((rounds[r]['peak_mb'] or 0) for r in rounds)))
    return rows


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpm-dir', required=True)
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--handoff', required=True, help='folder holding nested/ and fresh/')
    a.add_argument('--rounds', type=int, default=3)
    a.add_argument('--repeats', type=int, default=50)
    a.add_argument('--batch-size', type=int, default=256)
    a.add_argument('--seed', type=int, default=271000)
    a.add_argument('--output', required=True)
    args = a.parse_args()

    handoff = Path(args.handoff)
    thresholds = {name: float(np.load(handoff / path)['threshold']) for name, path in (
        ('nested_four', 'nested/adaptive_decisions.npz'), ('fresh_four', 'fresh/adaptive_decisions.npz'),
        ('fresh_two', 'fresh/adaptive_decisions_two_stage.npz'))}
    expected = np.load(handoff / 'nested/adaptive_decisions.npz')['node_ids']

    sys.path.insert(0, str(Path(args.gpm_dir).resolve()))
    install_dgl_stub_if_missing()
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. Set the Colab runtime to GPU and start again.')
    device = torch.device('cuda:0')
    seed_everything(42)
    params = load_baseline_params('computers')
    params['device'] = device
    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params.get('num_tasks') or int(graph.y.max()) + 1
    nodes = mask2idx(splits[0]['test']).cpu()
    if not np.array_equal(nodes.numpy(), expected):
        raise ValueError('Test nodes differ from the handoff files, so the split has changed.')
    labels = graph.y.cpu()[nodes].numpy().reshape(-1)
    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    sync = torch.cuda.synchronize

    def sample(batch, count):
        return get_patterns(graph, dict(params, pre_sample_pattern_num=count), nodes=batch)

    def predict(batch, pool):
        run = dict(params, pattern_set={'pattern': pool[0].contiguous(), 'eid': pool[1].contiguous()})
        return model(graph, torch.arange(len(batch)), run, mode='eval')[0].softmax(-1)

    def fixed_batch(batch, method):
        probs = sum(predict(batch, sample(batch, method['k'])) for _ in range(method.get('m', 1)))
        cost = method['k'] * method.get('m', 1)
        return probs.argmax(-1).cpu().numpy(), np.full(len(batch), cost)

    def adaptive_batch(batch, method):
        n = len(batch)
        answer, cost = np.full(n, -1), np.zeros(n, dtype=np.int64)
        active, pool, previous = np.arange(n), None, 0
        for k in method['steps']:
            if not len(active):
                break
            active_nodes = batch[torch.from_numpy(active)]
            if method['nested']:
                extra = sample(active_nodes, k - previous)
                pool = extra if pool is None else tuple(torch.cat((old, new), 0) for old, new in zip(pool, extra))
                current = pool
            else:
                current = sample(active_nodes, k)
            probs = predict(active_nodes, current).cpu().numpy()
            cost[active] += k
            stop = (probs.max(-1) >= method['threshold']) | (k == method['steps'][-1])
            answer[active[stop]] = probs[stop].argmax(-1)
            active = active[~stop]
            if pool is not None:
                keep = torch.from_numpy(np.flatnonzero(~stop))
                pool = tuple(x[:, keep].contiguous() for x in pool)
            previous = k
        return answer, cost

    def run_method(method, round_index):
        answers = np.zeros((args.repeats, len(nodes)), dtype=np.int64)
        costs = np.zeros_like(answers)
        step = adaptive_batch if method['kind'] == 'adaptive' else fixed_batch
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        for r in range(args.repeats):
            set_sampling_seed(args.seed + 10000 * round_index + r)
            for offset in range(0, len(nodes), args.batch_size):
                batch = nodes[offset:offset + args.batch_size]
                answers[r, offset:offset + len(batch)], costs[r, offset:offset + len(batch)] = step(batch, method)
        sync()
        seconds = time.perf_counter() - start
        if (answers < 0).any():
            raise ValueError(f'{method["name"]} left nodes without a prediction.')
        return dict(method=method['name'], round=round_index, seconds=seconds,
                    accuracy=float((answers == labels).mean()), pattern_passes=float(costs.mean()),
                    peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)

    methods = method_table(thresholds)
    runs = []
    with torch.inference_mode():
        predict(nodes[:args.batch_size], sample(nodes[:args.batch_size], 128))  # warm-up, not counted
        sync()
        for round_index in range(args.rounds):
            for method in round_order(methods, round_index, args.seed):
                runs.append(run_method(method, round_index))
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in runs[-1].items()}, flush=True)
            result = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, rounds=round_index + 1,
                          repeats=args.repeats, batch_size=args.batch_size, nodes=len(nodes),
                          summary=summarise(runs, len(nodes), args.repeats), runs=runs)
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:  # saved every round in case Colab disconnects
                json.dump(result, f, indent=2)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
