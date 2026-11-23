"""Matched timing with fixed K=48 added, for one checkpoint.

The same procedure and methods as time_methods.py, plus fixed K=48, so the
measured uniform budget at about the adaptive cost is timed in the same session
as fixed K=128 and adaptive inference:
  - every method runs over all test nodes, in a shuffled order each round
  - speedups against fixed K=128 are computed within a round, then pooled
  - the adaptive threshold is read from the live decisions file, so nothing is tuned here
  - the timer counts sampling, inference, transfers and loop overhead, not
    loading or the warm-up, matching final_inference.py
Batch size defaults to 32, the size used for every dataset in the report.
Writes runs/<dataset>/ckpt<id>/timing_fixed_budgets.json, replacing any
earlier result for that checkpoint; timing.json is not touched.
Close other GPU programs while it runs, and keep the laptop plugged in.

Example, from the repository root:
    python cross_dataset_validation/fixed_budget_timing.py --dataset wikics --checkpoint-id 1
"""
import argparse
import json
import sys
import time

import numpy as np

from common import (EVAL_DIR, checkpoint_file, environment, find_checkpoint, run_folder, sha256, use_gpm)
from time_methods import method_table

sys.path.insert(0, str(EVAL_DIR))
from matched_timing import round_order, summarise  # noqa: E402

EXTRA = dict(name='fixed_48', kind='fixed', k=48)


def methods_with_48(threshold):
    methods = method_table(threshold)
    at = next(i for i, m in enumerate(methods) if m['name'] == 'fixed_64')
    return methods[:at] + [EXTRA] + methods[at:]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--rounds', type=int, default=3)
    a.add_argument('--repeats', type=int, default=10)
    a.add_argument('--batch-size', type=int, default=32)
    a.add_argument('--seed', type=int, default=271000)  # time_methods.py default
    args = a.parse_args()

    row = find_checkpoint(args.dataset, args.checkpoint_id)
    checkpoint = checkpoint_file(row)
    folder = run_folder(args.dataset, args.checkpoint_id)
    output = folder / 'timing_fixed_budgets.json'
    if output.exists():
        print('Overwriting', output, flush=True)
    with np.load(folder / 'adaptive_decisions.npz', allow_pickle=False) as z:
        threshold, expected = float(z['threshold']), z['node_ids']
        if not bool(z['nested']) or str(z['rule']) != 'four':
            raise ValueError('Expected decisions from the nested four-stage rule.')
        if str(z['checkpoint_sha256']) != sha256(checkpoint):
            raise ValueError('The decisions file was made with a different checkpoint.')

    use_gpm()
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. Timing must run on the GPU.')
    device = torch.device('cuda:0')
    seed_everything(42)
    params = load_baseline_params(args.dataset)
    params['device'] = device
    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params.get('num_tasks') or int(graph.y.max()) + 1
    nodes = mask2idx(splits[0]['test']).cpu()
    if not np.array_equal(nodes.numpy(), expected):
        raise ValueError('Test nodes differ from the decisions file, so the split has changed.')
    labels = graph.y.cpu()[nodes].numpy().reshape(-1)
    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    sync = torch.cuda.synchronize

    # sample, predict and the two batch steps are the same as in time_methods.py.
    def sample(batch, count):
        return get_patterns(graph, dict(params, pre_sample_pattern_num=count), nodes=batch)

    def predict(batch, pool):
        run = dict(params, pattern_set={'pattern': pool[0].contiguous(), 'eid': pool[1].contiguous()})
        # The model drops the batch dimension for a single node; keep it so stopping masks stay 1-D.
        return model(graph, torch.arange(len(batch)), run, mode='eval')[0].softmax(-1).reshape(len(batch), -1)

    def fixed_batch(batch, method):
        probs = sum(predict(batch, sample(batch, method['k'])) for _ in range(method.get('m', 1)))
        return probs.argmax(-1).cpu().numpy(), np.full(len(batch), method['k'] * method.get('m', 1))

    def adaptive_batch(batch, method):
        n = len(batch)
        answer, cost = np.full(n, -1), np.zeros(n, dtype=np.int64)
        active, pool, previous = np.arange(n), None, 0
        for k in method['steps']:
            if not len(active):
                break
            active_nodes = batch[torch.from_numpy(active)]
            extra = sample(active_nodes, k - previous)
            pool = extra if pool is None else tuple(torch.cat((old, new), 0) for old, new in zip(pool, extra))
            probs = predict(active_nodes, pool).cpu().numpy()
            cost[active] += k
            stop = (probs.max(-1) >= method['threshold']) | (k == method['steps'][-1])
            answer[active[stop]] = probs[stop].argmax(-1)
            active = active[~stop]
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

    methods = methods_with_48(threshold)
    runs = []
    with torch.inference_mode():
        predict(nodes[:args.batch_size], sample(nodes[:args.batch_size], 128))  # warm-up, not counted
        sync()
        for round_index in range(args.rounds):
            for method in round_order(methods, round_index, args.seed):
                runs.append(run_method(method, round_index))
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in runs[-1].items()}, flush=True)
            result = dict(dataset=args.dataset, checkpoint_id=args.checkpoint_id, threshold=threshold,
                          rounds=round_index + 1, repeats=args.repeats, batch_size=args.batch_size,
                          nodes=len(nodes), **environment(), summary=summarise(runs, len(nodes), args.repeats),
                          runs=runs)
            with open(output, 'w') as f:  # saved every round in case the run is interrupted
                json.dump(result, f, indent=2)

    print('\nmethod              ms/node  speedup vs fixed_128     accuracy  passes  peak MB')
    for row in sorted(result['summary'], key=lambda r: r['ms_per_node']):
        print(f"{row['method']:<18} {row['ms_per_node']:8.3f}  {row['speedup_vs_fixed_128']:5.2f} "
              f"({row['speedup_min']:.2f}-{row['speedup_max']:.2f})    {100 * row['accuracy']:6.2f}%  "
              f"{row['pattern_passes']:6.1f}  {row['peak_gpu_mb']:7.0f}")
    print('Saved', output)


if __name__ == '__main__':
    main()
