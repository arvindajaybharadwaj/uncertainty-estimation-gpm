"""Matched timing of the original methods and cached adaptive inference, for one checkpoint.

The procedure and methods of cross_dataset_validation/fixed_budget_timing.py
(time_methods.py plus fixed K=48), with cached adaptive inference added:
  - every method runs over all test nodes, in a shuffled order each round
  - speedups against fixed K=128 are computed within a round, then pooled
  - nested_four_stage is the original rule, with the threshold of the original
    adaptive_decisions.npz; cached_four_stage is live_cached.py's rule, with the
    threshold of this folder's decisions.npz, so nothing is tuned here
  - the timer counts sampling, inference, transfers and loop overhead, not
    loading or the warm-up, matching final_inference.py
Batch size defaults to 32, the size used for every dataset in the report.
Writes <folder>/timing.json, a new file next to the other encoder-caching
outputs, and refuses to overwrite it. Nothing in cross_dataset_validation is
written to. Close other GPU programs while it runs, and keep the laptop plugged in.

Example, from the repository root:
    python encoder_caching/timing_cached.py --dataset wikics --checkpoint-id 1
"""
import argparse
import json
import sys
import time

import numpy as np

import paths
from cached_gpm import load
from layout import STEPS

sys.path.insert(0, str(paths.CROSS))
sys.path.insert(0, str(paths.EVAL_DIR))
from fixed_budget_timing import methods_with_48  # noqa: E402
from matched_timing import round_order, summarise  # noqa: E402


def decisions_threshold(path, digest, expected=None):
    with np.load(path, allow_pickle=False) as z:
        if not bool(z['nested']) or str(z['rule']) != 'four':
            raise ValueError(f'{path}: expected decisions from the nested four-stage rule.')
        if str(z['checkpoint_sha256']) != digest:
            raise ValueError(f'{path} was made with a different checkpoint.')
        if expected is not None and not np.array_equal(z['node_ids'], expected):
            raise ValueError(f'{path} has different test nodes.')
        return float(z['threshold']), z['node_ids']


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=paths.DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--rounds', type=int, default=3)
    a.add_argument('--repeats', type=int, default=10)
    a.add_argument('--batch-size', type=int, default=32)
    a.add_argument('--seed', type=int, default=paths.TIMING_SEED)
    args = a.parse_args()

    checkpoint, digest = paths.verified_checkpoint(args.dataset, args.checkpoint_id)
    out = paths.folder(args.dataset, args.checkpoint_id)
    output = out / 'timing.json'
    paths.refuse_existing(output)
    original, expected = decisions_threshold(
        paths.original_folder(args.dataset, args.checkpoint_id) / 'adaptive_decisions.npz', digest)
    cached, _ = decisions_threshold(out / 'decisions.npz', digest, expected)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU found. Timing must run on the GPU.')
    gpm, ids, set_sampling_seed = load(args.dataset, checkpoint, torch.device('cuda:0'))
    nodes = ids[1]
    if not np.array_equal(nodes.numpy(), expected):
        raise ValueError('Test nodes differ from the decisions file, so the split has changed.')
    labels = gpm.graph.y.cpu()[nodes].numpy().reshape(-1)
    sync = torch.cuda.synchronize
    sample, predict = gpm.sample, gpm.predict

    # fixed_batch and adaptive_batch are the same as in time_methods.py.
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

    def cached_batch(batch, method):
        _, answer, cost, _, _ = gpm.adaptive(batch, method['steps'], method['threshold'])
        return answer, cost

    step_of = dict(fixed=fixed_batch, repeated=fixed_batch, adaptive=adaptive_batch, cached=cached_batch)

    def run_method(method, round_index):
        answers = np.zeros((args.repeats, len(nodes)), dtype=np.int64)
        costs = np.zeros_like(answers)
        step = step_of[method['kind']]
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

    methods = methods_with_48(original) + [dict(name='cached_four_stage', kind='cached', steps=STEPS,
                                                threshold=cached)]
    runs = []
    out.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        predict(nodes[:args.batch_size], sample(nodes[:args.batch_size], 128))  # warm-up, not counted
        sync()
        for round_index in range(args.rounds):
            for method in round_order(methods, round_index, args.seed):
                runs.append(run_method(method, round_index))
                print({k: round(v, 4) if isinstance(v, float) else v for k, v in runs[-1].items()}, flush=True)
            result = dict(dataset=args.dataset, checkpoint_id=args.checkpoint_id,
                          threshold=dict(original=original, cached=cached), rounds=round_index + 1,
                          repeats=args.repeats, batch_size=args.batch_size, nodes=len(nodes), **paths.environment(),
                          summary=summarise(runs, len(nodes), args.repeats), runs=runs)
            with open(output, 'w') as f:  # this run's own file, saved every round in case it is interrupted
                json.dump(result, f, indent=2)

    print('\nmethod              ms/node  speedup vs fixed_128     accuracy  passes  peak MB')
    for row in sorted(result['summary'], key=lambda r: r['ms_per_node']):
        print(f"{row['method']:<18} {row['ms_per_node']:8.3f}  {row['speedup_vs_fixed_128']:5.2f} "
              f"({row['speedup_min']:.2f}-{row['speedup_max']:.2f})    {100 * row['accuracy']:6.2f}%  "
              f"{row['pattern_passes']:6.1f}  {row['peak_gpu_mb']:7.0f}")
    print('Saved', output)


if __name__ == '__main__':
    main()
