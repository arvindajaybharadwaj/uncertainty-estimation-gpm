"""Check whether the batch size changes the measured speedups, for one checkpoint.

CoraFull has to run at batch size 32 to fit in GPU memory, while Computers and
WikiCS were timed at 256. This times the same methods as time_methods.py at
several batch sizes in one session, so the speedups can be compared directly:
  - every batch size is re-timed here, including 256, rather than compared with
    an existing timing.json measured in a different session
  - within each round, the batch sizes and the methods run in a shuffled order,
    so GPU warming or throttling affects all of them alike
  - speedups against fixed K=128 are computed within a round and a batch size,
    then pooled over rounds, as in matched_timing.py
  - each batch size gets its own warm-up, which is not counted
  - the threshold comes from the existing live decisions file; nothing is tuned
It reads runs/<dataset>/ckpt<id>/ and writes only
results/batch_size_timing/<dataset>_ckpt<id>.json, refusing to overwrite it.
Close other GPU programs while it runs, and keep the laptop plugged in.

Example, from the repository root:
    python cross_dataset_validation/batch_size_timing.py --dataset wikics --checkpoint-id 1 --batch-sizes 32 256
"""
import argparse
import json
import sys
import time

import numpy as np

from common import HERE, checkpoint_file, environment, find_checkpoint, run_folder, sha256, use_gpm
from time_methods import method_table

from matched_timing import round_order, summarise  # on the path through time_methods

OUTPUT_DIR = HERE / 'results' / 'batch_size_timing'


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--batch-sizes', type=int, nargs='+', default=[32, 256])
    a.add_argument('--methods', nargs='+',
                   help='subset of the time_methods.py methods; fixed_128 is always included. Default: all')
    a.add_argument('--rounds', type=int, default=3)
    a.add_argument('--repeats', type=int, default=10)
    a.add_argument('--seed', type=int, default=271000)  # time_methods.py default
    args = a.parse_args()
    if len(set(args.batch_sizes)) < 2 or min(args.batch_sizes) < 1:
        raise ValueError('Give at least two different positive batch sizes.')

    row = find_checkpoint(args.dataset, args.checkpoint_id)
    checkpoint = checkpoint_file(row)
    folder = run_folder(args.dataset, args.checkpoint_id)
    output = OUTPUT_DIR / f'{args.dataset}_ckpt{args.checkpoint_id}.json'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    with np.load(folder / 'adaptive_decisions.npz', allow_pickle=False) as z:
        threshold, expected = float(z['threshold']), z['node_ids']
        if not bool(z['nested']) or str(z['rule']) != 'four':
            raise ValueError('Expected decisions from the nested four-stage rule.')
        if str(z['checkpoint_sha256']) != sha256(checkpoint):
            raise ValueError('The decisions file was made with a different checkpoint.')

    methods = method_table(threshold)
    if args.methods:
        unknown = set(args.methods) - {m['name'] for m in methods}
        if unknown:
            raise ValueError(f'Unknown methods: {sorted(unknown)}')
        methods = [m for m in methods if m['name'] in set(args.methods) | {'fixed_128'}]

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

    def run_method(method, batch_size, round_index):
        answers = np.zeros((args.repeats, len(nodes)), dtype=np.int64)
        costs = np.zeros_like(answers)
        step = adaptive_batch if method['kind'] == 'adaptive' else fixed_batch
        torch.cuda.reset_peak_memory_stats()
        sync()
        start = time.perf_counter()
        for r in range(args.repeats):
            # The same seeds at every batch size; the walks still differ because each batch draws from the stream.
            set_sampling_seed(args.seed + 10000 * round_index + r)
            for offset in range(0, len(nodes), batch_size):
                batch = nodes[offset:offset + batch_size]
                answers[r, offset:offset + len(batch)], costs[r, offset:offset + len(batch)] = step(batch, method)
        sync()
        seconds = time.perf_counter() - start
        if (answers < 0).any():
            raise ValueError(f'{method["name"]} left nodes without a prediction.')
        return dict(method=method['name'], batch_size=batch_size, round=round_index, seconds=seconds,
                    accuracy=float((answers == labels).mean()), pattern_passes=float(costs.mean()),
                    peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)

    def summary(runs):
        rows = []
        for batch_size in args.batch_sizes:
            for item in summarise([r for r in runs if r['batch_size'] == batch_size], len(nodes), args.repeats):
                rows.append(dict(batch_size=batch_size, **item))
        return rows

    runs = []
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for batch_size in args.batch_sizes:  # warm-up for each batch size, not counted
            predict(nodes[:batch_size], sample(nodes[:batch_size], 128))
        sync()
        for round_index in range(args.rounds):
            order = np.random.default_rng(args.seed + 500 + round_index).permutation(len(args.batch_sizes))
            for batch_size in (args.batch_sizes[i] for i in order):
                for method in round_order(methods, round_index, args.seed):
                    runs.append(run_method(method, batch_size, round_index))
                    print({k: round(v, 4) if isinstance(v, float) else v for k, v in runs[-1].items()}, flush=True)
            result = dict(dataset=args.dataset, checkpoint_id=args.checkpoint_id, threshold=threshold,
                          rounds=round_index + 1, repeats=args.repeats, batch_sizes=args.batch_sizes,
                          nodes=len(nodes), **environment(), summary=summary(runs), runs=runs)
            with open(output, 'w') as f:  # saved every round in case the run is interrupted
                json.dump(result, f, indent=2)

    print('\nbatch  method              ms/node  speedup vs fixed_128  accuracy')
    for row in result['summary']:
        print(f"{row['batch_size']:>5}  {row['method']:<18} {row['ms_per_node']:>8.3f}  "
              f"{row['speedup_vs_fixed_128']:>6.2f} ({row['speedup_min']:.2f}-{row['speedup_max']:.2f})"
              f"        {100 * row['accuracy']:.2f}")
    print('Saved', output)


if __name__ == '__main__':
    main()
