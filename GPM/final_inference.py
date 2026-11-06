"""GPM final predictions and genuinely live adaptive inference (no encoding cache)."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

KS = (16, 32, 64, 128)


def choose_threshold(probabilities, target, steps):
    """Choose using validation confidence only, counting every encoding pass."""
    scores = probabilities.max(-1)
    costs = np.cumsum(steps)
    candidates = np.unique(np.r_[0., np.quantile(scores, np.linspace(0, 1, 400)), 1.000001])
    chosen = 0.
    for threshold in candidates:
        passed = scores >= threshold
        passed[-1] = True
        cost = costs[passed.argmax(0)].mean()
        if cost <= target:
            chosen = float(threshold)
    return chosen


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--gpm-dir', required=True)
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--dataset', default='computers')
    a.add_argument('--stage', choices=['predictions', 'live'], required=True)
    a.add_argument('--sampling', choices=['nested', 'fresh'], required=True)
    a.add_argument('--rule', choices=['four', 'two'], default='four')
    a.add_argument('--calibration', help='Matching predictions file; thresholds use val nodes only')
    a.add_argument('--target', type=float, default=48.)
    a.add_argument('--repeats', type=int, default=50)
    a.add_argument('--batch-size', type=int, default=256)
    a.add_argument('--max-nodes-per-split', type=int, help='Smoke test only; never use for final reporting')
    a.add_argument('--seed', type=int, default=71000)
    a.add_argument('--output', required=True)
    args = a.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    if args.repeats < 1 or args.batch_size < 1 or (args.max_nodes_per_split is not None and args.max_nodes_per_split < 1):
        raise ValueError('Counts must be positive')
    if args.rule == 'two' and args.sampling != 'fresh':
        raise ValueError('The two-stage comparison requires fresh K=128 walks')
    sys.path.insert(0, str(Path(args.gpm_dir).resolve()))
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    seed_everything(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = load_baseline_params(args.dataset)
    params['device'] = device
    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params.get('num_tasks') or int(graph.y.max()) + 1
    ids = [mask2idx(splits[0][s]).cpu() for s in ('val', 'test')]
    if args.max_nodes_per_split:
        ids = [x[:args.max_nodes_per_split] for x in ids]
    nodes = torch.cat(ids)
    node_ids = nodes.numpy()
    node_split = np.array(['val'] * len(ids[0]) + ['test'] * len(ids[1]))
    labels = graph.y.cpu()[nodes].numpy().reshape(-1)
    if not len(ids[0]) or not len(ids[1]) or len(np.unique(node_ids)) != len(nodes):
        raise ValueError('Need nonempty, disjoint validation and test nodes')
    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def sample(batch, count):
        return get_patterns(graph, dict(params, pre_sample_pattern_num=count), nodes=batch)

    def predict(batch, pool):
        run = dict(params, pattern_set={'pattern': pool[0].contiguous(), 'eid': pool[1].contiguous()})
        logits = model(graph, torch.arange(len(batch)), run, mode='eval')[0]
        return logits.softmax(-1).cpu().numpy()

    threshold = None
    steps = KS if args.rule == 'four' else (16, 128)
    if args.stage == 'live':
        if not args.calibration:
            raise ValueError('Live inference requires --calibration')
        with np.load(args.calibration, allow_pickle=False) as z:
            if bool(z['nested']) != (args.sampling == 'nested'):
                raise ValueError('Calibration sampling mode does not match live mode')
            if str(z['dataset']) != args.dataset or str(z['checkpoint_sha256']) != checkpoint_hash(args.checkpoint):
                raise ValueError('Calibration dataset or checkpoint mismatch')
            cal = z['node_split'] == 'val'
            if not np.array_equal(z['node_ids'][cal], ids[0].numpy()):
                raise ValueError('Validation IDs differ from calibration file')
            p = z['probabilities'][[list(z['k_values']).index(k) for k in steps]][:, :, cal]
            if not np.isfinite(p).all():
                raise ValueError('Calibration probabilities are incomplete')
            threshold = choose_threshold(p, args.target, steps)
        print('Frozen validation threshold:', threshold, flush=True)
        nodes = ids[1]
        node_ids = nodes.numpy()

    probabilities = np.empty((4, args.repeats, len(nodes), params['output_dim']), dtype=np.float32) if args.stage == 'predictions' else None
    final_k = np.zeros((args.repeats, len(nodes)), dtype=np.int64)
    predicted = np.full_like(final_k, -1)
    pattern_passes = np.zeros_like(final_k)
    sampled = np.zeros_like(final_k)
    with torch.inference_mode():
        # Warm up once; excluded from recorded inference time.
        predict(nodes[:min(len(nodes), args.batch_size)], sample(nodes[:min(len(nodes), args.batch_size)], 16))
        sync()
        start = time.perf_counter()
        for r in range(args.repeats):
            set_sampling_seed(args.seed + r)
            for offset in range(0, len(nodes), args.batch_size):
                batch = nodes[offset:offset + args.batch_size]
                if args.stage == 'predictions':
                    pool = sample(batch, 128) if args.sampling == 'nested' else None
                    for i, k in enumerate(KS):
                        current = tuple(x[:k] for x in pool) if pool is not None else sample(batch, k)
                        probabilities[i, r, offset:offset + len(batch)] = predict(batch, current)
                else:
                    active = np.arange(len(batch))
                    pool = None
                    previous = 0
                    for k in steps:
                        if not len(active):
                            break
                        active_nodes = batch[torch.from_numpy(active)]
                        if args.sampling == 'nested':
                            extra = sample(active_nodes, k - previous)
                            pool = extra if pool is None else tuple(torch.cat((old, new), 0) for old, new in zip(pool, extra))
                            current = pool
                            sampled[r, offset + active] += k - previous
                        else:
                            current = sample(active_nodes, k)
                            sampled[r, offset + active] += k
                        probs = predict(active_nodes, current)
                        pattern_passes[r, offset + active] += k
                        stop = (probs.max(-1) >= threshold) | (k == steps[-1])
                        positions = offset + active[stop]
                        final_k[r, positions] = k
                        predicted[r, positions] = probs[stop].argmax(-1)
                        active = active[~stop]
                        if pool is not None:
                            keep = torch.from_numpy(np.flatnonzero(~stop))
                            pool = tuple(x[:, keep].contiguous() for x in pool)
                        previous = k
            print(f'Repetition {r + 1}/{args.repeats} complete', flush=True)
        sync()
        seconds = time.perf_counter() - start
    metadata = dict(dataset=args.dataset, checkpoint_sha256=checkpoint_hash(args.checkpoint),
                    nested=args.sampling == 'nested', encoding_reuse=False,
                    smoke_test=args.max_nodes_per_split is not None, seed=args.seed,
                    seconds_total=seconds, timing_scope='all repetitions; sampling, inference, transfers and loop overhead; excludes loading, warmup and file saving')
    output.parent.mkdir(parents=True, exist_ok=True)
    if probabilities is not None:
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(-1), 1, atol=1e-4):
            raise ValueError('Invalid probabilities')
        np.savez_compressed(output, probabilities=probabilities, labels=labels, node_ids=node_ids,
                            node_split=node_split, k_values=KS, **metadata)
    else:
        if not np.isin(final_k, steps).all() or (predicted < 0).any():
            raise ValueError('Incomplete decisions')
        np.savez_compressed(output, node_ids=node_ids, final_k=final_k, predicted=predicted,
                            pattern_passes=pattern_passes, sampled_patterns=sampled,
                            threshold=threshold, target=args.target, rule=args.rule, **metadata)
    print(json.dumps(dict(output=str(output), seconds_total=seconds, nodes=len(nodes)), indent=2))


def checkpoint_hash(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


if __name__ == '__main__':
    main()
