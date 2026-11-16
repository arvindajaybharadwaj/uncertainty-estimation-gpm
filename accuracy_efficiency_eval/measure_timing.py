"""Measure real GPM inference cost at each pattern budget.

Run this where GPM already runs, with a GPU. It needs no changes to GPM and
records time for sampling, pattern encoding and the rest of the model, plus
peak GPU memory. evaluate.py reads the output through --timing.

Example, from this folder:
    python measure_timing.py --gpm-dir ../GPM/GPM --checkpoint /path/to/epoch_380.pt
"""
import argparse
import json
import os.path as osp
import sys
import time
import types

DEFAULT_KS = (4, 8, 16, 32, 64, 128)


def summarise(k, nodes, passes, sample_s, total_model_s, encode_s, peak_mb):
    """Convert measured totals into seconds per node for one pass."""
    if passes < 1 or nodes < 1:
        raise ValueError('Timing needs at least one node and one pass.')
    count = nodes * passes
    transform_s = total_model_s - encode_s
    if transform_s < 0:
        raise ValueError('Encoding time cannot exceed total model time.')
    return dict(k=k, nodes=nodes, passes=passes,
                sample_s_per_node=sample_s / count,
                encode_s_per_node=encode_s / count,
                transform_s_per_node=transform_s / count,
                total_s_per_node=(sample_s + total_model_s) / count,
                peak_gpu_mb=peak_mb)


def batches(items, size):
    return [items[i:i + size] for i in range(0, len(items), size)]


def install_dgl_stub_if_missing():
    """GPM imports dgl's node2vec sampler but never calls it, so allow it to be absent."""
    try:
        import dgl  # noqa: F401
    except Exception:
        dgl = types.ModuleType('dgl')
        sampling = types.ModuleType('dgl.sampling')

        def unavailable(*args, **kwargs):
            raise RuntimeError('dgl is not installed, and this call needs it.')

        sampling.node2vec_random_walk = unavailable
        dgl.sampling = sampling
        sys.modules['dgl'] = dgl
        sys.modules['dgl.sampling'] = sampling


def compare_with_saved(rows, node_ids, saved):
    """Confirm this run has the same nodes and similar accuracy as the saved predictions.

    `saved` is a dict with node_ids (array) and accuracy ({k: value}). Node IDs must
    match exactly. Accuracy is compared separately, because a different checkpoint
    or library version would show up there.
    """
    if list(node_ids) != list(saved['node_ids']):
        raise ValueError('Test nodes differ from the saved predictions, so the split has changed.')
    gaps = {r['k']: r['accuracy'] - saved['accuracy'][r['k']] for r in rows if r['k'] in saved['accuracy']}
    return dict(accuracy_gaps=gaps, largest_gap=max((abs(v) for v in gaps.values()), default=None))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--gpm-dir', required=True, help='folder containing sampling_experiment.py')
    a.add_argument('--checkpoint', required=True)
    a.add_argument('--dataset', default='computers')
    a.add_argument('--k-values', type=int, nargs='+', default=list(DEFAULT_KS))
    a.add_argument('--repeats', type=int, default=5)
    a.add_argument('--batch-size', type=int, default=256)
    a.add_argument('--base-seed', type=int, default=50000)
    a.add_argument('--output', default='timing.json')
    a.add_argument('--expected-predictions', help='predictions.npz to check nodes and accuracy against')
    args = a.parse_args()

    sys.path.insert(0, osp.abspath(args.gpm_dir))
    install_dgl_stub_if_missing()
    import copy
    import numpy as np
    import torch
    from torch_geometric.data import Data, InMemoryDataset
    from data.pyg_data_loader import load_data
    from model.model import Model
    from model.random_walk import get_patterns
    from sampling_experiment import load_baseline_params, set_sampling_seed
    from utils.utils import mask2idx, seed_everything

    gpu = torch.cuda.is_available()
    device = torch.device('cuda:0' if gpu else 'cpu')

    def sync():
        if gpu:
            torch.cuda.synchronize()

    seed_everything(42)
    params = load_baseline_params(args.dataset)
    params['device'] = device
    graph, splits = load_data(params)
    if isinstance(graph, Data):
        params['input_dim'] = graph.x.size(1)
        params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    elif isinstance(graph, InMemoryDataset):
        params['input_dim'] = graph._data.x_feat.size(1)
        params['edge_dim'] = graph._data.e_feat.size(1) if graph._data.edge_attr is not None else 0
    params['output_dim'] = params['num_tasks'] if params.get('num_tasks') is not None else graph.y.max().item() + 1

    model = Model(params=params).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Time the pattern encoder without changing the GPM source.
    encode_seconds = [0.0]
    original = model.pattern_encoder.encode_node

    def timed_encode(*a_, **k_):
        sync()
        start = time.perf_counter()
        out = original(*a_, **k_)
        sync()
        encode_seconds[0] += time.perf_counter() - start
        return out

    model.pattern_encoder.encode_node = timed_encode

    test_idx = mask2idx(splits[0]['test'])
    labels = graph.y.cpu()
    test_idx = test_idx.cpu()
    groups = batches(test_idx, args.batch_size)
    rows = []

    def one_pass(batch, k):
        run_params = copy.copy(params)
        run_params['pre_sample_pattern_num'] = k
        sync()
        t0 = time.perf_counter()
        patterns, eids = get_patterns(graph, run_params, nodes=batch)
        run_params['pattern_set'] = {'pattern': patterns, 'eid': eids}
        sync()
        t1 = time.perf_counter()
        logits = model(graph, torch.arange(len(batch), dtype=torch.long), run_params, mode='eval')[0]
        sync()
        return t1 - t0, time.perf_counter() - t1, logits

    with torch.no_grad():
        for k in args.k_values:
            one_pass(groups[0], k)  # warm-up, not counted
            if gpu:
                torch.cuda.reset_peak_memory_stats()
            encode_seconds[0] = 0.0
            sample_s = model_s = 0.0
            right = 0
            for r in range(args.repeats):
                set_sampling_seed(args.base_seed + 1000 * k + r)
                for batch in groups:
                    s, m, logits = one_pass(batch, k)
                    sample_s += s
                    model_s += m
                    right += (logits.argmax(-1).cpu() == labels[batch].cpu().view(-1)).sum().item()
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if gpu else None
            rows.append(summarise(k, len(test_idx), args.repeats, sample_s, model_s, encode_seconds[0], peak))
            rows[-1]['accuracy'] = right / (len(test_idx) * args.repeats)
            print(rows[-1], flush=True)

    check = None
    if args.expected_predictions:
        with np.load(args.expected_predictions, allow_pickle=False) as z:
            p, y = z['probabilities'], z['labels']
            saved = dict(node_ids=z['node_ids'].tolist(),
                         accuracy={int(k): float((p[i].argmax(-1) == y).mean()) for i, k in enumerate(z['k_values'])})
        check = compare_with_saved(rows, test_idx.tolist(), saved)
        print('Comparison with saved predictions:', check, flush=True)

    with open(args.output, 'w') as f:
        json.dump(dict(dataset=args.dataset, checkpoint=args.checkpoint, batch_size=args.batch_size,
                       device=torch.cuda.get_device_name(0) if gpu else 'cpu', torch=torch.__version__,
                       per_k=rows, saved_prediction_check=check), f, indent=2)
    print('Saved', args.output)


if __name__ == '__main__':
    main()
