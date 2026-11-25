"""Live nested four-stage inference with cached walk encodings, for one checkpoint.

The same as final_inference.py --stage live --rule four --sampling nested,
except for the layout and caching:
  1. The threshold comes from choose_threshold in final_inference.py, on the
     validation nodes of this folder's predictions.npz (per-length layout),
     at a target of 48 pattern-passes with costs 16/48/112/240.
  2. On test nodes, with seeds 171000 + r and the original batch size, each
     stage samples k - previous walks for the active nodes, encodes only those,
     and reruns the pattern Transformer on all of the node's cached vectors.
Costs stay in pattern-passes, so the decisions are comparable with the
original run; caching shows up in the measured time.

--check also runs a fresh forward pass on the same walks at every stage and
stops if the cached probabilities differ by more than 1e-4. It slows the run,
so use it for smoke tests.

Writes <folder>/decisions.npz, in the format of adaptive_decisions.npz, and
refuses to overwrite it.

Example, from the repository root:
    python encoder_caching/live_cached.py --dataset wikics --checkpoint-id 1
"""
import argparse
import json
import sys
import time

import numpy as np

import paths
from cached_gpm import load
from layout import STEPS

sys.path.insert(0, str(paths.ROOT / 'GPM'))
from final_inference import choose_threshold  # noqa: E402


def frozen_threshold(calibration, dataset, digest, val_ids, target):
    """choose_threshold on validation nodes, with the checks final_inference.py makes."""
    with np.load(calibration, allow_pickle=False) as z:
        if not bool(z['nested']) or str(z['layout']) != 'per_length':
            raise ValueError(f'{calibration} is not a nested per-length prediction run.')
        if str(z['dataset']) != dataset or str(z['checkpoint_sha256']) != digest:
            raise ValueError('Calibration dataset or checkpoint mismatch')
        cal = z['node_split'] == 'val'
        if not np.array_equal(z['node_ids'][cal], val_ids):
            raise ValueError('Validation IDs differ from calibration file')
        p = z['probabilities'][[list(z['k_values']).index(k) for k in STEPS]][:, :, cal]
        if not np.isfinite(p).all():
            raise ValueError('Calibration probabilities are incomplete')
        repeats = p.shape[1]
    return choose_threshold(p, target, STEPS), repeats


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=paths.DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--batch-size', type=int, help='defaults to the original run: 32 for cora_full, 256 otherwise')
    a.add_argument('--target', type=float, default=paths.TARGET)
    a.add_argument('--smoke', action='store_true', help='32 nodes per split and 2 repetitions')
    a.add_argument('--check', action='store_true', help='compare every cached stage with a fresh pass')
    args = a.parse_args()

    checkpoint, digest = paths.verified_checkpoint(args.dataset, args.checkpoint_id)
    out = paths.folder(args.dataset, args.checkpoint_id, args.smoke)
    output = out / 'decisions.npz'
    paths.refuse_existing(output)
    batch_size = paths.batch_size(args.dataset, args.batch_size)

    import torch
    gpm, ids, set_sampling_seed = load(args.dataset, checkpoint)
    if args.smoke:
        ids = [x[:paths.SMOKE_NODES] for x in ids]
    threshold, repeats = frozen_threshold(out / 'predictions.npz', args.dataset, digest, ids[0].numpy(), args.target)
    print('Frozen validation threshold:', threshold, flush=True)
    nodes = ids[1]
    node_ids = nodes.numpy()

    def sync():
        if gpm.device.type == 'cuda':
            torch.cuda.synchronize()

    final_k = np.zeros((repeats, len(nodes)), dtype=np.int64)
    predicted = np.full_like(final_k, -1)
    pattern_passes = np.zeros_like(final_k)
    sampled = np.zeros_like(final_k)
    largest = 0.
    with torch.inference_mode():
        # Warm up once, as in final_inference.py; excluded from recorded time.
        first = nodes[:min(len(nodes), batch_size)]
        gpm.predict(first, gpm.sample(first, 16))
        sync()
        start = time.perf_counter()
        for r in range(repeats):
            set_sampling_seed(paths.LIVE_SEED + r)
            for offset in range(0, len(nodes), batch_size):
                batch = nodes[offset:offset + batch_size]
                span = slice(offset, offset + len(batch))
                final_k[r, span], predicted[r, span], pattern_passes[r, span], sampled[r, span], diff = \
                    gpm.adaptive(batch, STEPS, threshold, check=args.check)
                if args.check:
                    largest = max(largest, diff)
            print(f'Repetition {r + 1}/{repeats} complete', flush=True)
        sync()
        seconds = time.perf_counter() - start

    if args.check and largest > paths.CHECK_TOLERANCE:
        raise ValueError(f'Cached probabilities differ from a fresh pass by up to {largest:.2e}. Nothing was saved.')
    if not np.isin(final_k, STEPS).all() or (predicted < 0).any():
        raise ValueError('Incomplete decisions')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, node_ids=node_ids, final_k=final_k, predicted=predicted,
                        pattern_passes=pattern_passes, sampled_patterns=sampled, threshold=threshold,
                        target=args.target, rule='four', dataset=args.dataset, checkpoint_sha256=digest,
                        nested=True, encoding_reuse=True, layout='per_length', smoke_test=args.smoke,
                        seed=paths.LIVE_SEED, batch_size=batch_size, seconds_total=seconds,
                        check_max_abs_difference=largest if args.check else np.nan,
                        timing_scope='all repetitions; sampling, inference, transfers and loop overhead; '
                                     'excludes loading, warmup and file saving'
                                     + ('; includes the --check fresh passes' if args.check else ''))
    print(json.dumps(dict(output=str(output), threshold=threshold, seconds_total=seconds, nodes=len(nodes),
                          check_max_abs_difference=largest if args.check else None), indent=2))


if __name__ == '__main__':
    main()
