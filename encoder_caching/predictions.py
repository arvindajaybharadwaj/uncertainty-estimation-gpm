"""Prediction stage in the per-length layout, for one checkpoint.

The same as final_inference.py --stage predictions --sampling nested, except
for the layout: validation and test nodes, seeds 71000 + r, the original batch
size, and one 128-walk pool per batch. Each budget uses the walks layout.order
gives, grouped by length, instead of the first K walks. Budgets K=16, 32, 64
and 128 are the stages, used to choose the threshold and for replay; fixed
K=48 (12 walks per length) is the equal-cost baseline.

K=16 is the same in both layouts, so it must match the original
final_predictions.npz; the run stops without saving if it does not.

Writes <folder>/predictions.npz, in the format of final_predictions.npz plus
`fixed_48`, and refuses to overwrite it.

Example, from the repository root:
    python encoder_caching/predictions.py --dataset wikics --checkpoint-id 1
"""
import argparse
import json
import time

import numpy as np

import paths
from cached_gpm import load
from layout import STEPS

FIXED = 48


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=paths.DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--batch-size', type=int, help='defaults to the original run: 32 for cora_full, 256 otherwise')
    a.add_argument('--smoke', action='store_true', help='32 nodes per split and 2 repetitions')
    args = a.parse_args()

    checkpoint, digest = paths.verified_checkpoint(args.dataset, args.checkpoint_id)
    out = paths.folder(args.dataset, args.checkpoint_id, args.smoke)
    output = out / 'predictions.npz'
    paths.refuse_existing(output)
    original = paths.original_folder(args.dataset, args.checkpoint_id, args.smoke) / 'final_predictions.npz'
    with np.load(original, allow_pickle=False) as z:
        if str(z['checkpoint_sha256']) != digest or not bool(z['nested']) or int(z['seed']) != paths.PREDICTION_SEED:
            raise ValueError(f'{original} is not the nested prediction run of this checkpoint.')
        stored_16, stored_ids = z['probabilities'][list(z['k_values']).index(16)], z['node_ids']
    repeats = paths.SMOKE_REPEATS if args.smoke else paths.REPEATS
    if stored_16.shape[0] != repeats:
        raise ValueError(f'{original} has {stored_16.shape[0]} repetitions, not {repeats}.')
    batch_size = paths.batch_size(args.dataset, args.batch_size)

    import torch
    gpm, ids, set_sampling_seed = load(args.dataset, checkpoint)
    if args.smoke:
        ids = [x[:paths.SMOKE_NODES] for x in ids]
    nodes = torch.cat(ids)
    node_ids = nodes.numpy()
    if not np.array_equal(node_ids, stored_ids):
        raise ValueError('Node order differs from the original predictions, so the split has changed.')
    node_split = np.array(['val'] * len(ids[0]) + ['test'] * len(ids[1]))
    labels = gpm.graph.y.cpu()[nodes].numpy().reshape(-1)
    classes = gpm.params['output_dim']

    def sync():
        if gpm.device.type == 'cuda':
            torch.cuda.synchronize()

    probabilities = np.empty((len(STEPS), repeats, len(nodes), classes), dtype=np.float32)
    fixed = np.empty((repeats, len(nodes), classes), dtype=np.float32)
    with torch.inference_mode():
        # Warm up once, as in final_inference.py; excluded from recorded time.
        first = nodes[:min(len(nodes), batch_size)]
        gpm.predict(first, gpm.sample(first, 16))
        sync()
        start = time.perf_counter()
        for r in range(repeats):
            set_sampling_seed(paths.PREDICTION_SEED + r)
            for offset in range(0, len(nodes), batch_size):
                batch = nodes[offset:offset + batch_size]
                pool = gpm.sample(batch, STEPS[-1])
                for i, k in enumerate(STEPS):
                    probabilities[i, r, offset:offset + len(batch)] = gpm.predict(batch, gpm.arranged(pool, k)).cpu().numpy()
                fixed[r, offset:offset + len(batch)] = gpm.predict(batch, gpm.arranged(pool, FIXED)).cpu().numpy()
            print(f'Repetition {r + 1}/{repeats} complete', flush=True)
        sync()
        seconds = time.perf_counter() - start

    diff = float(np.abs(probabilities[0] - stored_16).max())
    if diff > paths.CHECK_TOLERANCE:
        raise ValueError(f'K=16 differs from {original} by up to {diff:.2e}. It should be identical in both '
                         'layouts, so the walks or environment have changed. Nothing was saved.')
    for p in (probabilities, fixed):
        if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=1e-4):
            raise ValueError('Invalid probabilities')
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, probabilities=probabilities, fixed_48=fixed, labels=labels, node_ids=node_ids,
                        node_split=node_split, k_values=np.array(STEPS), dataset=args.dataset,
                        checkpoint_sha256=digest, nested=True, encoding_reuse=False, layout='per_length',
                        smoke_test=args.smoke, seed=paths.PREDICTION_SEED, batch_size=batch_size,
                        stage1_max_abs_difference=diff, seconds_total=seconds,
                        timing_scope='all repetitions; sampling, inference, transfers and loop overhead; '
                                     'excludes loading, warmup and file saving')
    print(json.dumps(dict(output=str(output), seconds_total=seconds, nodes=len(nodes),
                          stage1_max_abs_difference=diff), indent=2))


if __name__ == '__main__':
    main()
