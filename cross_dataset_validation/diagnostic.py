"""Does a dataset leave room for adaptive sampling? K=16 against K=128 on one checkpoint.

The check the project plan asks for before the full comparison on a new dataset:
  - how much average accuracy changes from K=16 to K=128
  - how many errors more patterns correct, and how many correct predictions break
  - whether confidence identifies the nodes that benefit
Patterns are nested, so in each run the K=128 set contains that run's K=16 set,
and corrected or broken predictions are counted within the same run.

Uses validation nodes by default, so no test labels influence any decision.

Example, from the repository root:
    python cross_dataset_validation/diagnostic.py --predictions cross_dataset_validation/runs/wikics/ckpt1/final_predictions.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np

from uncertainty_comparison import compare, load


def diagnose(data):
    i16, i128 = data['ks'].index(16), data['ks'].index(128)
    right16 = data['p'][i16].argmax(-1) == data['y']      # [run, node]
    right128 = data['p'][i128].argmax(-1) == data['y']
    changed = data['p'][i16].argmax(-1) != data['p'][i128].argmax(-1)
    gain = right128.mean(0) - right16.mean(0)                # per node, over runs
    confidence = compare(data).set_index('method').loc['Maximum confidence']
    return dict(
        nodes=int(right16.shape[1]), runs=int(right16.shape[0]),
        accuracy_by_k={str(k): 100 * float((data['p'][i].argmax(-1) == data['y']).mean())
                       for i, k in enumerate(data['ks'])},
        accuracy_change_pp=100 * float(right128.mean() - right16.mean()),
        prediction_changed_pct=100 * float(changed.mean()),
        corrected_pct_of_k16_errors=100 * float((~right16 & right128).sum() / max((~right16).sum(), 1)),
        broken_pct_of_k16_correct=100 * float((right16 & ~right128).sum() / max(right16.sum(), 1)),
        nodes_benefit_pct=100 * float((gain > 0).mean()),
        nodes_unchanged_pct=100 * float((gain == 0).mean()),
        nodes_harmed_pct=100 * float((gain < 0).mean()),
        nodes_wrong_in_every_run_pct=100 * float((~right16.any(0) & ~right128.any(0)).mean()),
        confidence_auroc=float(confidence['auroc']),
        confidence_spearman=float(confidence['spearman']),
        confidence_top20_benefit_pp=float(confidence['top20_benefit_pp']),
        all_nodes_benefit_pp=float(confidence['all_benefit_pp']),
    )


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--predictions', required=True)
    a.add_argument('--split', default='val', choices=['val', 'test'])
    a.add_argument('--output', help='JSON file; default: diagnostic_<split>.json beside the predictions')
    args = a.parse_args()
    result = dict(predictions=args.predictions, split=args.split, **diagnose(load(args.predictions, args.split)))
    out = Path(args.output) if args.output else Path(args.predictions).parent / f'diagnostic_{args.split}.json'
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    print('Saved', out)


if __name__ == '__main__':
    main()
