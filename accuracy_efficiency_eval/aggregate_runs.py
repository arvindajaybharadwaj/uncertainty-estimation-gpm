"""Combine final evaluate.py output folders across datasets and checkpoints."""
import argparse
import json
from pathlib import Path

import pandas as pd


def parse_runs(items):
    """Read 'dataset:checkpoint=folder' arguments."""
    runs = []
    for item in items:
        if '=' not in item or ':' not in item.split('=', 1)[0]:
            raise ValueError(f'Expected dataset:checkpoint=folder, found {item!r}.')
        name, folder = item.split('=', 1)
        dataset, checkpoint = name.split(':', 1)
        runs.append((dataset, checkpoint, Path(folder)))
    return runs


def combine(runs):
    curves, successes, units = [], [], set()
    for dataset, checkpoint, folder in runs:
        with open(folder / 'summary.json') as f:
            summary = json.load(f)
        if summary['mode'] != 'FINAL':
            raise ValueError(f'{folder} is a pilot run and cannot be included in final results.')
        units.add(summary['cost_unit'])
        curve = pd.read_csv(folder / 'curve.csv')
        curve['dataset'], curve['checkpoint'] = dataset, checkpoint
        curve['point'] = curve.apply(
            lambda r: f'target {int(r.target)}' if r.method in ('adaptive', 'random') else r.setting, axis=1)
        curves.append(curve)
        if summary.get('success'):
            successes.append(dict(dataset=dataset, checkpoint=checkpoint, **{
                k: v for k, v in summary['success'].items() if k != 'stop_share'}))
    if len(units) > 1:
        raise ValueError(f'Runs mix cost units: {sorted(units)}.')
    table = (pd.concat(curves).groupby(['dataset', 'method', 'point'], sort=False)
             .agg(checkpoints=('accuracy', 'size'), cost_mean=('cost', 'mean'), cost_sd=('cost', 'std'),
                  accuracy_mean=('accuracy', 'mean'), accuracy_sd=('accuracy', 'std')
                  )
             .reset_index())
    return table, pd.DataFrame(successes)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('runs', nargs='+', help='dataset:checkpoint=folder')
    a.add_argument('--output', default='combined')
    args = a.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    curve, success = combine(parse_runs(args.runs))
    curve.to_csv(out / 'combined_curve.csv', index=False)
    success.to_csv(out / 'combined_success.csv', index=False)
    print(curve.to_string(index=False))


if __name__ == '__main__':
    main()
