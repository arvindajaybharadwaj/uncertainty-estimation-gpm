"""Fig. 1 of the ESANN paper, redrawn from the per-length layout predictions.

Validation nodes, K=16 against K=128, for every checkpoint in runs/:
  (a) prediction outcomes: nodes wrong at both budgets in every repetition,
      K=16 errors recovered by K=128, and K=16 correct predictions harmed
  (b) AUROC of confidence, entropy and sampling disagreement for detecting
      nodes that benefit from more walks
Bars and dots are means over the three checkpoints; whiskers show their range.

The numbers come from diagnostic.py and uncertainty_comparison.py in
cross_dataset_validation, called unchanged on this folder's predictions.npz.
K=16 is identical in both layouts, so only the K=128 side can differ from the
paper. The original values are printed next to the new ones for comparison.

Reads only. Writes <output>.png and .pdf, refusing to replace them unless --overwrite.

Example, from the repository root:
    python encoder_caching/plot_sampling_benefit.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import paths
from diagnostic import diagnose
from uncertainty_comparison import METHODS, compare, load

# The colours and order of the paper's figure.
COLOURS = {'computers': '#0072b2', 'wikics': '#e69f00', 'cora_full': '#009e73'}
NAMES = {'computers': 'Computers', 'wikics': 'WikiCS', 'cora_full': 'CoraFull'}
GRID = '#e7e6e1'
OUTCOMES = (('nodes_wrong_in_every_run_pct', 'Wrong at\nboth budgets'),
            ('corrected_pct_of_k16_errors', 'Recovered'),
            ('broken_pct_of_k16_correct', 'Harmed'))
SIGNALS = dict(zip(METHODS, ('Confidence', 'Entropy', 'Sampling\ndisagreement')))


def results():
    """Outcome percentages and AUROCs per checkpoint, for the new and the original layout."""
    rows = []
    for dataset in paths.DATASETS:
        for checkpoint in (1, 2, 3):
            new = paths.folder(dataset, checkpoint) / 'predictions.npz'
            if not new.is_file():
                continue
            data = load(new, 'val')
            row = dict(dataset=dataset, checkpoint=checkpoint, layout='new', **diagnose(data))
            row.update({m: a for m, a in compare(data).set_index('method').auroc.items()})
            rows.append(row)
            # The original run's diagnostic and AUROCs, already computed by cross_dataset_validation.
            old = paths.original_folder(dataset, checkpoint) / 'diagnostic_val.json'
            if old.is_file():
                with open(old) as f:
                    rows.append(dict(dataset=dataset, checkpoint=checkpoint, layout='original', **json.load(f)))
    table = pd.DataFrame(rows)
    aurocs = paths.CROSS / 'results' / 'uncertainty' / 'per_checkpoint_val.csv'
    if aurocs.is_file():
        old = pd.read_csv(aurocs).pivot_table(index=['dataset', 'checkpoint'], columns='method', values='auroc')
        for method in METHODS:
            for (dataset, checkpoint), value in old[method].items():
                table.loc[(table.layout == 'original') & (table.dataset == dataset)
                          & (table.checkpoint == int(checkpoint)), method] = value
    return table


def draw(table):
    new = table[table.layout == 'new']
    datasets = [d for d in paths.DATASETS if d in set(new.dataset)]
    width = .8 / len(datasets)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    fig, (a, b) = plt.subplots(1, 2, figsize=(7, 3.4), facecolor='white')

    for j, dataset in enumerate(datasets):
        rows = new[new.dataset == dataset]
        shift = (j - (len(datasets) - 1) / 2) * width
        for i, (column, _) in enumerate(OUTCOMES):
            values = rows[column]
            a.bar(i + shift, values.mean(), width * .8, color=COLOURS[dataset], zorder=2)
            a.errorbar(i + shift, values.mean(), yerr=[[values.mean() - values.min()], [values.max() - values.mean()]],
                       fmt='none', ecolor='#333333', elinewidth=.8, capsize=2, zorder=3)
        for i, method in enumerate(METHODS):
            values = rows[method]
            b.errorbar(i + shift, values.mean(), yerr=[[values.mean() - values.min()], [values.max() - values.mean()]],
                       fmt='o', ms=4.5, color=COLOURS[dataset], ecolor='#333333', elinewidth=.8, capsize=2, zorder=3)

    a.set_xticks(range(len(OUTCOMES)))
    a.set_xticklabels([label for _, label in OUTCOMES])
    a.set_ylabel('Percent')
    a.set_ylim(0, 25)
    a.set_title('(a) Prediction outcomes', fontsize=9.5)
    b.set_xticks(range(len(METHODS)))
    b.set_xticklabels([SIGNALS[m] for m in METHODS])
    b.set_ylabel('AUROC')
    b.set_ylim(.5, 1)
    b.set_title('(b) Identifying benefit', fontsize=9.5)
    for ax in (a, b):
        ax.grid(axis='y', color=GRID, lw=.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
    fig.legend(handles=[Patch(color=COLOURS[d], label=NAMES[d]) for d in datasets], loc='lower center',
               ncol=len(datasets), frameon=False)
    fig.subplots_adjust(left=.08, right=.98, top=.9, bottom=.27, wspace=.3)
    return fig


def report(table):
    """Means over checkpoints, new layout against the original, as the paper quotes them."""
    columns = [c for c, _ in OUTCOMES] + list(METHODS)
    means = table.groupby(['dataset', 'layout'], sort=False)[columns].mean()
    ranges = table.groupby(['dataset', 'layout'], sort=False)[columns].agg(lambda v: v.max() - v.min())
    print('Means over checkpoints (range in brackets). Outcomes in percent, signals as AUROC.')
    for dataset in paths.DATASETS:
        for layout in ('original', 'new'):
            if (dataset, layout) not in means.index:
                continue
            cells = [f'{means.loc[(dataset, layout), c]:.3f} ({ranges.loc[(dataset, layout), c]:.3f})'
                     if c in METHODS else f'{means.loc[(dataset, layout), c]:.1f} ({ranges.loc[(dataset, layout), c]:.1f})'
                     for c in columns]
            print(f'{NAMES[dataset]:<10} {layout:<9} ' + '  '.join(f'{n.split(chr(10))[0][:12]}={v}' for n, v in
                  zip([l for _, l in OUTCOMES] + [SIGNALS[m] for m in METHODS], cells)))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--output', default=str(paths.RESULTS / 'sampling_benefit'),
                   help='path without extension; .png and .pdf are written')
    a.add_argument('--overwrite', action='store_true', help='replace an existing figure')
    args = a.parse_args()
    output = Path(args.output)
    targets = [output.with_suffix('.png'), output.with_suffix('.pdf')]
    if not args.overwrite:
        paths.refuse_existing(*targets)

    table = results()
    report(table)
    fig = draw(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in targets:
        fig.savefig(path, dpi=300 if path.suffix == '.png' else None)
        print('Saved', path)


if __name__ == '__main__':
    main()
