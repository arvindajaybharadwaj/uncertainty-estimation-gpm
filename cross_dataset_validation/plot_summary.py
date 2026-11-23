"""Draw one figure comparing all datasets, from results that already exist.

Reads only; nothing is run on the GPU:
  results/per_checkpoint.csv                    accuracy gap with CI, and gains at the same cost
  results/uncertainty/per_checkpoint_val.csv    AUROC of each uncertainty signal
  runs/<dataset>/ckpt<id>/diagnostic_val.json   nodes that more walks help, cannot help, or harm
  results/batch_size_timing/<dataset>_ckpt1.json, or else runs/<dataset>/ckpt1/timing.json
Writes results/summary_figure.png and .pdf, and refuses to replace them unless --overwrite.

Panels:
  A  adaptive minus fixed K=128 per checkpoint, with 95% CI and the 0.5-point target
  B  gain over the even K=32/64 mix and over random allocation, at the same cost
  C  speedup over fixed K=128, by inference batch size
  D  validation nodes that more walks cannot fix, do fix, or break
  E  how well each uncertainty signal picks out the nodes that benefit (AUROC)

Example, from the repository root:
    python cross_dataset_validation/plot_summary.py
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

from common import DATASETS, HERE, run_folder

# Categorical slots 1-3 of the reference palette, which stay distinct for colour-blind readers.
COLOURS = {'computers': '#2a78d6', 'wikics': '#eb6834', 'cora_full': '#1baf7a'}
NAMES = {'computers': 'Amazon Computers', 'wikics': 'WikiCS', 'cora_full': 'CoraFull'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
SIGNALS = ('Maximum confidence', 'Predictive entropy', 'Sampling disagreement')
ADAPTIVE = 'nested_four_stage'


def load(results=HERE / 'results'):
    """Everything the figure shows, as plain tables."""
    checkpoints = pd.read_csv(results / 'per_checkpoint.csv')
    datasets = [d for d in DATASETS if d in set(checkpoints.dataset)]
    uncertainty = pd.read_csv(results / 'uncertainty' / 'per_checkpoint_val.csv')

    diagnostics = []
    for row in checkpoints.itertuples():
        path = run_folder(row.dataset, row.checkpoint) / 'diagnostic_val.json'
        if path.is_file():
            with open(path) as f:
                d = json.load(f)
            diagnostics.append(dict(dataset=row.dataset, checkpoint=row.checkpoint,
                                    hopeless=d['nodes_wrong_in_every_run_pct'],
                                    corrected=d['corrected_pct_of_k16_errors'],
                                    broken=d['broken_pct_of_k16_correct']))

    # The batch-size experiment times every batch size in one session, so prefer it when it exists.
    timing = []
    for dataset in datasets:
        sweep = results / 'batch_size_timing' / f'{dataset}_ckpt1.json'
        single = run_folder(dataset, 1) / 'timing.json'
        if sweep.is_file():
            with open(sweep) as f:
                rows = json.load(f)['summary']
        elif single.is_file():
            with open(single) as f:
                t = json.load(f)
            rows = [dict(r, batch_size=t['batch_size']) for r in t['summary']]
        else:
            continue
        timing += [dict(dataset=dataset, batch_size=int(r['batch_size']), speedup=r['speedup_vs_fixed_128'],
                        low=r['speedup_min'], high=r['speedup_max'])
                   for r in rows if r['method'] == ADAPTIVE]
    return datasets, checkpoints, uncertainty, pd.DataFrame(diagnostics), pd.DataFrame(timing)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis='y', color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def grouped_bars(ax, datasets, groups, means, points, fmt):
    """One bar per dataset in each group, at the checkpoint mean, with each checkpoint as a dot."""
    width = .8 / len(datasets)
    for j, dataset in enumerate(datasets):
        xs = np.arange(len(groups)) + (j - (len(datasets) - 1) / 2) * width
        ax.bar(xs, means[dataset], width * .88, color=COLOURS[dataset], zorder=2)
        for x, mean, values in zip(xs, means[dataset], points[dataset]):
            ax.scatter(np.full(len(values), x), values, s=14, color=INK, alpha=.75, zorder=3, linewidths=0)
            end = max([mean, *values]) if mean >= 0 else min([mean, *values])
            ax.annotate(fmt(mean), (x, end), xytext=(0, 4 if mean >= 0 else -4), textcoords='offset points',
                        ha='center', va='bottom' if mean >= 0 else 'top', fontsize=8.5, color=INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.axhline(0, color='#c3c2b7', lw=1, zorder=1)


def panel_gap(ax, datasets, checkpoints):
    offsets = {1: -.22, 2: 0, 3: .22}
    for i, dataset in enumerate(datasets):
        rows = checkpoints[checkpoints.dataset == dataset]
        for row in rows.itertuples():
            x = i + offsets.get(int(row.checkpoint), 0)
            ax.errorbar(x, row.gap_vs_fixed_128_pp, yerr=[[row.gap_vs_fixed_128_pp - row.gap_ci_low_pp],
                                                          [row.gap_ci_high_pp - row.gap_vs_fixed_128_pp]],
                        fmt='o', ms=8, color=COLOURS[dataset], ecolor=COLOURS[dataset], elinewidth=2,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        within = int(rows.within_target.astype(str).str.lower().eq('true').sum())
        ax.text(i, .32, f'{within}/{len(rows)} within target', ha='center', fontsize=9, color=INK)
    ax.axhline(0, color='#c3c2b7', lw=1, zorder=1)
    ax.axhline(-.5, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.text(-.55, -.5, '0.5-point target', color=MUTED, fontsize=9, ha='left', va='top')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels([NAMES[d] for d in datasets])
    ax.set_xlim(-.6, len(datasets) - .4)
    ax.set_ylim(min(-1.1, checkpoints.gap_ci_low_pp.min() - .1), .45)
    style(ax, 'A. Accuracy lost against fixed K=128', 'Adaptive minus fixed K=128 (points)')


def panel_gains(ax, datasets, checkpoints):
    columns = ('gain_vs_fixed_same_cost_pp', 'gain_vs_random_same_cost_pp')
    groups = ('vs even K=32/64 mix', 'vs random allocation')
    means = {d: [checkpoints[checkpoints.dataset == d][c].mean() for c in columns] for d in datasets}
    points = {d: [checkpoints[checkpoints.dataset == d][c].tolist() for c in columns] for d in datasets}
    grouped_bars(ax, datasets, groups, means, points, lambda v: f'{v:+.2f}')
    style(ax, 'B. Gain over baselines at the same cost', 'Accuracy gain (points)')


def panel_speedup(ax, datasets, timing):
    if timing.empty:
        ax.text(.5, .5, 'No timing results found', ha='center', transform=ax.transAxes, color=MUTED)
        style(ax, 'C. Speedup over fixed K=128', 'Times faster than fixed K=128')
        return
    order = {d: i for i, d in enumerate(datasets)}
    timing = timing.assign(rank=timing.dataset.map(order)).sort_values(['rank', 'batch_size'])
    xs = np.arange(len(timing))
    ax.bar(xs, timing.speedup, .7, color=[COLOURS[d] for d in timing.dataset], zorder=2)
    ax.errorbar(xs, timing.speedup, yerr=[timing.speedup - timing.low, timing.high - timing.speedup],
                fmt='none', ecolor=INK, elinewidth=1.2, capsize=3, zorder=3)
    for x, row in zip(xs, timing.itertuples()):
        ax.annotate(f'{row.speedup:.2f}x', (x, row.high), xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_xticks(xs)
    short = {'computers': 'Computers', 'wikics': 'WikiCS', 'cora_full': 'CoraFull'}
    ax.set_xticklabels([f'{short.get(d, d)}\nbatch {b}' for d, b in zip(timing.dataset, timing.batch_size)], fontsize=9)
    ax.set_ylim(0, timing.high.max() * 1.18)
    style(ax, 'C. Speedup over fixed K=128 (checkpoint 1)', 'Times faster than fixed K=128')
    ax.text(0, -.30, 'Compare bars at the same batch size only.\nLines show the range over timing rounds.',
            transform=ax.transAxes, fontsize=8.5, color=MUTED)


def panel_diagnostics(ax, datasets, diagnostics):
    columns = ('hopeless', 'corrected', 'broken')
    groups = ('Wrong even at K=128\n(% of nodes)', 'K=16 errors fixed by K=128\n(% of K=16 errors)',
              'K=16 correct answers broken\nby K=128 (% of K=16 correct)')
    present = [d for d in datasets if d in set(diagnostics.dataset)] if not diagnostics.empty else []
    means = {d: [diagnostics[diagnostics.dataset == d][c].mean() for c in columns] for d in present}
    points = {d: [diagnostics[diagnostics.dataset == d][c].tolist() for c in columns] for d in present}
    if present:
        grouped_bars(ax, present, groups, means, points, lambda v: f'{v:.1f}%')
    style(ax, 'D. What more walks do on validation nodes (K=16 to K=128)', 'Percent')


def panel_signals(ax, datasets, uncertainty):
    passes = uncertainty.groupby('method').passes.first()
    width = .8 / len(datasets)
    for j, dataset in enumerate(datasets):
        rows = uncertainty[uncertainty.dataset == dataset]
        for i, signal in enumerate(SIGNALS):
            values = rows[rows.method == signal].auroc
            if values.empty:
                continue
            x = i + (j - (len(datasets) - 1) / 2) * width
            ax.plot([x, x], [values.min(), values.max()], color=COLOURS[dataset], lw=2, zorder=2)
            ax.scatter(x, values.mean(), s=64, color=COLOURS[dataset], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.annotate(f'{values.mean():.2f}', (x, values.mean()), xytext=(7, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=INK)
    labels = [f'{s}\n({int(passes[s])} pass{"es" if passes.get(s, 1) > 1 else ""})' if s in passes else s
              for s in SIGNALS]
    ax.set_xticks(range(len(SIGNALS)))
    ax.set_xticklabels(labels)
    ax.set_ylim(.5, 1)
    ax.text(len(SIGNALS) - .55, .51, '0.5 = no better than chance', color=MUTED, fontsize=8.5, ha='right', va='bottom')
    style(ax, 'E. How well each signal finds nodes that benefit from more walks', 'AUROC (validation)')
    ax.text(0, -.30, 'Dot: mean of 3 checkpoints. Line: range across checkpoints.',
            transform=ax.transAxes, fontsize=8.5, color=MUTED)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--output', default=str(HERE / 'results' / 'summary_figure'),
                   help='path without extension; .png and .pdf are written')
    a.add_argument('--overwrite', action='store_true', help='replace an existing figure')
    args = a.parse_args()
    output = Path(args.output)
    targets = [output.with_suffix('.png'), output.with_suffix('.pdf')]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f'Refusing to overwrite {", ".join(existing)}. Use --overwrite to replace.')

    datasets, checkpoints, uncertainty, diagnostics, timing = load()

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(16, 10.5), facecolor=SURFACE)
    axes = fig.subplot_mosaic([['A', 'A', 'B', 'B', 'C', 'C'], ['D', 'D', 'D', 'E', 'E', 'E']],
                              gridspec_kw=dict(hspace=.62, wspace=.9))
    panel_gap(axes['A'], datasets, checkpoints)
    panel_gains(axes['B'], datasets, checkpoints)
    panel_speedup(axes['C'], datasets, timing)
    panel_diagnostics(axes['D'], datasets, diagnostics)
    panel_signals(axes['E'], datasets, uncertainty)

    fig.text(.05, .965, 'Adaptive pattern sampling across datasets', fontsize=15, fontweight='bold', color=INK)
    fig.text(.05, .935, 'A-C on test nodes, D-E on validation nodes. Bars are means of 3 checkpoints; '
                        'black dots are individual checkpoints.', fontsize=10.5, color=MUTED)
    fig.legend(handles=[Patch(color=COLOURS[d], label=NAMES[d]) for d in datasets], loc='upper right',
               bbox_to_anchor=(.97, .975), ncol=len(datasets), frameon=False, labelcolor=INK)
    fig.subplots_adjust(left=.05, right=.97, top=.86, bottom=.11)
    output.parent.mkdir(parents=True, exist_ok=True)
    for path in targets:
        fig.savefig(path, dpi=200 if path.suffix == '.png' else None, facecolor=SURFACE)
        print('Saved', path)


if __name__ == '__main__':
    main()
