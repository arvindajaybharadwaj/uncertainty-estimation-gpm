"""Draw the encoder-caching figures, from results that already exist.

Reads only; nothing is run on the GPU:
  runs/<dataset>/ckpt1/timing.json       time per node and accuracy of every method
  runs/<dataset>/ckpt1/comparison.json   walks encoded by cached adaptive inference
  results/summary.csv                    paired differences with 95% CIs, per checkpoint
Writes, into results/, each as .png and .pdf, refusing to replace them unless --overwrite:
  accuracy_vs_time        accuracy against time per node, one panel per dataset
  paired_differences      paired differences with 95% CI, per checkpoint
  time_vs_walks_encoded   time per node against walks put through the walk encoder

Accuracy in accuracy_vs_time comes from the timing runs (seed 271000), so it
differs slightly from the accuracies in summary.csv.

Example, from the repository root:
    python encoder_caching/plot_results.py
"""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paths

# The palette and names of cross_dataset_validation/plot_summary.py.
COLOURS = {'computers': '#2a78d6', 'wikics': '#eb6834', 'cora_full': '#1baf7a'}
NAMES = {'computers': 'Amazon Computers', 'wikics': 'WikiCS', 'cora_full': 'CoraFull'}
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
FIGURES = ('accuracy_vs_time', 'paired_differences', 'time_vs_walks_encoded')

FIXED = ('fixed_16', 'fixed_32', 'fixed_48', 'fixed_64', 'fixed_128')
REPEATED = ('repeated_4x16', 'repeated_8x16')
LABELS = {'nested_four_stage': 'original adaptive', 'cached_four_stage': 'cached adaptive',
          'repeated_4x16': '4 x 16', 'repeated_8x16': '8 x 16'}
LABELS.update({m: 'K=' + m.split('_')[1] for m in FIXED})

COMPARISONS = (  # (column in summary.csv, label), grouped as in evaluate_cached.py
    ('cached_vs_new_fixed_32', 'cached adaptive - fixed K=32'),
    ('cached_vs_new_fixed_48', 'cached adaptive - fixed K=48'),
    ('cached_vs_new_fixed_128', 'cached adaptive - fixed K=128'),
    ('cached_vs_original_adaptive', 'cached - original adaptive'),
    ('layout_fixed_32', 'new - old layout, K=32'),
    ('layout_fixed_48', 'new - old layout, K=48'),
    ('layout_fixed_64', 'new - old layout, K=64'),
    ('layout_fixed_128', 'new - old layout, K=128'),
    ('replay_vs_cached', 'offline replay - live'),
)
GROUPS = ((0, 3, 'Main result'), (3, 4, 'Caching'), (4, 8, 'Layout'), (8, 9, 'Replay'))


def load():
    """Timing rows per dataset (with walks encoded), and the per-checkpoint summary."""
    timing = {}
    for dataset in paths.DATASETS:
        folder = paths.folder(dataset, 1)
        if not (folder / 'timing.json').is_file():
            continue
        with open(folder / 'timing.json') as f:
            rows = {r['method']: dict(r) for r in json.load(f)['summary']}
        with open(folder / 'comparison.json') as f:
            sampled = json.load(f)['cost']['cached_sampled']
        for name, row in rows.items():
            # Walks through the walk encoder: fixed and repeated methods and the original rule
            # encode every walk they are charged for; cached adaptive encodes each sampled walk once.
            row['walks_encoded'] = sampled if name == 'cached_four_stage' else row['pattern_passes']
        timing[dataset] = rows
    return timing, pd.read_csv(paths.RESULTS / 'summary.csv')


def style(ax, title, xlabel, ylabel, grid='both'):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11.5, fontweight='bold', color=INK, pad=10)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis=grid, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('bottom', 'left'):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)


def header(fig, title, subtitle):
    fig.text(.05, .965, title, fontsize=15, fontweight='bold', color=INK, va='top')
    fig.text(.05, .905, subtitle, fontsize=10.5, color=MUTED, va='top')


def accuracy_vs_time(timing):
    fig, axes = plt.subplots(1, len(timing), figsize=(5.4 * len(timing), 5.4), facecolor=SURFACE)
    for ax, (dataset, rows) in zip(np.atleast_1d(axes), timing.items()):
        colour = COLOURS[dataset]
        fixed = [rows[m] for m in FIXED if m in rows]
        ax.plot([r['ms_per_node'] for r in fixed], [100 * r['accuracy'] for r in fixed],
                color=MUTED, lw=1.2, zorder=1)
        for name, row in rows.items():
            x, y = row['ms_per_node'], 100 * row['accuracy']
            if name in FIXED:
                ax.scatter(x, y, s=44, color=MUTED, zorder=3, linewidths=0)
            elif name in REPEATED:
                ax.scatter(x, y, s=44, facecolors=SURFACE, edgecolors=MUTED, linewidths=1.3, zorder=3)
            elif name == 'nested_four_stage':
                ax.scatter(x, y, s=80, facecolors=SURFACE, edgecolors=colour, linewidths=2, zorder=4)
            else:
                ax.scatter(x, y, s=80, color=colour, edgecolors=SURFACE, linewidths=1.5, zorder=5)
            # The adaptive points are named by the legend and the arrow; fixed K labels sit above
            # the curve, repeated ones to the right, so none of them collide.
            if name == 'fixed_16':  # at the left edge, so its label goes to the right
                ax.annotate(LABELS[name], (x, y), xytext=(7, 0), textcoords='offset points', va='center',
                            fontsize=8.5, color=MUTED)
            elif name in FIXED:
                ax.annotate(LABELS[name], (x, y), xytext=(-6, 6), textcoords='offset points', ha='right',
                            fontsize=8.5, color=MUTED)
            elif name in REPEATED:
                ax.annotate(LABELS[name], (x, y), xytext=(7, 0), textcoords='offset points', va='center',
                            fontsize=8.5, color=MUTED)
        if {'nested_four_stage', 'cached_four_stage'} <= set(rows):
            old, new = rows['nested_four_stage'], rows['cached_four_stage']
            ax.annotate('', (new['ms_per_node'], 100 * new['accuracy']),
                        (old['ms_per_node'], 100 * old['accuracy']),
                        arrowprops=dict(arrowstyle='->', color=colour, lw=1.6, shrinkA=6, shrinkB=6), zorder=4)
            ax.text(.98, .04, f"arrow: caching, {old['speedup_vs_fixed_128']:.2f}x to "
                              f"{new['speedup_vs_fixed_128']:.2f}x faster than K=128",
                    transform=ax.transAxes, ha='right', fontsize=9, color=INK)
        style(ax, NAMES[dataset], 'Time per node (ms), lower is faster', 'Test accuracy (%)')
    handles = [Line2D([], [], marker='o', ls='-', color=MUTED, label='fixed K'),
               Line2D([], [], marker='o', ls='', mfc=SURFACE, mec=MUTED, label='repeated 16-walk passes'),
               Line2D([], [], marker='o', ls='', mfc=SURFACE, mec=INK, mew=2, label='original adaptive'),
               Line2D([], [], marker='o', ls='', color=INK, label='cached adaptive')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .975), ncol=4, frameon=False,
               labelcolor=INK)
    header(fig, 'Accuracy against time per node',
           'Checkpoint 1, batch 32, 3 rounds x 10 repetitions on the GPU. Up and to the left is better.')
    fig.subplots_adjust(left=.06, right=.98, top=.8, bottom=.13, wspace=.28)
    return fig


def paired_differences(summary):
    datasets = [d for d in paths.DATASETS if d in set(summary.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 7.2), facecolor=SURFACE,
                             sharey=True)
    rows = np.arange(len(COMPARISONS))
    offsets = {1: -.22, 2: 0, 3: .22}
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        colour = COLOURS[dataset]
        for start, end, _ in GROUPS[1:]:
            ax.axhline(start - .5, color=GRID, lw=1, zorder=1)
        for i, (column, _) in enumerate(COMPARISONS):
            for row in summary[summary.dataset == dataset].itertuples():
                diff, low, high = (getattr(row, column + s) for s in ('_pp', '_ci_low', '_ci_high'))
                clear = low > 0 or high < 0
                y = i + offsets.get(row.checkpoint, 0)
                ax.plot([low, high], [y, y], color=colour, lw=2, alpha=1 if clear else .45, zorder=2)
                ax.scatter(diff, y, s=36, zorder=3, color=colour if clear else SURFACE, edgecolors=colour,
                           linewidths=1.5)
        ax.axvline(0, color=INK, lw=1, zorder=1)
        style(ax, NAMES[dataset], 'Difference in test accuracy (points)', '', grid='x')
        ax.set_yticks(rows)
        ax.set_yticklabels([label for _, label in COMPARISONS])
        ax.set_ylim(len(COMPARISONS) - .5, -.5)
    for start, end, name in GROUPS:
        axes[0].text(-.62, (start + end - 1) / 2, name, transform=axes[0].get_yaxis_transform(), rotation=90,
                     ha='center', va='center', fontsize=9.5, fontweight='bold', color=MUTED)
    handles = [Line2D([], [], marker='o', ls='-', color=INK, label='CI excludes 0 (clear difference)'),
               Line2D([], [], marker='o', ls='-', color=INK, mfc=SURFACE, alpha=.6, label='CI includes 0')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .975), ncol=2, frameon=False,
               labelcolor=INK)
    header(fig, 'Paired differences on test nodes, with 95% CI',
           'One line per checkpoint (1 to 3, top to bottom). Paired node bootstrap, 2,000 draws. '
           'Right of 0: the first method is more accurate.')
    fig.subplots_adjust(left=.2, right=.98, top=.84, bottom=.09, wspace=.12)
    return fig


def time_vs_walks_encoded(timing):
    fig, axes = plt.subplots(1, len(timing), figsize=(5.4 * len(timing), 5.4), facecolor=SURFACE)
    for ax, (dataset, rows) in zip(np.atleast_1d(axes), timing.items()):
        colour = COLOURS[dataset]
        fixed = [rows[m] for m in FIXED if m in rows]
        xs = np.array([r['walks_encoded'] for r in fixed])
        ys = np.array([r['ms_per_node'] for r in fixed])
        slope, intercept = np.polyfit(xs, ys, 1)
        line = np.array([0, xs.max()])
        ax.plot(line, intercept + slope * line, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
        ax.scatter(xs, ys, s=44, color=MUTED, zorder=3, linewidths=0)
        for r, x, y in zip(fixed, xs, ys):
            ax.annotate(LABELS[r['method']], (x, y), xytext=(6, -10), textcoords='offset points', fontsize=8.5,
                        color=MUTED)
        notes = []
        for name, filled in (('cached_four_stage', True), ('nested_four_stage', False)):
            if name not in rows:
                continue
            x, y = rows[name]['walks_encoded'], rows[name]['ms_per_node']
            expected = intercept + slope * x
            ax.scatter(x, y, s=80, zorder=5, color=colour if filled else SURFACE, edgecolors=colour,
                       linewidths=2)
            ax.plot([x, x], [expected, y], color=colour, lw=1.2, ls=':', zorder=2)
            notes.append(f'{LABELS[name]}: {x:.0f} walks, {y:.2f} ms (line: {expected:.2f} ms)')
        # The adaptive points sit among the fixed ones, so their numbers go in the empty corner below the line.
        ax.text(.97, .05, '\n'.join(notes), transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5,
                color=INK, linespacing=1.5)
        style(ax, NAMES[dataset], 'Walks through the walk encoder per node', 'Time per node (ms)')
        ax.set_xlim(0, xs.max() * 1.08)
        ax.set_ylim(0, max(ys.max(), max(r['ms_per_node'] for r in rows.values() if r['method'] not in REPEATED))
                    * 1.12)
    handles = [Line2D([], [], marker='o', ls=(0, (4, 3)), color=MUTED, label='fixed K, with a straight-line fit'),
               Line2D([], [], marker='o', ls='', mfc=SURFACE, mec=INK, mew=2, label='original adaptive'),
               Line2D([], [], marker='o', ls='', color=INK, label='cached adaptive')]
    fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(.98, .975), ncol=3, frameon=False,
               labelcolor=INK)
    header(fig, 'Time per node against walks encoded',
           'Checkpoint 1. The original rule re-encodes every walk at each stage (its pattern-passes); cached '
           'adaptive encodes each walk once.\nThe height above the line is time not explained by encoding: '
           'up to 4 stages, small batches and the pattern Transformer rerun.')
    fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.13, wspace=.28)
    return fig


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--overwrite', action='store_true', help='replace existing figures')
    args = a.parse_args()
    targets = [paths.RESULTS / f'{name}{suffix}' for name in FIGURES for suffix in ('.png', '.pdf')]
    if not args.overwrite:
        paths.refuse_existing(*targets)

    timing, summary = load()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    figures = dict(accuracy_vs_time=accuracy_vs_time(timing), paired_differences=paired_differences(summary),
                   time_vs_walks_encoded=time_vs_walks_encoded(timing))
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        for suffix in ('.png', '.pdf'):
            path = paths.RESULTS / f'{name}{suffix}'
            fig.savefig(path, dpi=200 if suffix == '.png' else None, facecolor=SURFACE)
            print('Saved', path)


if __name__ == '__main__':
    main()
