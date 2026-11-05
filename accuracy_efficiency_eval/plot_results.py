"""Draw the accuracy-cost chart from evaluate.py's output folder."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--results', default='results')
    results = Path(a.parse_args().results)
    curve = pd.read_csv(results / 'curve.csv')
    with open(results / 'summary.json') as f:
        summary = json.load(f)
    seconds = summary['cost_unit'] != 'pattern-passes'
    if seconds:  # plot milliseconds, which are easier to read per node
        curve['cost'] *= 1000

    fixed = curve[(curve.method == 'fixed') & curve.setting.isin(['K=16', 'K=32', 'K=64', 'K=128'])]
    endpoint = fixed[fixed.setting.isin(['K=16', 'K=128'])]
    randoms = pd.concat([endpoint, curve[curve.method == 'random']]).sort_values('cost')
    repeated = curve[curve.setting.str.contains('at K=16', regex=False) & (curve.method == 'repeated')]
    repeated = pd.concat([fixed[fixed.setting == 'K=16'], repeated]).sort_values('cost')
    adaptive = curve[curve.method == 'adaptive'].sort_values('cost')
    baseline = float(fixed[fixed.setting == 'K=128'].accuracy.iloc[0]) * 100

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhline(baseline - .5, color=MUTED, lw=1, ls=(0, (4, 4)), zorder=1)
    ax.text(fixed.cost.max(), baseline - .5 - .05, '0.5-point target below fixed K=128', color=MUTED,
            ha='right', va='top', fontsize=10)
    series = [(fixed, ORANGE, 'Fixed K', 'o', 3), (randoms, AQUA, 'Random allocation', 's', 2),
              (repeated, YELLOW, 'Repeated sampling (K=16)', 'D', 2), (adaptive, BLUE, 'Adaptive replay', 'o', 4)]
    for rows, colour, label, marker, z in series:
        ax.plot(rows.cost, rows.accuracy * 100, color=colour, lw=2, marker=marker, ms=8, label=label, zorder=z,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    def label(rows, text, share, offset, **kw):
        """Name a line at a point part-way along the cost axis."""
        x = fixed.cost.min() + share * (fixed.cost.max() - fixed.cost.min())
        y = float(np.interp(x, rows.cost, rows.accuracy * 100))
        ax.text(x, y + offset, text, color=INK, ha='center', **kw)

    label(adaptive.iloc[1:], 'Adaptive', .46, .035, fontweight='bold')
    label(fixed, 'Fixed K', .72, .04)
    label(repeated, 'Repeated', .72, -.09)
    label(randoms, 'Random', .52, -.11)
    ax.set_xlabel('Estimated milliseconds per node, from measured GPM time' if seconds else 'Patterns sent through GPM per node, counting every pass',
                  color=MUTED)
    ax.set_ylabel('Accuracy (%)', color=MUTED)
    ax.grid(axis='y', color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc='lower right', labelcolor=INK, bbox_to_anchor=(1, .03))
    mode = 'Pilot on saved test nodes, not a final result' if summary['mode'] == 'PILOT' else 'Final evaluation on test nodes'
    fig.text(.075, .955, 'Accuracy against estimated inference time' if seconds else
             'Accuracy against patterns sent through GPM', fontsize=14,
             fontweight='bold', color=INK)
    fig.text(.075, .91, f'{mode}. Amazon Computers, {summary["n_evaluation"]:,} evaluation nodes.', fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=.075, right=.97, top=.87, bottom=.11)
    fig.savefig(results / 'accuracy_vs_cost.png', dpi=200, facecolor=SURFACE)
    fig.savefig(results / 'accuracy_vs_cost.pdf', facecolor=SURFACE)


if __name__ == '__main__':
    main()
