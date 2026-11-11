"""Draw adaptive gains per dataset and checkpoint from combined_live.csv."""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BLUE, AQUA, ORANGE = '#2a78d6', '#1baf7a', '#eb6834'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
PANELS = (('fixed_128', 'Vs fixed K = 128', ORANGE), ('fixed_mix', 'Vs fixed K at the same cost', BLUE),
          ('random_same_cost', 'Vs random allocation at the same cost', AQUA))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--live', required=True)
    a.add_argument('--output', required=True)
    args = a.parse_args()
    live = pd.read_csv(args.live).sort_values(['dataset', 'checkpoint'], ascending=[False, True]).reset_index(drop=True)
    labels = [f"{'Computers' if d == 'computers' else 'WikiCS'} {c}" for d, c in zip(live.dataset, live.checkpoint)]
    y = list(range(len(live)))[::-1]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    for ax, (key, title, colour) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        d, lo, hi = live[f'{key}_diff'] * 100, live[f'{key}_ci_low'] * 100, live[f'{key}_ci_high'] * 100
        ax.axvline(0, color=MUTED, lw=1)
        if key == 'fixed_128':
            ax.axvline(-.5, color=MUTED, lw=1, ls=(0, (4, 4)))
            ax.text(-.49, -.45, '0.5-point limit', color=MUTED, fontsize=9.5, ha='left', va='center')
        ax.errorbar(d, y, xerr=[d - lo, hi - d], fmt='o', color=colour, ms=8, elinewidth=2, capsize=0,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.set_title(title, fontsize=11, color=INK, loc='left')
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0)
        ax.set_xlabel('Accuracy gain, points', color=MUTED)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(-.6, len(live) - .1)
    axes[0].axhline(2.5, color=GRID, lw=1)
    fig.text(.02, .955, 'Adaptive held within 0.5 points everywhere, but gained less on WikiCS', fontsize=14,
             fontweight='bold', color=INK)
    fig.text(.02, .905, 'Nested four-stage adaptive inference, threshold frozen at a target of 48 pattern-passes. '
             'Bars are 95% intervals.', fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=.1, right=.98, top=.8, bottom=.13, wspace=.12)
    fig.savefig(args.output, dpi=200, facecolor=SURFACE)


if __name__ == '__main__':
    main()
