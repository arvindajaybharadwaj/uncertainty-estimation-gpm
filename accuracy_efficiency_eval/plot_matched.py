"""Draw accuracy against matched inference time from matched_timing.py."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BLUE, ORANGE, YELLOW = '#2a78d6', '#eb6834', '#eda100'
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e7e6e1', '#fcfcfb'
NAMES = {'nested_four_stage': 'Nested four-stage', 'fresh_four_stage': 'Fresh four-stage', 'fresh_two_stage': 'Fresh two-stage'}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--timing', required=True)
    a.add_argument('--output', required=True)
    args = a.parse_args()
    with open(args.timing) as f:
        data = json.load(f)
    rows = {r['method']: r for r in data['summary']}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    fixed = sorted((r for n, r in rows.items() if n.startswith('fixed_')), key=lambda r: r['ms_per_node'])
    repeated = sorted((r for n, r in rows.items() if n.startswith('repeated_')), key=lambda r: r['ms_per_node'])
    adaptive = [rows[n] for n in NAMES if n in rows]
    for group, colour, label, marker, style in ((fixed, ORANGE, 'Fixed K', 'o', '-'), (repeated, YELLOW, 'Repeated K=16', 'D', 'none')):
        ax.plot([r['ms_per_node'] for r in group], [r['accuracy'] * 100 for r in group], color=colour, marker=marker,
                ls=style, lw=2, ms=8, label=label, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.scatter([r['ms_per_node'] for r in adaptive], [r['accuracy'] * 100 for r in adaptive], s=95, color=BLUE,
               edgecolor=SURFACE, linewidth=1.5, zorder=5, label='Adaptive')
    for r, (dx, dy, ha) in zip(adaptive, ((-.012, -.035, 'right'), (.012, -.04, 'left'), (.012, .015, 'left'))):
        ax.text(r['ms_per_node'] + dx, r['accuracy'] * 100 + dy, NAMES[r['method']], color=INK, fontsize=10, ha=ha)
    ax.set_xlabel('Milliseconds per node, measured', color=MUTED)
    ax.set_ylabel('Accuracy (%)', color=MUTED)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(frameon=False, loc='lower right', labelcolor=INK)
    best = max(adaptive, key=lambda r: r['speedup_vs_fixed_128'])
    fig.text(.08, .95, f'Fastest adaptive rule ran {best["speedup_vs_fixed_128"]:.1f} times faster than fixed K=128',
             fontsize=14, fontweight='bold', color=INK)
    fig.text(.08, .905, f'Amazon Computers, {data["device"]}, {data["rounds"]} matched round{"s" if data["rounds"] != 1 else ""} of {data["repeats"]} repetitions.',
             fontsize=10.5, color=MUTED)
    fig.subplots_adjust(left=.08, right=.97, top=.87, bottom=.11)
    fig.savefig(args.output, dpi=200, facecolor=SURFACE)


if __name__ == '__main__':
    main()
