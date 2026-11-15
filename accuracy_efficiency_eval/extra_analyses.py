"""Offline follow-ups on saved predictions. Exploratory, like the pilot.

1. Incremental batches: start with one K=16 pass. Each step doubles the number
   of independent K=16 passes and averages their probabilities, so all earlier
   work is kept. Stopping after 1, 2, 4 or 8 passes costs 16, 32, 64 or 128.
2. Score comparison: rerun the K-escalation replay with each uncertainty score.
3. Two-stage rule: use K=16, then jump straight to K=128 for uncertain nodes.
   Escalated nodes cost 16 + 128 = 144, with every pass counted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate as ev

STEPS = (1, 2, 4, 8)
COST = np.array([16., 32., 64., 128.])
TARGETS = (32, 48, 64, 96)


def cumulative_means(p16):
    """Turn [repetition, node, class] K=16 runs into [step, group, node, class]."""
    groups = p16.shape[0] // STEPS[-1]
    if groups < 1:
        raise ValueError('Incremental replay needs at least 8 repetitions.')
    runs = p16[:groups * 8].reshape(groups, 8, *p16.shape[1:])
    return np.stack([runs[:, :n].mean(1) for n in STEPS])


def replay(steps, labels, threshold):
    """Per-node correctness and stop step for [step, group, node, class] input."""
    passed = ev.score_of(steps, 'confidence') >= threshold
    passed[-1] = True
    stop = passed.argmax(0)
    right = steps.argmax(-1) == labels
    g, n = stop.shape
    return right[stop, np.arange(g)[:, None], np.arange(n)[None, :]].mean(0), stop


def incremental(data, seed, draws):
    cal, test, mode = ev.split_nodes(data, seed)
    p16 = data['p'][data['ks'].index(16)]
    y = data['y']
    cal_steps, test_steps = cumulative_means(p16[:, cal]), cumulative_means(p16[:, test])
    cal_scores = ev.score_of(cal_steps, 'confidence')
    candidates = ev.candidate_thresholds(cal_scores)
    base = ev.fixed(data, 128, test)
    fixed_e = {k: ev.fixed(data, k, test) for k in ev.ADAPTIVE_STEPS}
    rows = []
    for target in TARGETS:
        t = ev.threshold_for_cost(cal_scores, target, COST, candidates)
        e, stop = replay(test_steps, y[test], t)
        cost = float(COST[stop].mean())
        d128 = ev.bootstrap_diff(e, base, draws, seed + 5)
        low = max(k for k in ev.ADAPTIVE_STEPS if k <= cost)
        high = min((k for k in ev.ADAPTIVE_STEPS if k > cost), default=128)
        w = 0 if high == low else (cost - low) / (high - low)
        mixed = (1 - w) * fixed_e[low] + w * fixed_e[high]
        dfixed = ev.bootstrap_diff(e, mixed, draws, seed + 6)
        rows.append(dict(target=target, threshold=float(t), cost=cost, accuracy=float(e.mean()),
                         diff_vs_fixed_128=d128[0], fixed128_low=d128[1], fixed128_high=d128[2],
                         diff_vs_fixed_same_cost=dfixed[0], fixed_cost_low=dfixed[1], fixed_cost_high=dfixed[2],
                         **{f'stop_{16 * n}': float((stop == i).mean()) for i, n in enumerate(STEPS)}))
    return mode, pd.DataFrame(rows)


def compare_scores(data, seed, draws):
    cal, test, _ = ev.split_nodes(data, seed)
    costs = ev.load_costs(data['ks'])[0]
    rerun, _ = ev.stop_costs(costs)
    base = ev.fixed(data, 128, test)
    rows = []
    for score in ev.SCORES:
        cal_scores = ev.adaptive_scores(data, cal, score)
        test_scores = ev.adaptive_scores(data, test, score)
        candidates = ev.candidate_thresholds(cal_scores)
        for target in (32, 48, 64):
            t = ev.threshold_for_cost(cal_scores, target, rerun, candidates)
            e, stop = ev.adaptive(data, test, test_scores, t)
            d = ev.bootstrap_diff(e, base, draws, seed + 7)
            rows.append(dict(score=score, target=target, cost=float(rerun[stop].mean()), accuracy=float(e.mean()),
                             diff_vs_fixed_128=d[0], ci_low=d[1], ci_high=d[2]))
    return pd.DataFrame(rows)


def two_stage(data, seed, draws):
    cal, test, _ = ev.split_nodes(data, seed)
    i16, i128 = data['ks'].index(16), data['ks'].index(128)
    conf_cal = ev.score_of(data['p'][i16][:, cal], 'confidence')
    conf_test = ev.score_of(data['p'][i16][:, test], 'confidence')
    y = data['y'][test]
    low = data['p'][i16][:, test].argmax(-1) == y
    high = data['p'][i128][:, test].argmax(-1) == y
    base = ev.fixed(data, 128, test)
    rows = []
    for target in TARGETS:
        share = (target - 16) / 128  # share of nodes that can escalate
        t = float(np.quantile(conf_cal, share))
        escalate = conf_test < t
        e = np.where(escalate, high, low).mean(0)
        d = ev.bootstrap_diff(e, base, draws, seed + 8)
        rows.append(dict(target=target, threshold=t, cost=float(16 + 128 * escalate.mean()),
                         escalated=float(escalate.mean()), accuracy=float(e.mean()),
                         diff_vs_fixed_128=d[0], ci_low=d[1], ci_high=d[2]))
    return pd.DataFrame(rows)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--predictions', required=True)
    a.add_argument('--output', default='results')
    a.add_argument('--bootstrap', type=int, default=2000)
    a.add_argument('--seed', type=int, default=20261105)
    args = a.parse_args()
    out = Path(args.output)
    data = ev.load_predictions(args.predictions)
    mode, inc = incremental(data, args.seed, args.bootstrap)
    scores = compare_scores(data, args.seed, args.bootstrap)
    stages = two_stage(data, args.seed, args.bootstrap)
    stages.to_csv(out / 'two_stage.csv', index=False)
    inc.to_csv(out / 'incremental_batches.csv', index=False)
    scores.to_csv(out / 'score_comparison.csv', index=False)
    print(mode)
    print(inc.round(4).T.to_string())
    print(scores.round(4).to_string(index=False))
    print(stages.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
