"""Accuracy-efficiency evaluation for adaptive GPM inference.

Compares fixed K, random allocation, repeated sampling and adaptive inference
on saved class probabilities with shape [K, repetition, node, class].

Two modes, chosen by the input file:
  PILOT: no `node_split` field (today's cache). Nodes are split at random into
         calibration and evaluation halves. All of them are test nodes, so
         results are exploratory.
  FINAL: the file has `node_split` with 'val' and 'test' entries. Thresholds
         use val nodes only and results use test nodes only.

Cost is "pattern-passes" by default: the patterns sent through GPM, counted
every time. Pass --timing to replace it with measured seconds per node.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ADAPTIVE_STEPS = (16, 32, 64, 128)
SCORES = ('confidence', 'margin', 'negative_entropy')
TARGETS = (32, 48, 64, 96)  # nominal pattern-passes per node


# ---------------------------------------------------------------- input
def load_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        missing = {'probabilities', 'labels', 'node_ids', 'k_values'} - set(z.files)
        if missing:
            raise ValueError(f'{path} is missing fields: {sorted(missing)}')
        data = dict(p=z['probabilities'].astype(np.float32), y=z['labels'],
                    node_ids=z['node_ids'], ks=[int(k) for k in z['k_values']],
                    node_split=z['node_split'].astype(str) if 'node_split' in z.files else None,
                    nested=bool(z['nested']) if 'nested' in z.files else False)
    p, y = data['p'], data['y']
    if p.ndim != 4 or p.shape[0] != len(data['ks']) or p.shape[2] != len(y):
        raise ValueError(f'Expected [K, repetition, node, class]; found {p.shape}.')
    if not np.isfinite(p).all() or not np.allclose(p.sum(-1), 1, atol=1e-4):
        raise ValueError('Probabilities must be finite and sum to 1 for every node.')
    if y.min() < 0 or y.max() >= p.shape[-1]:
        raise ValueError('Labels fall outside the class range.')
    absent = set(ADAPTIVE_STEPS) - set(data['ks'])
    if absent:
        raise ValueError(f'Budgets needed for adaptive inference are missing: {sorted(absent)}')
    return data


def split_nodes(data, seed):
    """Return (calibration indices, evaluation indices, mode)."""
    if data['node_split'] is not None:
        cal = np.flatnonzero(data['node_split'] == 'val')
        ev = np.flatnonzero(data['node_split'] == 'test')
        if not len(cal) or not len(ev):
            raise ValueError("node_split must contain both 'val' and 'test'.")
        return cal, ev, 'FINAL'
    order = np.random.default_rng(seed).permutation(len(data['y']))
    half = len(order) // 2
    return np.sort(order[:half]), np.sort(order[half:]), 'PILOT'


# ---------------------------------------------------------------- cost
def load_costs(ks, timing_path=None):
    """Cost of one pass at each K, divided into its reusable parts.

    Sampling and pattern encoding can be reused when K grows. The transformer
    cannot, because it attends across all of a node's patterns.
    """
    if timing_path is None:
        return {k: dict(sample=0.0, encode=float(k), transform=0.0) for k in ks}, 'pattern-passes'
    with open(timing_path) as f:
        rows = json.load(f)['per_k']
    costs = {int(r['k']): dict(sample=r['sample_s_per_node'], encode=r['encode_s_per_node'],
                               transform=r['transform_s_per_node']) for r in rows}
    absent = set(ADAPTIVE_STEPS) - set(costs)
    if absent:
        raise ValueError(f'Timing file lacks budgets: {sorted(absent)}')
    return costs, 'seconds per node'


def full(costs, k):
    return sum(costs[k].values())


def stop_costs(costs, steps=ADAPTIVE_STEPS):
    """Cost of stopping at each step: (rerun everything, reuse encodings)."""
    rerun = np.cumsum([full(costs, k) for k in steps])
    transforms = np.cumsum([costs[k]['transform'] for k in steps])
    reuse = np.array([costs[k]['sample'] + costs[k]['encode'] for k in steps]) + transforms
    return rerun, reuse


# ---------------------------------------------------------------- methods
def correct(p, y):
    return p.argmax(-1) == y


def fixed(data, k, nodes):
    i = data['ks'].index(k)
    return correct(data['p'][i][:, nodes], data['y'][nodes]).mean(0)


def random_allocation(e_low, e_high, cost_low, cost_high, target):
    """Random share of nodes gets the high budget; expected over allocations."""
    f = float(np.clip((target - cost_low) / (cost_high - cost_low), 0, 1))
    return (1 - f) * e_low + f * e_high, f


def repeated(data, k, m, nodes):
    """Average m independent K-pattern runs, using disjoint groups of runs."""
    i = data['ks'].index(k)
    p, y = data['p'][i][:, nodes], data['y'][nodes]
    groups = p.shape[0] // m
    if groups < 1:
        raise ValueError(f'{m} passes need at least {m} repetitions.')
    averaged = p[:groups * m].reshape(groups, m, *p.shape[1:]).mean(1)
    return correct(averaged, y).mean(0)


def score_of(q, name):
    """Higher means more certain."""
    if name == 'confidence':
        return q.max(-1)
    if name == 'margin':
        top = np.partition(q, -2, axis=-1)
        return top[..., -1] - top[..., -2]
    if name == 'negative_entropy':
        return (q * np.log(np.clip(q, 1e-15, 1))).sum(-1)
    raise ValueError(f'Unknown score: {name}')


def adaptive_scores(data, nodes, name):
    """Scores with shape [step, repetition, node]."""
    return np.stack([score_of(data['p'][data['ks'].index(k)][:, nodes], name)
                     for k in ADAPTIVE_STEPS])


def adaptive(data, nodes, scores, threshold):
    """Replay the stopping rule. Returns per-node correctness and stop step.

    A node stops at the first step whose score reaches the threshold, and
    always stops at the last step. The result is shape [repetition, node].
    """
    passed = scores >= threshold
    passed[-1] = True
    step = passed.argmax(0)
    all_correct = np.stack([correct(data['p'][data['ks'].index(k)][:, nodes], data['y'][nodes])
                            for k in ADAPTIVE_STEPS])
    reps, n = step.shape
    final = all_correct[step, np.arange(reps)[:, None], np.arange(n)[None, :]]
    return final.mean(0), step


def threshold_for_cost(scores, target, rerun, candidates):
    """Largest threshold that stays within target. Uses no labels."""
    best = candidates[0]
    for t in candidates:
        passed = scores >= t
        passed[-1] = True
        if rerun[passed.argmax(0)].mean() <= target:
            best = t
    return best


def threshold_for_accuracy(data, nodes, scores, gap, rerun, candidates):
    """Cheapest threshold within `gap` of fixed K=128 on calibration nodes."""
    limit = fixed(data, 128, nodes).mean() - gap
    choice = None
    for t in candidates:
        e, step = adaptive(data, nodes, scores, t)
        if e.mean() >= limit:
            cost = rerun[step].mean()
            if choice is None or cost < choice[1]:
                choice = (t, cost)
    return None if choice is None else choice[0]


def candidate_thresholds(scores, n=400):
    return np.unique(np.quantile(scores[:-1], np.linspace(0, 1, n)))


# ---------------------------------------------------------------- statistics
def bootstrap_diff(a, b, draws, seed):
    """Paired node bootstrap of mean(a - b)."""
    rng = np.random.default_rng(seed)
    d = a - b
    picks = rng.integers(0, len(d), (draws, len(d)))
    return float(d.mean()), *[float(v) for v in np.quantile(d[picks].mean(1), [.025, .975])]


# ---------------------------------------------------------------- decisions from the real implementation
def load_adaptive_decisions(path, labels, node_ids, costs):
    """Read per-node results from the live adaptive script.

    Required fields: node_ids [n], final_k [repetition, n], predicted [repetition, n].
    Optional: pattern_passes [repetition, n], the work each node really used,
    and seconds_total, the measured time of the whole run. Recorded
    pattern_passes always override costs inferred from final_k, because
    different rules reach the same final K at different costs.
    """
    with np.load(path, allow_pickle=False) as z:
        missing = {'node_ids', 'final_k', 'predicted'} - set(z.files)
        if missing:
            raise ValueError(f'{path} is missing fields: {sorted(missing)}')
        ids, final_k, predicted = z['node_ids'], z['final_k'], z['predicted']
        seconds = float(z['seconds_total']) if 'seconds_total' in z.files else None
        passes = z['pattern_passes'] if 'pattern_passes' in z.files else None
        sampled = z['sampled_patterns'] if 'sampled_patterns' in z.files else None
        reused = bool(z['encoding_reuse']) if 'encoding_reuse' in z.files else None
        threshold = float(z['threshold']) if 'threshold' in z.files else None
    lookup = {int(n): i for i, n in enumerate(node_ids)}
    unknown = [int(n) for n in ids if int(n) not in lookup]
    if unknown:
        raise ValueError(f'{len(unknown)} decision nodes are not in the predictions.')
    positions = np.array([lookup[int(n)] for n in ids])
    y = labels[positions]
    if final_k.shape != predicted.shape or final_k.ndim != 2 or final_k.shape[1] != len(ids):
        raise ValueError('final_k and predicted must both be [repetition, node].')
    bad = set(np.unique(final_k).tolist()) - set(ADAPTIVE_STEPS)
    if bad:
        raise ValueError(f'Unexpected final K values: {sorted(bad)}')
    if passes is not None and passes.shape != final_k.shape:
        raise ValueError('pattern_passes must match final_k.')
    rerun, reuse = stop_costs(costs)
    step = np.searchsorted(ADAPTIVE_STEPS, final_k)
    per_node = (predicted == y).mean(0)
    out = dict(accuracy=float((predicted == y).mean()), mean_final_k=float(final_k.mean()),
               rerun_cost=float(passes.mean()) if passes is not None else float(rerun[step].mean()),
               reuse_cost=None if passes is not None and not reused else float(reuse[step].mean()),
               sampled_patterns=None if sampled is None else float(sampled.mean()),
               threshold=threshold, seconds_total=seconds,
               seconds_per_node=None if seconds is None else seconds / final_k.size,
               stop_share={str(k): float((final_k == k).mean()) for k in ADAPTIVE_STEPS if (final_k == k).any()})
    return out, per_node, positions


def compare_live(per_node, positions, data, cost, draws, seed):
    """Paired-by-node comparisons. Walks differ between runs, so these are not same-walk pairs."""
    fixed_e = {k: fixed(data, k, positions) for k in ADAPTIVE_STEPS}
    low = max(k for k in ADAPTIVE_STEPS if k <= cost)
    high = min(k for k in ADAPTIVE_STEPS if k >= cost)
    w = 0 if high == low else (cost - low) / (high - low)
    interpolated = (1 - w) * fixed_e[low] + w * fixed_e[high]
    random_e = random_allocation(fixed_e[16], fixed_e[128], 16, 128, cost)[0]
    rows = {'fixed_128': fixed_e[128], f'fixed_{low}_{high}_interpolated': interpolated, 'random_same_cost': random_e}
    return {name: dict(zip(('diff', 'ci_low', 'ci_high'), bootstrap_diff(per_node, base, draws, seed)))
            for name, base in rows.items()}


# ---------------------------------------------------------------- full evaluation
def run(data, costs, seed=20261105, draws=2000, score='confidence', gap=.005):
    cal, ev, mode = split_nodes(data, seed)
    rerun, reuse = stop_costs(costs)
    cost_16, cost_128 = full(costs, 16), full(costs, 128)
    unit_scale = full(costs, 128) / 128  # nominal target -> this cost unit

    base = fixed(data, 128, ev)
    curve, comparisons = [], []

    def add(method, label, cost, e, extra=None):
        diff, low, high = bootstrap_diff(e, base, draws, seed + 1)
        curve.append(dict(method=method, setting=label, cost=cost, accuracy=float(e.mean()),
                          diff_vs_fixed_128=diff, ci_low=low, ci_high=high, **(extra or {})))

    fixed_e = {k: fixed(data, k, ev) for k in data['ks']}
    for k in data['ks']:
        add('fixed', f'K={k}', full(costs, k), fixed_e[k])
    for m, k in [(2, 16), (4, 16), (8, 16), (2, 32), (4, 32), (2, 64)]:
        if data['p'].shape[1] >= m and k in data['ks']:
            add('repeated', f'{m} passes at K={k}', m * full(costs, k), repeated(data, k, m, ev))

    cal_scores = adaptive_scores(data, cal, score)
    ev_scores = adaptive_scores(data, ev, score)
    candidates = candidate_thresholds(cal_scores)
    for target in TARGETS:
        goal = target * unit_scale
        e, f = random_allocation(fixed_e[16], fixed_e[128], cost_16, cost_128, goal)
        add('random', f'{f:.1%} of nodes at K=128', goal, e, dict(target=target))
        t = threshold_for_cost(cal_scores, goal, rerun, candidates)
        e_a, step = adaptive(data, ev, ev_scores, t)
        add('adaptive', f'threshold {t:.4f}', float(rerun[step].mean()), e_a,
            dict(reuse_cost=float(reuse[step].mean()), threshold=float(t), target=target))
        random_at_cost = random_allocation(fixed_e[16], fixed_e[128], cost_16, cost_128, rerun[step].mean())[0]
        d, low, high = bootstrap_diff(e_a, random_at_cost, draws, seed + 2)
        comparisons.append(dict(target=target, baseline='random', adaptive_cost=float(rerun[step].mean()),
                                diff=d, ci_low=low, ci_high=high))
        for k_low, k_high in zip(ADAPTIVE_STEPS, ADAPTIVE_STEPS[1:]):
            c_low, c_high = full(costs, k_low), full(costs, k_high)
            actual = rerun[step].mean()
            if c_low <= actual <= c_high:
                w = (actual - c_low) / (c_high - c_low)
                mixed = (1 - w) * fixed_e[k_low] + w * fixed_e[k_high]
                d, low, high = bootstrap_diff(e_a, mixed, draws, seed + 3)
                comparisons.append(dict(target=target, baseline=f'fixed K={k_low}-{k_high} interpolated',
                                        adaptive_cost=float(actual), diff=d, ci_low=low, ci_high=high))

    within = threshold_for_accuracy(data, cal, cal_scores, gap, rerun, candidates)
    success = None
    if within is not None:
        e_a, step = adaptive(data, ev, ev_scores, within)
        d, low, high = bootstrap_diff(e_a, base, draws, seed + 4)
        success = dict(threshold=float(within), accuracy=float(e_a.mean()),
                       fixed_128_accuracy=float(base.mean()), diff=d, ci_low=low, ci_high=high,
                       rerun_cost=float(rerun[step].mean()), reuse_cost=float(reuse[step].mean()),
                       fixed_128_cost=cost_128,
                       stop_share={str(k): float((step == i).mean()) for i, k in enumerate(ADAPTIVE_STEPS)})

    return dict(mode=mode, score=score, n_calibration=len(cal), n_evaluation=len(ev), nested=data['nested'],
                curve=pd.DataFrame(curve), comparisons=pd.DataFrame(comparisons), success=success)


def split_sensitivity(data, costs, splits=20, target=64, score='confidence'):
    """Repeat threshold choice across node splits (pilot mode only)."""
    if data['node_split'] is not None:
        return None
    rerun, reuse = stop_costs(costs)
    goal = target * full(costs, 128) / 128
    rows = []
    for s in range(splits):
        cal, ev, _ = split_nodes(data, 1000 + s)
        cs, es = adaptive_scores(data, cal, score), adaptive_scores(data, ev, score)
        t = threshold_for_cost(cs, goal, rerun, candidate_thresholds(cs, 200))
        e, step = adaptive(data, ev, es, t)
        rows.append(dict(split=s, accuracy=e.mean(), fixed_128=fixed(data, 128, ev).mean(),
                         gap=e.mean() - fixed(data, 128, ev).mean(), rerun_cost=rerun[step].mean(),
                         reuse_cost=reuse[step].mean()))
    return pd.DataFrame(rows)


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--predictions', required=True)
    a.add_argument('--output', default='results')
    a.add_argument('--timing', help='timing.json from measure_timing.py')
    a.add_argument('--adaptive-decisions', action='append', default=[], metavar='[NAME=]PATH',
                   help='per-node results from the live adaptive script; repeat for each rule')
    a.add_argument('--score', default='confidence', choices=SCORES)
    a.add_argument('--bootstrap', type=int, default=2000)
    a.add_argument('--seed', type=int, default=20261105)
    args = a.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.predictions)
    costs, unit = load_costs(data['ks'], args.timing)
    result = run(data, costs, args.seed, args.bootstrap, args.score)
    result['curve'].to_csv(out / 'curve.csv', index=False)
    result['comparisons'].to_csv(out / 'comparisons.csv', index=False)
    summary = {k: result[k] for k in ('mode', 'score', 'n_calibration', 'n_evaluation', 'nested', 'success')}
    summary.update(cost_unit=unit, seed=args.seed, bootstrap=args.bootstrap, predictions=str(args.predictions))
    sensitivity = split_sensitivity(data, costs, score=args.score)
    if sensitivity is not None:
        sensitivity.to_csv(out / 'split_sensitivity.csv', index=False)
        summary['split_sensitivity'] = dict(
            gap_mean=float(sensitivity.gap.mean()), gap_min=float(sensitivity.gap.min()),
            gap_max=float(sensitivity.gap.max()), rerun_cost_mean=float(sensitivity.rerun_cost.mean()))
    for item in args.adaptive_decisions:
        name, path = item.split('=', 1) if '=' in item else (Path(item).stem, item)
        live, per_node, positions = load_adaptive_decisions(path, data['y'], data['node_ids'], costs)
        live['comparisons'] = compare_live(per_node, positions, data, live['rerun_cost'], args.bootstrap, args.seed + 9)
        summary.setdefault('live_adaptive', {})[name] = live
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
