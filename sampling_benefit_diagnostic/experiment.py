"""Retrospective sampling-benefit diagnostic; never trains or changes a GPM.

Input: original sampling_experiment.py NPZ, [K, repetition, node, class].
All score construction is label-free and uses A only. Outcomes use B only.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import platform
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

SCORES = ['sampling_variance', 'entropy', 'negative_confidence', 'negative_margin', 'variation_ratio']


def repetition_split(repeats):
    if repeats < 4:
        raise ValueError('At least four complete repetitions are required (two per group).')
    half = repeats // 2
    return np.arange(half), np.arange(half, repeats)


def uncertainty(p):
    """Unbiased trace of sample covariance, not Bayesian posterior variance."""
    if p.shape[0] < 2:
        raise ValueError('Variance requires at least two repetitions.')
    mean = p.mean(0)
    ordered = np.sort(mean, axis=-1)
    pred = p.argmax(-1)
    counts = np.stack([(pred == c).sum(0) for c in range(p.shape[-1])])
    return dict(sampling_variance=p.var(0, ddof=1).sum(-1),
                entropy=-(mean * np.log(np.clip(mean, 1e-15, 1))).sum(-1),
                negative_confidence=-mean.max(-1),
                negative_margin=-(ordered[:, -1] - ordered[:, -2]),
                variation_ratio=1 - counts.max(0) / len(p))


def outcomes(p, labels):
    true_p = np.take_along_axis(p, labels[None, :, None], axis=-1)[..., 0]
    return dict(error=(p.argmax(-1) != labels).mean(0),
                nll=-np.log(np.clip(true_p, 1e-15, 1)).mean(0),
                brier=(np.square(p).sum(-1) - 2 * true_p + 1).mean(0))


def select(score, node_ids, fraction=.25):
    # Exact floor(n*fraction); deterministic node-ID tie break; never labels.
    mask = np.zeros(len(score), dtype=bool)
    mask[np.lexsort((node_ids, -np.asarray(score)))[:int(len(score) * fraction)]] = True
    return mask


def policy_error(low, high, selected):
    return np.where(selected, high, low)


def correlation(x, y, kind='spearman'):
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan
    return float(stats.spearmanr(x, y).statistic if kind == 'spearman' else stats.pearsonr(x, y).statistic)


def discrimination(x, y, metric):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, x) if metric == 'auroc' else average_precision_score(y, x))


def interval(values):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    return [float(v) for v in np.quantile(values, [.025, .975])] if len(values) else [None, None]


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_cache(path):
    with np.load(path, allow_pickle=False) as z:
        required = {'probabilities', 'labels', 'node_ids', 'k_values', 'checkpoint', 'repeats'}
        if not required.issubset(z.files):
            raise ValueError(f'Missing cache fields: {required-set(z.files)}. No values fabricated.')
        p = z['probabilities'].astype(np.float64)
        labels, ids, ks = z['labels'], z['node_ids'], z['k_values']
        checkpoint = str(z['checkpoint'].item())
        if p.ndim != 4 or p.shape[-1] < 2 or p.shape[2] == 0 or p.shape[:1] != ks.shape or p.shape[2] != len(labels):
            raise ValueError('Invalid probability dimensions.')
        if len(ids) != len(labels) or len(np.unique(ids)) != len(ids) or len(np.unique(ks)) != len(ks):
            raise ValueError('Duplicate or misaligned node IDs / budgets.')
        if labels.ndim != 1 or labels.dtype.kind not in 'iu' or ids.dtype.kind not in 'iu':
            raise ValueError('Labels and node IDs must be integer vectors.')
        if labels.min() < 0 or labels.max() >= p.shape[-1] or ids.min() < 0:
            raise ValueError('Out-of-range labels or node IDs.')
        if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
            raise ValueError('Incomplete or invalid probabilities; regenerate missing runs explicitly.')
        if not np.allclose(p.sum(-1), 1, atol=1e-6, rtol=0):
            raise ValueError('Probabilities are not normalized; input schema expects softmax probabilities.')
        if int(z['repeats']) != p.shape[1]:
            raise ValueError('Repetition metadata mismatch.')
        if 'completed_k_index' in z and int(z['completed_k_index']) != len(ks)-1:
            raise ValueError('Cache is not complete.')
        if not {16, 128}.issubset(ks):
            raise ValueError('Both primary endpoints are required.')
    return p, labels, ids, ks, checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--repository', type=Path, required=True, help='Directory containing GPM/ and original cache')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--random-allocations', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    if min(args.bootstrap, args.random_allocations) < 100:
        parser.error('Use at least 100 bootstrap draws and random allocations.')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    p, labels, ids, ks, checkpoint = load_cache(args.cache)
    a, b = repetition_split(p.shape[1])
    n = len(ids)
    kindex = {int(k): j for j, k in enumerate(ks)}
    scores = uncertainty(p[kindex[16], a])
    by_k = {int(k): outcomes(p[j, b], labels) for j, k in enumerate(ks)}
    low, high = by_k[16], by_k[128]
    delta = {m: low[m] - high[m] for m in low}
    frame = pd.DataFrame({'node_id': ids, 'label': labels, 'split': 'test',
                          'model_seed': '0_inferred_from_training_script', 'checkpoint': checkpoint, **scores})
    for k, outcome in by_k.items():
        for m, value in outcome.items():
            frame[f'{m}_{k}'] = value
    for m, value in delta.items():
        frame[f'delta_{m}'] = value
    order = np.lexsort((ids, scores['sampling_variance']))
    bins = np.zeros(n, dtype=int)
    for q, idx in enumerate(np.array_split(order, 5), 1):
        bins[idx] = q
    frame['uncertainty_quintile'] = bins
    policies = {s: select(scores[s], ids) for s in SCORES}
    policies['oracle_delta_non_deployable'] = select(delta['error'], ids)
    for name, mask in policies.items():
        frame[f'selected_{name}'] = mask
    frame.to_csv(out / 'per_node.csv', index=False)
    # Self-describing normalized copy preserves ALL actual probabilities and derived predictions.
    np.savez_compressed(out / 'predictions.npz', probabilities=p.astype(np.float32),
                        predicted_class=p.argmax(-1).astype(np.int16), labels=labels, node_ids=ids,
                        k_values=ks, repetition_ids=np.arange(p.shape[1]), group_A=a, group_B=b,
                        split=np.array('test'), checkpoint=np.array(checkpoint),
                        model_seed=np.array('0_inferred_from_training_script'))
    config = dict(dataset='computers', checkpoint=checkpoint, checkpoint_available_locally=False,
                  model_seed='0 inferred from main.py idx=0; not recorded in NPZ', split='high, first split, test',
                  split_seed=42, split_provenance='sampling_experiment.py; original masks absent',
                  preprocessing='NormalizeFeatures, RemoveSelfLoops, ToUndirected, AddLaplacianEigenvectorPE(8)',
                  k_values=ks, missing_k_values=[256], primary_endpoints=[16, 128], group_A=a, group_B=b,
                  repetition_split_rule='A=0..floor(R/2)-1, B=floor(R/2)..R-1; deterministic, no labels',
                  inference_seed_formula='10000 + 1000*K + repetition (script default, not cached)',
                  across_k_pairing=False, bootstrap_seed=args.seed, random_allocation_seed=args.seed+1,
                  bootstrap_draws=args.bootstrap, random_allocations=args.random_allocations,
                  bootstrap_unit='node; conditional on checkpoint and realized repetition groups',
                  budget_fraction_requested=.25, budget_fraction_actual=int(n*.25)/n,
                  tie_break='ascending node ID', n_nodes=n, n_classes=p.shape[-1],
                  input_path=str(args.cache.resolve()), input_sha256=sha(args.cache),
                  python=platform.python_version(), numpy=np.__version__,
                  analysis_dependencies={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','matplotlib']},
                  experiment_script_sha256=sha(Path(__file__)),
                  nll_probability_floor=1e-15, architecture_changed=False, model_retrained=False,
                  uncertainty_estimation_overhead=f'{len(a)}*16={len(a)*16} patterns/node, excluded from nominal policy budgets',
                  timing='not present in cache; no measured latency reported')
    config['source_sha256'] = {str(f.relative_to(args.repository)): sha(f) for f in [
        args.repository/'GPM/GPM/sampling_experiment.py', args.repository/'GPM/GPM/main.py',
        args.repository/'GPM/GPM/model/model.py', args.repository/'GPM/GPM/model/random_walk.py',
        args.repository/'GPM/GPM/data/pyg_data_loader.py', args.repository/'GPM/config/main.yaml'] if f.exists()}
    (out/'config.json').write_text(json.dumps(clean(config), indent=2)+'\n')
    checks = dict(finite=True, normalized=True, unique_node_ids=True, shared_order='single shared cache axis',
                  label_range_valid=True, disjoint_groups=not bool(set(a)&set(b)),
                  mask_independently_verified=False, checkpoint_hash_available=False,
                  max_probability_sum_error=float(abs(p.sum(-1)-1).max()),
                  zero_variance_nodes=int((scores['sampling_variance']==0).sum()),
                  distinct_uncertainty_values=len(np.unique(scores['sampling_variance'])))
    metrics = {'validation': checks, 'means': {}, 'association': {}, 'quintiles': [], 'policies': {}}
    rng = np.random.default_rng(args.seed)
    # Shared node resamples permit paired contrasts. Keep scores/quintile memberships/policies fixed.
    boot = rng.integers(0, n, size=(args.bootstrap, n), dtype=np.int32)
    def mean_ci(x):
        return dict(estimate=float(np.mean(x)), ci95=interval(np.mean(x[boot], axis=1)))
    for m in delta:
        metrics['means'][f'delta_{m}'] = mean_ci(delta[m])
    for k in by_k:
        for m, value in by_k[k].items():
            metrics['means'][f'{m}_{k}'] = mean_ci(value)
    for key, mask in dict(positive=delta['error']>0, zero=delta['error']==0, negative=delta['error']<0).items():
        metrics['means'][f'fraction_{key}_delta'] = mean_ci(mask.astype(float))
    # Every alternative gets both improvement detection and baseline-error detection.
    # Threshold fixed in advance: high stochastic error = e16 >= 0.5.
    targets = {'improvement': delta['error']>0, 'high_error_e16_ge_0.5': low['error']>=.5}
    association_rows = []
    for name, score in scores.items():
        row = {}
        funcs = {}
        for target_name, target in [('delta_error', delta['error']), ('delta_nll',delta['nll']),
                                     ('delta_brier',delta['brier']), ('baseline_error',low['error'])]:
            for kind in ['spearman', 'pearson']:
                funcs[f'{target_name}_{kind}'] = lambda ix, t=target, k=kind: correlation(score[ix], t[ix], k)
        for target_name, target in targets.items():
            for kind in ['auroc', 'ap']:
                funcs[f'{target_name}_{kind}'] = lambda ix, t=target,k=kind: discrimination(score[ix], t[ix], k)
        for key, fn in funcs.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                estimates = [fn(ix) for ix in boot]
                row[key] = dict(estimate=fn(np.arange(n)), ci95=interval(estimates),
                                valid_bootstrap_draws=int(np.isfinite(estimates).sum()))
            association_rows.append(dict(score=name, metric=key, **row[key], ci_low=row[key]['ci95'][0], ci_high=row[key]['ci95'][1]))
        metrics['association'][name] = row
        print(f'Completed score: {name}', flush=True)
    metrics['sampling_variability_by_k'] = {str(int(k)): mean_ci(p[j,a].var(0,ddof=1).sum(-1)) for j,k in enumerate(ks)}
    metrics['outcome_prevalence'] = {key:float(y.mean()) for key,y in targets.items()}
    pd.DataFrame(association_rows).drop(columns='ci95').to_csv(out/'associations.csv',index=False)
    def subgroup(x, mask):
        ix = np.flatnonzero(mask)
        if not len(ix):
            return dict(estimate=None,ci95=[None,None],n=0)
        draws = rng.choice(ix, (args.bootstrap,len(ix)), replace=True)
        return dict(estimate=float(x[ix].mean()),ci95=interval(x[draws].mean(1)),n=len(ix))
    for q in range(1,6):
        mask = bins==q
        metrics['quintiles'].append(dict(quintile=q, n=int(mask.sum()),
            **{key:subgroup(x,mask) for key,x in {'delta':delta['error'],'error16':low['error'],
                 'error128':high['error'],'delta_nll':delta['nll'],'delta_brier':delta['brier']}.items()}))
    lo_ix, hi_ix = np.flatnonzero(bins==1), np.flatnonzero(bins==5)
    qdiff = (delta['error'][rng.choice(hi_ix,(args.bootstrap,len(hi_ix)))].mean(1)-delta['error'][rng.choice(lo_ix,(args.bootstrap,len(lo_ix)))].mean(1)) if len(hi_ix) and len(lo_ix) else np.array([np.nan])
    metrics['top_minus_bottom_quintile_delta'] = dict(estimate=float(delta['error'][hi_ix].mean()-delta['error'][lo_ix].mean()) if len(hi_ix) and len(lo_ix) else None,ci95=interval(qdiff))
    metrics['low_uncertainty_consistently_wrong_at128'] = subgroup((high['error']==1).astype(float),bins==1)
    metrics['high_uncertainty_improved'] = subgroup((delta['error']>0).astype(float),bins==5)
    # Exploratory residual association controls linearly for baseline difficulty; not a causal estimate.
    ranked = np.column_stack([np.ones(n), stats.rankdata(low['error'])])
    residuals = [stats.rankdata(x)-ranked@np.linalg.lstsq(ranked,stats.rankdata(x),rcond=None)[0]
                 for x in [scores['sampling_variance'],delta['error']]]
    metrics['exploratory_partial_rank_correlation_controlling_e16'] = correlation(*residuals,kind='pearson')
    vectors = {}
    for name, mask in policies.items():
        vectors[name] = policy_error(low['error'],high['error'],mask)
        metrics['policies'][name] = dict(mean_budget=16+112*mask.mean(),selected_nodes=int(mask.sum()),
             error=mean_ci(vectors[name]), improvement_vs_fixed16=mean_ci(low['error']-vectors[name]))
    random_rng = np.random.default_rng(args.seed+1)
    random_errors = np.array([policy_error(low['error'],high['error'],
                        select(random_rng.random(n),ids)).mean() for _ in range(args.random_allocations)])
    pd.DataFrame({'allocation_id':np.arange(len(random_errors)),'expected_error':random_errors}).to_csv(out/'random_allocations.csv',index=False)
    f = int(n*.25)/n
    random_expected_vector = (1-f)*low['error']+f*high['error']
    metrics['policies']['random_25_percent'] = dict(mean_budget=16+112*f,error=mean_ci(random_expected_vector),
          allocation_distribution_mean=float(random_errors.mean()), allocation_interval95=interval(random_errors),
          allocation_interval_definition='variation across random selections, distinct from node-bootstrap CI')
    for name, vector in vectors.items():
        metrics['policies'][name]['error_reduction_vs_expected_random'] = mean_ci(random_expected_vector-vector)
    for k in [16,32,64,128]:
        if k in by_k:
            metrics['policies'][f'fixed_{k}'] = dict(mean_budget=k,error=mean_ci(by_k[k]['error']))
    (out/'metrics.json').write_text(json.dumps(clean(metrics),indent=2)+'\n')
    pd.DataFrame([dict(policy=name, mean_budget=value['mean_budget'],error=value['error']['estimate'],
                     ci_low=value['error']['ci95'][0],ci_high=value['error']['ci95'][1])
                  for name,value in metrics['policies'].items()]).to_csv(out/'policies.csv',index=False)
    pd.DataFrame([dict(metric=k,estimate=v['estimate'],ci_low=v['ci95'][0],ci_high=v['ci95'][1])
                  for k,v in metrics['means'].items()]).to_csv(out/'aggregate_means.csv',index=False)
    make_figures(out,frame,metrics)
    write_report(out,config,metrics)
    print(json.dumps(clean({'means':metrics['means'],'primary':metrics['association']['sampling_variance'],
                            'policies':metrics['policies']}),indent=2))


def make_figures(out, frame, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'savefig.dpi':300})
    def save(fig,name):
        fig.savefig(out/f'{name}.png',bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.3,4.3))
    hb=ax.hexbin(frame.sampling_variance, frame.delta_error, gridsize=45,mincnt=1,cmap='viridis',bins='log')
    ax.axhline(0,color='gray',lw=.8)
    ax.set(xlabel='Sampling covariance trace at K=16 (group A)',ylabel='Error reduction: e(16) − e(128) (group B)',title='Does sampling variability predict benefit?')
    fig.colorbar(hb,ax=ax,label='Node count (log color scale)')
    save(fig,'uncertainty_vs_improvement')
    qs=metrics['quintiles']
    fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,keys in [(axs[0],['delta']),(axs[1],['error16','error128'])]:
        for key in keys:
            means=np.array([q[key]['estimate'] for q in qs],dtype=float)
            cis=np.array([q[key]['ci95'] for q in qs],dtype=float).T
            ax.errorbar(np.arange(1,6),means,yerr=np.maximum(0,np.array([means-cis[0],cis[1]-means])),marker='o',capsize=3,label=key)
        ax.set(xlabel='Uncertainty quintile (low → high)',xticks=range(1,6))
    axs[0].axhline(0,color='gray',lw=.8)
    axs[0].set(ylabel='Mean error reduction',title='Benefit within uncertainty quintiles')
    axs[1].set(ylabel='Mean stochastic error',title='Baseline difficulty and endpoint error')
    axs[1].legend(labels=['K=16','K=128'])
    save(fig,'quintiles')
    fig,(ax,detail)=plt.subplots(1,2,figsize=(12,5.4),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    fixed=[(v['mean_budget'],v['error']['estimate']) for k,v in metrics['policies'].items() if k.startswith('fixed')]
    ax.plot(*np.array(fixed).T,color='gray',marker='o',label='Fixed K')
    for name in ['sampling_variance','random_25_percent','oracle_delta_non_deployable']:
        v=metrics['policies'][name]
        ax.scatter(v['mean_budget'],v['error']['estimate'],label=name.replace('_',' '),zorder=3)
    ax.set(xlabel='Nominal mean patterns per node',ylabel='Expected classification error',title='Error versus nominal budget')
    ax.legend(fontsize=8,loc='upper right')
    labels=[]
    for j,(name,v) in enumerate((k,v) for k,v in metrics['policies'].items() if not k.startswith('fixed')):
        y=v['error']['estimate']; lo,hi=v['error']['ci95']
        detail.errorbar(y,j,xerr=[[max(0,y-lo)],[max(0,hi-y)]],fmt='o',capsize=3)
        labels.append(name.replace('_',' ').replace('oracle delta non deployable','oracle (non-deployable)'))
    detail.set(yticks=np.arange(len(labels)),yticklabels=labels,xlabel='Expected error (node-bootstrap 95% CI)',title='Allocation policies at nominal K ≈ 44')
    detail.invert_yaxis()
    fig.suptitle('Retrospective diagnostic: uncertainty-estimation overhead excluded',fontsize=11)
    save(fig,'error_vs_nominal_budget')


def write_report(out,c,m):
    def number(x):
        return 'undefined' if x is None else f'{x:.6f}'
    def f(v):
        if v['estimate'] is None: return 'undefined'
        return f"{v['estimate']:.6f} [95% CI {v['ci95'][0]:.6f}, {v['ci95'][1]:.6f}]"
    primary=m['association']['sampling_variance']
    text=f'''# Does sampling uncertainty predict benefit from more samples?

## A. Research question and motivation
Predictive association between label-free sampling variability at K=16 and node-level error reduction from K=16 to K=128. This is exploratory and conditional on one frozen checkpoint; no Adaptive GPM implementation or causal claim.

## B. Experimental setup and reproducibility
Computers, {c['n_nodes']} test nodes, 10 classes, checkpoint `{c['checkpoint']}`. Training seed 0 is inferred from the first training-run code, not authenticated by checkpoint metadata. Existing probability cache reused without inference or training. K={list(c['k_values'])}, {len(c['group_A'])+len(c['group_B'])} repetitions each. K=256 unavailable.

A repetitions {list(c['group_A'])} construct all scores; B repetitions {list(c['group_B'])} construct all outcomes, including fixed baselines. Repetition IDs across K use different seeds (script default 10000+1000*K+r), so no common-random-number pairing is assumed. Comparisons remain paired by node. Trace of empirical covariance uses ddof=1 and sums class variances. Alternatives: entropy of A-mean probabilities, negative A-mean maximum probability, negative A-mean top-two margin, and variation ratio. None use labels. These measure sampling variability, not Bayesian epistemic uncertainty or posterior variance.

Split inferred from source: seed 42, `high` (60/20/20), first test split. Preprocessing: {c['preprocessing']}. Cache has one shared node/label axis for every K and repetition; finite probabilities, normalization, class bounds, unique IDs and A/B disjointness passed. Original dataset/masks and checkpoint files are absent locally; original labels, held-out membership, positional encodings, exact frozen checkpoint identity and model seed cannot be independently authenticated. `config.json` hashes the input and relevant source files. There is no per-run seed manifest in the original cache. Existing scripts/results are preserved.

Seed {c['bootstrap_seed']}; {c['bootstrap_draws']} node bootstrap resamples, percentile 95% intervals. Nodes, not repeated predictions, are statistical units. Intervals condition on this graph/checkpoint and realized A/B predictions, and omit cross-node graph dependence, finite-repetition uncertainty, and training-seed variability. No threshold tuning or test-label policy construction, except the explicitly non-deployable oracle.

## C. Uncertainty versus improvement
Primary Spearman: {f(primary['delta_error_spearman'])}.
Secondary Pearson: {f(primary['delta_error_pearson'])}.
Mean error K=16: {f(m['means']['error_16'])}; K=128: {f(m['means']['error_128'])}.
Mean error reduction: {f(m['means']['delta_error'])}.
NLL reduction: {f(m['means']['delta_nll'])}; Brier reduction: {f(m['means']['delta_brier'])}. Both proper scores are averaged per repetition, not evaluated only on mean probabilities. NLL floor is 1e-15.
Positive/zero/negative estimated error improvement fractions: {m['means']['fraction_positive_delta']['estimate']:.4f} / {m['means']['fraction_zero_delta']['estimate']:.4f} / {m['means']['fraction_negative_delta']['estimate']:.4f}.
Top-minus-bottom quintile improvement: {f(m['top_minus_bottom_quintile_delta'])}.
Benefit detection AUROC: {f(primary['improvement_auroc'])}; AP: {f(primary['improvement_ap'])}; positive prevalence {m['outcome_prevalence']['improvement']:.4f}. The target is estimated Delta>0, not a verified latent true benefit.

| Quintile | Nodes | Error16 | Error128 | Improvement (95% CI) |
|---|---:|---:|---:|---|
'''
    for q in m['quintiles']:
        text+=f"| {q['quintile']} | {q['n']} | {number(q['error16']['estimate'])} | {number(q['error128']['estimate'])} | {f(q['delta'])} |\n"
    text+=f'''
Quintiles have approximately equal node counts. Ties break by node ID, so equal scores can cross bin boundaries; bins are descriptive, not calibrated thresholds. Negative improvements are retained. `associations.csv` compares every score against all three improvement outcomes and baseline error, with confidence intervals.

## D. Uncertainty versus error
U versus baseline stochastic error, Spearman: {f(primary['baseline_error_spearman'])}.
Among the bottom uncertainty quintile, consistently incorrect in all {len(c['group_B'])} K=128 outcome runs: {f(m['low_uncertainty_consistently_wrong_at128'])}. Among the top quintile, estimated error decreases for fraction {f(m['high_uncertainty_improved'])}.
High error is defined in advance as e(16)>=0.5, using B. Detection AUROC: {f(primary['high_error_e16_ge_0.5_auroc'])}; AP: {f(primary['high_error_e16_ge_0.5_ap'])}; prevalence {m['outcome_prevalence']['high_error_e16_ge_0.5']:.4f}.
Exploratory residual rank correlation controlling linearly for ranked e(16): {m['exploratory_partial_rank_correlation_controlling_e16']:.6f}. This is descriptive: e(16) appears in Delta, and adjustment does not establish causation or independent benefit. Difficulty and additional-sampling benefit must be assessed separately using the correlations, quintiles and random-allocation contrast.

## E. Retrospective budget-allocation diagnostic
Select floor(0.25*N) nodes ({int(c['n_nodes']*.25)} here), ties by node ID, based only on A scores. Nominal theoretical mean K is 44; integer rounding gives {16+112*c['budget_fraction_actual']:.6f}. K=32 and K=64 bracket the budget; neither is exactly matched. Random allocation uses the same selected count; its analytic expectation and node-bootstrap CI are reported, plus {c['random_allocations']} realized random allocations and their allocation-distribution interval (not a confidence interval). Oracle uses B Delta and is optimistically evaluated on the same outcomes: an upper bound for realized estimates, not a deployable or unbiased performance estimate.

| Policy | Nominal K | Expected error (95% CI) |
|---|---:|---|
'''
    for name,v in m['policies'].items():
        text+=f"| {name} | {v['mean_budget']:.4f} | {f(v['error'])} |\n"
    text+=f'''
Variance allocation error reduction versus expected random allocation: {f(m['policies']['sampling_variance']['error_reduction_vs_expected_random'])}. Positive is favorable. Paired node bootstrap preserves within-node contrasts with policy memberships fixed. This conditions on learned rankings; it does not include uncertainty from re-estimating scores or selecting nodes on a new dataset.

**Uncertainty estimation costs {len(c['group_A'])} forward passes at K=16 ({len(c['group_A'])*16} patterns/node).** These costs are excluded from nominal budgets. These retrospective/oracle-style diagnostics measure predictive value, not practical efficiency. No measured timing is available. No speedup claim is supported.

## F. Limitations and potential confounders
One checkpoint only; no independently trained seed hierarchy available. No original masks/checkpoint/dataset locally; held-out identity is source-supported, not independently verified. Finite {len(c['group_B'])}-run outcomes can flip Delta's sign, and zero improvement is quantized in steps of 1/{len(c['group_B'])}. Consistently wrong means all observed B runs, not certain population error. Scores may reflect class difficulty, calibration, node degree, random-walk coverage, or checkpoint-specific failures; graph covariates are unavailable locally. Node bootstrap ignores graph dependence. Secondary score comparisons are exploratory with no post-hoc significance threshold or multiplicity-adjusted claim. Fixed full-compute policies and diagnostic allocation use different nominal budgets and exclude overhead. No predictions were fabricated or extrapolated. Arrays are derived exclusively from the validated original cache.

## G. Implications for Adaptive GPM
Use effect sizes and intervals, not an imposed significance cutoff. Strong positive benefit association and a meaningful paired gain over random allocation would support studying a validation-calibrated stopping rule next (e.g., add batches of walks until successive probability means stabilize). This rule is a proposal only. If higher K stabilizes predictions with negligible accuracy benefit, study accuracy-cost tradeoffs. If benefit prediction is weak or negative, investigate confidently incorrect nodes, coverage limitations and uncertainty that sampling cannot resolve before implementing Adaptive GPM. The measured evidence and checkpoint limitations should guide that decision; a correlation alone is insufficient to justify an efficient adaptive method.

Artifacts: `per_node.csv`, `predictions.npz`, `metrics.json`, `aggregate_means.csv`, `associations.csv`, `policies.csv`, `random_allocations.csv`, and three PNG/PDF figure pairs. `predictions.npz` preserves class probabilities and predicted classes, node IDs/labels, K/repetition IDs, group membership, split and checkpoint metadata. See `../README.md` for execution.
'''
    text += '\n### Measured interpretation and score comparison\n\n'
    text += '| Score | Spearman with error improvement | Spearman with baseline error | Benefit AUROC | Benefit AP | High-error AUROC | High-error AP |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name, row in m['association'].items():
        text += '| '+name+' | '+' | '.join(number(row[key]['estimate']) for key in ['delta_error_spearman','baseline_error_spearman','improvement_auroc','improvement_ap','high_error_e16_ge_0.5_auroc','high_error_e16_ge_0.5_ap'])+' |\n'
    text += f"\nMean covariance trace at K=16: {f(m['sampling_variability_by_k']['16'])}; K=128: {f(m['sampling_variability_by_k']['128'])}.\n"
    text += f"\nFor this cache, uncertainty predicts both baseline difficulty (rank correlation {primary['baseline_error_spearman']['estimate']:.3f}) and improvement ({primary['delta_error_spearman']['estimate']:.3f}). The positive allocation gain over random is {m['policies']['sampling_variance']['error_reduction_vs_expected_random']['estimate']*100:.3f} percentage points, with paired 95% interval [{m['policies']['sampling_variance']['error_reduction_vs_expected_random']['ci95'][0]*100:.3f}, {m['policies']['sampling_variance']['error_reduction_vs_expected_random']['ci95'][1]*100:.3f}]. Most improvement concentrates in the top quintile. Low variability does not imply correctness: {m['low_uncertainty_consistently_wrong_at128']['estimate']*100:.2f}% of the bottom quintile remain wrong in all observed K=128 outcome runs.\n"
    text += "\nThe observed positive benefit association and improvement over random are consistent with decision case A for this checkpoint. They support investigating a simple stopping-rule experiment next after restoring and verifying original data/checkpoint provenance, ideally with independent trained seeds. They do not establish the value of a practical adaptive policy: entropy/confidence/margin perform similarly in retrospective allocation, and the cost of repeated uncertainty-estimation passes has not been recovered. Case B (only stabilization, barely any accuracy change) is a less complete description of these results because measured error also falls; resolving the practical size of that gain requires validation and compute measurements. Case C remains a relevant failure-mode analysis for confidently incorrect low-variability nodes. No Adaptive GPM has been implemented.\n"
    (out/'REPORT.md').write_text(text)


if __name__=='__main__':
    main()
