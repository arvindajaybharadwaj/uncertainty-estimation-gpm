import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import evaluate as ev
import measure_timing as mt
import aggregate_runs as ar
import extra_analyses as ex
import matched_timing as tm


def make(probabilities=None, reps=8, nodes=40):
    ks = [16, 32, 64, 128]
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, nodes)
    if probabilities is None:
        p = rng.dirichlet(np.ones(3), (4, reps, nodes)).astype(np.float32)
    else:
        p = probabilities
    return dict(p=p, y=y, node_ids=np.arange(nodes) + 100, ks=ks, node_split=None, nested=False)


def one_hot_like(cls, strength):
    """Probabilities favouring cls with the given top probability."""
    rest = (1 - strength) / 2
    q = np.full(3, rest, dtype=np.float32)
    q[cls] = strength
    return q


class Costs(unittest.TestCase):
    def test_nominal_rerun_counts_every_pass(self):
        costs, _ = ev.load_costs([16, 32, 64, 128])
        rerun, reuse = ev.stop_costs(costs)
        self.assertEqual(rerun.tolist(), [16, 48, 112, 240])
        self.assertEqual(reuse.tolist(), [16, 32, 64, 128])

    def test_transformer_is_never_reused(self):
        costs = {k: dict(sample=1., encode=2., transform=3.) for k in ev.ADAPTIVE_STEPS}
        rerun, reuse = ev.stop_costs(costs)
        self.assertEqual(rerun.tolist(), [6, 12, 18, 24])
        self.assertEqual(reuse.tolist(), [6, 9, 12, 15])

    def test_timing_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.json'
            rows = [dict(k=k, sample_s_per_node=.1, encode_s_per_node=.2, transform_s_per_node=.3)
                    for k in ev.ADAPTIVE_STEPS]
            path.write_text(json.dumps(dict(per_k=rows)))
            costs, unit = ev.load_costs([16, 32, 64, 128], path)
            self.assertEqual(unit, 'seconds per node')
            self.assertAlmostEqual(ev.full(costs, 16), .6)


class Adaptive(unittest.TestCase):
    def test_hand_worked_node(self):
        # Confidence .5, .6, .9, .95. Wrong at step 1 and 2, right from step 3.
        p = np.zeros((4, 1, 1, 3), dtype=np.float32)
        p[0, 0, 0] = one_hot_like(1, .5)
        p[1, 0, 0] = one_hot_like(1, .6)
        p[2, 0, 0] = one_hot_like(0, .9)
        p[3, 0, 0] = one_hot_like(0, .95)
        d = make(p, reps=1, nodes=1)
        d['y'] = np.array([0])
        nodes = np.array([0])
        s = ev.adaptive_scores(d, nodes, 'confidence')
        e, step = ev.adaptive(d, nodes, s.copy(), .85)
        self.assertEqual(step.item(), 2)
        self.assertEqual(e.item(), 1)
        e, step = ev.adaptive(d, nodes, s.copy(), .55)
        self.assertEqual(step.item(), 1)
        self.assertEqual(e.item(), 0)

    def test_last_step_always_stops(self):
        d = make()
        nodes = np.arange(40)
        s = ev.adaptive_scores(d, nodes, 'confidence')
        _, step = ev.adaptive(d, nodes, s.copy(), 2.0)
        self.assertTrue((step == 3).all())

    def test_scores_are_not_modified(self):
        d = make()
        nodes = np.arange(40)
        s = ev.adaptive_scores(d, nodes, 'confidence')
        before = s.copy()
        rerun, _ = ev.stop_costs(ev.load_costs(d['ks'])[0])
        ev.threshold_for_cost(s, 64, rerun, ev.candidate_thresholds(s, 20))
        self.assertTrue((s == before).all())

    def test_threshold_respects_target_and_ignores_labels(self):
        d = make()
        nodes = np.arange(40)
        s = ev.adaptive_scores(d, nodes, 'confidence')
        rerun, _ = ev.stop_costs(ev.load_costs(d['ks'])[0])
        c = ev.candidate_thresholds(s, 50)
        t = ev.threshold_for_cost(s, 64, rerun, c)
        passed = s >= t
        passed[-1] = True
        self.assertLessEqual(rerun[passed.argmax(0)].mean(), 64)
        d['y'] = (d['y'] + 1) % 3
        self.assertEqual(t, ev.threshold_for_cost(s, 64, rerun, c))

    def test_higher_threshold_never_costs_less(self):
        d = make()
        nodes = np.arange(40)
        s = ev.adaptive_scores(d, nodes, 'confidence')
        rerun, _ = ev.stop_costs(ev.load_costs(d['ks'])[0])
        costs = [rerun[ev.adaptive(d, nodes, s.copy(), t)[1]].mean() for t in (.4, .6, .8, .95)]
        self.assertEqual(costs, sorted(costs))


class Baselines(unittest.TestCase):
    def test_random_hits_target(self):
        e, f = ev.random_allocation(np.zeros(4), np.ones(4), 16, 128, 44)
        self.assertAlmostEqual(f, .25)
        self.assertAlmostEqual(e.mean(), .25)

    def test_repeated_averages_probabilities_not_votes(self):
        # Pass 1 strongly favours 0; passes 2 and 3 weakly favour 1.
        p = np.zeros((4, 3, 1, 3), dtype=np.float32)
        p[:] = one_hot_like(0, .9)
        p[:, 1:] = np.array([.3, .4, .3], dtype=np.float32)
        d = make(p, reps=3, nodes=1)
        d['y'] = np.array([0])
        self.assertEqual(ev.repeated(d, 16, 3, np.array([0])).item(), 1)

    def test_repeated_uses_disjoint_groups(self):
        d = make(reps=8)
        self.assertEqual(ev.repeated(d, 16, 3, np.arange(40)).shape, (40,))

    def test_bootstrap_identical_is_zero(self):
        a = np.array([0., 1., 1., 0.])
        self.assertEqual(ev.bootstrap_diff(a, a, 50, 0), (0., 0., 0.))


class Inputs(unittest.TestCase):
    def test_final_mode_uses_val_and_test_only(self):
        d = make()
        d['node_split'] = np.array(['val'] * 10 + ['test'] * 20 + ['train'] * 10)
        cal, test, mode = ev.split_nodes(d, 0)
        self.assertEqual(mode, 'FINAL')
        self.assertEqual((len(cal), len(test)), (10, 20))
        self.assertFalse(set(cal) & set(test))

    def test_pilot_split_is_disjoint(self):
        cal, test, mode = ev.split_nodes(make(), 0)
        self.assertEqual(mode, 'PILOT')
        self.assertEqual(len(set(cal) | set(test)), 40)
        self.assertFalse(set(cal) & set(test))

    def test_bad_probabilities_are_refused(self):
        d = make()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'x.npz'
            np.savez(path, probabilities=d['p'] * 2, labels=d['y'], node_ids=d['node_ids'], k_values=d['ks'])
            with self.assertRaises(ValueError):
                ev.load_predictions(path)

    def test_live_decisions(self):
        y = np.array([0, 1, 2])
        ids = np.array([5, 6, 7])
        costs, _ = ev.load_costs([16, 32, 64, 128])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.npz'
            np.savez(path, node_ids=ids, final_k=np.array([[16, 32, 128]]), predicted=np.array([[0, 1, 0]]),
                     seconds_total=np.array(3.0))
            out = ev.load_adaptive_decisions(path, y, ids, costs)[0]
        self.assertAlmostEqual(out['accuracy'], 2 / 3)
        self.assertAlmostEqual(out['rerun_cost'], (16 + 48 + 240) / 3)
        self.assertAlmostEqual(out['seconds_per_node'], 1)

    def test_live_decisions_are_aligned_by_node_id(self):
        y = np.array([0, 1])
        ids = np.array([5, 6])
        costs, _ = ev.load_costs([16, 32, 64, 128])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.npz'
            np.savez(path, node_ids=np.array([6, 5]), final_k=np.array([[16, 16]]), predicted=np.array([[1, 0]]))
            self.assertEqual(ev.load_adaptive_decisions(path, y, ids, costs)[0]['accuracy'], 1)

    def test_recorded_passes_override_four_stage_assumption(self):
        y = np.array([0, 1])
        ids = np.array([5, 6])
        costs, _ = ev.load_costs([16, 32, 64, 128])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'd.npz'
            np.savez(path, node_ids=ids, final_k=np.array([[16, 128]]), predicted=np.array([[0, 1]]),
                     pattern_passes=np.array([[16, 144]]), encoding_reuse=np.array(False))
            out = ev.load_adaptive_decisions(path, y, ids, costs)[0]
        self.assertEqual(out['rerun_cost'], 80)
        self.assertIsNone(out['reuse_cost'])

    def test_live_comparison_baselines(self):
        d = make(reps=8, nodes=40)
        out = ev.compare_live(np.ones(40), np.arange(40), d, 44., 20, 0)
        self.assertEqual(set(out), {'fixed_128', 'fixed_32_64_interpolated', 'random_same_cost'})


class Timing(unittest.TestCase):
    def test_summary_is_per_node_per_pass(self):
        r = mt.summarise(16, 100, 2, sample_s=2., total_model_s=10., encode_s=6., peak_mb=5.)
        self.assertAlmostEqual(r['sample_s_per_node'], .01)
        self.assertAlmostEqual(r['encode_s_per_node'], .03)
        self.assertAlmostEqual(r['transform_s_per_node'], .02)
        self.assertAlmostEqual(r['total_s_per_node'], .06)

    def test_impossible_timing_is_refused(self):
        with self.assertRaises(ValueError):
            mt.summarise(16, 100, 1, 1., 1., 2., None)

    def test_output_is_readable_by_evaluate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 't.json'
            rows = [mt.summarise(k, 10, 1, 1., 3., 1., None) for k in ev.ADAPTIVE_STEPS]
            path.write_text(json.dumps(dict(per_k=rows)))
            costs, _ = ev.load_costs(ev.ADAPTIVE_STEPS, path)
            self.assertAlmostEqual(ev.full(costs, 16), .4)


class SavedCheck(unittest.TestCase):
    def test_same_nodes_report_accuracy_gap(self):
        rows = [dict(k=128, accuracy=.90)]
        out = mt.compare_with_saved(rows, [1, 2], dict(node_ids=[1, 2], accuracy={128: .91}))
        self.assertAlmostEqual(out['largest_gap'], .01)

    def test_different_nodes_are_refused(self):
        with self.assertRaises(ValueError):
            mt.compare_with_saved([], [1, 2], dict(node_ids=[2, 1], accuracy={}))

    def test_dgl_stub_never_hides_a_call(self):
        import sys
        saved = {k: sys.modules.pop(k, None) for k in ('dgl', 'dgl.sampling')}
        try:
            mt.install_dgl_stub_if_missing()
            from dgl.sampling import node2vec_random_walk
            with self.assertRaises(RuntimeError):
                node2vec_random_walk()
        finally:
            for k in ('dgl', 'dgl.sampling'):
                sys.modules.pop(k, None)
                if saved[k] is not None:
                    sys.modules[k] = saved[k]


class TimedCosts(unittest.TestCase):
    def _run(self, per_pattern, flat):
        d = make(reps=8, nodes=200)
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / 't.json'
            rows = [dict(k=k, sample_s_per_node=flat, encode_s_per_node=per_pattern * k,
                         transform_s_per_node=per_pattern * k ** 2 / 64) for k in ev.ADAPTIVE_STEPS]
            path.write_text(json.dumps(dict(per_k=rows)))
            costs, unit = ev.load_costs(d['ks'], path)
        return ev.run(d, costs, draws=20), unit

    def test_superlinear_time(self):
        out, unit = self._run(1e-6, 0.)
        self.assertEqual(unit, 'seconds per node')
        self.assertFalse(out['curve'].cost.isna().any())

    def test_time_dominated_by_a_flat_cost(self):
        out, _ = self._run(1e-9, 1e-3)
        self.assertFalse(out['curve'].cost.isna().any())


class Aggregate(unittest.TestCase):
    def _folder(self, root, name, adaptive_accuracy, final):
        folder = Path(root) / name
        folder.mkdir()
        rows = [dict(method='fixed', setting='K=128', cost=128., accuracy=.9, target=''),
                dict(method='adaptive', setting=f'threshold {adaptive_accuracy}', cost=44., accuracy=adaptive_accuracy, target=48)]
        ar.pd.DataFrame(rows).to_csv(folder / 'curve.csv', index=False)
        (folder / 'summary.json').write_text(json.dumps(dict(mode=final, cost_unit='seconds per node', success=dict(
            diff=-.001, rerun_cost=40., fixed_128_cost=128.))))
        return str(folder)

    def test_thresholds_differ_but_targets_combine(self):
        with tempfile.TemporaryDirectory() as t:
            a = self._folder(t, 'a', .90, 'FINAL')
            b = self._folder(t, 'b', .92, 'FINAL')
            curve, success, live = ar.combine(ar.parse_runs([f'computers:1={a}', f'computers:2={b}']))
        row = curve[curve.method == 'adaptive'].iloc[0]
        self.assertEqual(row.checkpoints, 2)
        self.assertAlmostEqual(row.accuracy_mean, .91)
        self.assertAlmostEqual(row.accuracy_sd, .0141421, places=5)
        self.assertEqual(len(success), 2)

    def test_pilot_results_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            a = self._folder(t, 'a', .90, 'PILOT')
            with self.assertRaises(ValueError):
                ar.combine(ar.parse_runs([f'computers:1={a}']))

    def test_mixed_cost_units_are_refused(self):
        with tempfile.TemporaryDirectory() as t:
            a = self._folder(t, 'a', .90, 'FINAL')
            b = self._folder(t, 'b', .90, 'FINAL')
            path = Path(b) / 'summary.json'
            data = json.loads(path.read_text())
            data['cost_unit'] = 'pattern-passes'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                ar.combine(ar.parse_runs([f'computers:1={a}', f'computers:2={b}']))

    def test_bad_run_name_is_refused(self):
        with self.assertRaises(ValueError):
            ar.parse_runs(['computers=results'])


class Incremental(unittest.TestCase):
    def test_steps_are_nested_averages(self):
        p = np.random.default_rng(1).dirichlet(np.ones(3), (16, 5)).astype(np.float32)
        s = ex.cumulative_means(p)
        self.assertEqual(s.shape, (4, 2, 5, 3))
        self.assertTrue(np.allclose(s[0, 0], p[0]))
        self.assertTrue(np.allclose(s[1, 0], p[:2].mean(0)))
        self.assertTrue(np.allclose(s[3, 1], p[8:16].mean(0)))

    def test_replay_stops_at_first_confident_average(self):
        q = np.zeros((4, 1, 1, 3), dtype=np.float32)
        q[0, 0, 0] = one_hot_like(1, .5)
        q[1, 0, 0] = one_hot_like(0, .9)
        q[2:, 0, 0] = one_hot_like(1, .95)
        e, stop = ex.replay(q, np.array([0]), .85)
        self.assertEqual(stop.item(), 1)
        self.assertEqual(e.item(), 1)

    def test_two_stage_runs_and_counts_both_passes(self):
        out = ex.two_stage(make(reps=8, nodes=200), 0, 20)
        self.assertTrue(((out.cost - 16 - 128 * out.escalated).abs() < 1e-9).all())

    def test_cost_never_exceeds_128(self):
        self.assertEqual(ex.COST.max(), 128)


class Matched(unittest.TestCase):
    def _run(self, method, r, seconds):
        return dict(method=method, round=r, seconds=seconds, accuracy=.9, pattern_passes=50., peak_mb=10.)

    def test_speedup_is_paired_within_round(self):
        runs = [self._run('fixed_128', 0, 10.), self._run('fixed_128', 1, 20.), self._run('fresh_two_stage', 0, 5.),
                self._run('fresh_two_stage', 1, 10.)]
        row = [r for r in tm.summarise(runs, 100, 2) if r['method'] == 'fresh_two_stage'][0]
        self.assertEqual(row['speedup_vs_fixed_128'], 2)
        self.assertEqual((row['speedup_min'], row['speedup_max']), (2, 2))
        self.assertAlmostEqual(row['ms_per_node'], 37.5)

    def test_missing_baseline_is_refused(self):
        with self.assertRaises(ValueError):
            tm.summarise([self._run('fresh_two_stage', 0, 5.)], 100, 2)

    def test_missing_round_is_refused(self):
        with self.assertRaises(ValueError):
            tm.summarise([self._run('fixed_128', 0, 1.), self._run('fixed_128', 1, 1.), self._run('fixed_16', 0, 1.)], 10, 1)

    def test_every_round_runs_every_method(self):
        methods = tm.method_table(dict(nested_four=.8, fresh_four=.8, fresh_two=.9))
        self.assertEqual(len(methods), 9)
        for r in range(3):
            self.assertEqual({m['name'] for m in tm.round_order(methods, r, 1)}, {m['name'] for m in methods})
        self.assertNotEqual([m['name'] for m in tm.round_order(methods, 0, 1)], [m['name'] for m in tm.round_order(methods, 1, 1)])


class EndToEnd(unittest.TestCase):
    def test_run(self):
        out = ev.run(make(reps=8, nodes=200), ev.load_costs([16, 32, 64, 128])[0], draws=20)
        self.assertEqual(out['mode'], 'PILOT')
        self.assertTrue({'fixed', 'random', 'repeated', 'adaptive'} <= set(out['curve'].method))


if __name__ == '__main__':
    unittest.main()
