"""Tests for the parts that run without a GPU or GPM checkpoint.

From the repository root:
    python -m unittest discover -s cross_dataset_validation -p "test_*.py" -v
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import common
import diagnostic
import export_handoff
import run_checkpoint
import summarize_results
import uncertainty_comparison as uc


def favouring(cls, strength, classes=3):
    """Probabilities with `strength` on cls and the rest spread evenly."""
    q = np.full(classes, (1 - strength) / (classes - 1))
    q[cls] = strength
    return q


def make_data(nodes=50, runs=10, classes=3, seed=0):
    """Nodes 0-9 are wrong at K=16 and right at K=128. Only they have low K=16 confidence."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, classes, nodes)
    p = np.empty((4, runs, nodes, classes))
    for n in range(nodes):
        p[:, :, n] = favouring(y[n], 0.9, classes)
    for n in range(10):
        p[0, :, n] = favouring((y[n] + 1) % classes, 0.5 + 0.02 * n, classes)
    return dict(p=p, y=y, node_ids=np.arange(nodes) + 100, ks=[16, 32, 64, 128])


class CheapScores(unittest.TestCase):
    def test_definitions_match_the_notebook(self):
        p1 = np.array([[0.7, 0.2, 0.1]])
        p2 = np.array([[0.5, 0.3, 0.2]])
        s = uc.cheap_scores(p1, p2)
        self.assertAlmostEqual(s['Maximum confidence'][0], 0.3)
        self.assertAlmostEqual(s['Predictive entropy'][0], -(p1 * np.log(p1)).sum())
        self.assertAlmostEqual(s['Sampling disagreement'][0], 0.2)

    def test_scores_and_benefit_use_separate_runs(self):
        score_runs, benefit_runs = uc.split_runs(make_data(runs=10))
        self.assertEqual(score_runs.tolist(), [0, 1])
        self.assertEqual(benefit_runs.tolist(), list(range(2, 10)))
        self.assertFalse(np.intersect1d(score_runs, benefit_runs).size)

    def test_too_few_runs_is_refused(self):
        with self.assertRaises(ValueError):
            uc.split_runs(make_data(runs=3))

    def test_confidence_finds_the_nodes_that_benefit(self):
        table = uc.compare(make_data()).set_index('method')
        row = table.loc['Maximum confidence']
        self.assertAlmostEqual(row.auroc, 1.0)
        self.assertEqual(row.top_nodes, 10)
        self.assertAlmostEqual(row.top20_benefit_pp, 100.0)
        self.assertAlmostEqual(row.all_benefit_pp, 20.0)

    def test_summary_spreads_across_checkpoints(self):
        tables = []
        for checkpoint in ('1', '2'):
            t = uc.compare(make_data(seed=int(checkpoint)))
            t.insert(0, 'checkpoint', checkpoint)
            t.insert(0, 'dataset', 'toy')
            tables.append(t)
        summary = uc.summarise(pd.concat(tables))
        self.assertEqual(set(summary.checkpoints), {2})
        self.assertEqual(len(summary), 3)


class Diagnostic(unittest.TestCase):
    def test_counts_corrections_within_each_run(self):
        result = diagnostic.diagnose(make_data())
        self.assertAlmostEqual(result['accuracy_by_k']['16'], 80.0)
        self.assertAlmostEqual(result['accuracy_by_k']['128'], 100.0)
        self.assertAlmostEqual(result['corrected_pct_of_k16_errors'], 100.0)
        self.assertAlmostEqual(result['broken_pct_of_k16_correct'], 0.0)
        self.assertAlmostEqual(result['nodes_benefit_pct'], 20.0)
        self.assertAlmostEqual(result['prediction_changed_pct'], 20.0)


class Registry(unittest.TestCase):
    def row(self, checkpoint_id=1):
        return {field: '' for field in common.REGISTRY_FIELDS} | dict(dataset='wikics', checkpoint_id=checkpoint_id,
                                                                       path='GPM/model/wikics/seed0/best.pt')

    def test_add_find_and_refuse_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'checkpoints.csv'
            common.add_to_registry(self.row(), path)
            common.add_to_registry(self.row(2), path)
            self.assertEqual(common.find_checkpoint('wikics', 2, path)['checkpoint_id'], '2')
            with self.assertRaises(ValueError):
                common.add_to_registry(self.row(), path)
            with self.assertRaises(KeyError):
                common.find_checkpoint('computers', 1, path)

    def test_relative_paths_resolve_from_the_repository(self):
        self.assertEqual(common.checkpoint_file(self.row()), common.ROOT / 'GPM/model/wikics/seed0/best.pt')

    def test_hash_matches_final_inference(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'x.pt'
            path.write_bytes(b'checkpoint')
            self.assertEqual(common.sha256(path), hashlib.sha256(b'checkpoint').hexdigest())


class Pipeline(unittest.TestCase):
    def steps(self, smoke):
        folder = Path('runs/wikics/ckpt1')
        return {name: (output, command) for name, output, command in
                run_checkpoint.commands('wikics', Path('best.pt'), folder, 10, 256, 48, smoke)}

    def test_live_step_uses_the_predictions_for_its_threshold(self):
        steps = self.steps(False)
        live = steps['live'][1]
        self.assertEqual(live[live.index('--calibration') + 1], str(steps['predictions'][0]))
        self.assertEqual(live[live.index('--sampling') + 1], 'nested')
        self.assertEqual(live[live.index('--rule') + 1], 'four')
        self.assertEqual(live[live.index('--seed') + 1], str(run_checkpoint.LIVE_SEED))
        self.assertNotIn('--max-nodes-per-split', live)

    def test_smoke_limits_nodes(self):
        for _, command in self.steps(True).values():
            if 'run_inference.py' in command[1]:
                self.assertIn('--max-nodes-per-split', command)

    def test_evaluate_reads_the_live_decisions(self):
        steps = self.steps(False)
        evaluate = steps['evaluate'][1]
        self.assertIn(f"nested_four_stage={steps['live'][0]}", evaluate)


class Summary(unittest.TestCase):
    def write_run(self, folder, smoke=False):
        rng = np.random.default_rng(1)
        n_val, n_test, reps, classes = 6, 8, 3, 3
        labels = rng.integers(0, classes, n_val + n_test)
        p = np.full((4, reps, n_val + n_test, classes), 0.1 / (classes - 1))
        p[..., np.arange(n_val + n_test), labels] = 0.9
        node_ids = np.arange(n_val + n_test)
        split = np.array(['val'] * n_val + ['test'] * n_test)
        folder.mkdir(parents=True)
        np.savez(folder / 'final_predictions.npz', probabilities=p, labels=labels, node_ids=node_ids,
                 node_split=split, k_values=np.array([16, 32, 64, 128]))
        test_labels = labels[n_val:]
        predicted = np.tile(test_labels, (reps, 1))
        predicted[0, 0] = (test_labels[0] + 1) % classes  # one error in one repetition
        final_k = np.full((reps, n_test), 16)
        final_k[:, :2] = 128
        np.savez(folder / 'adaptive_decisions.npz', node_ids=node_ids[n_val:], predicted=predicted, final_k=final_k,
                 pattern_passes=np.where(final_k == 128, 240, 16), sampled_patterns=final_k,
                 seconds_total=2.4, threshold=0.8, smoke_test=smoke)
        (folder / 'eval').mkdir()
        comparisons = {'fixed_128': dict(diff=-0.004, ci_low=-0.008, ci_high=0.0),
                       'fixed_64_128_interpolated': dict(diff=0.003, ci_low=0.001, ci_high=0.005),
                       'random_same_cost': dict(diff=0.008, ci_low=0.005, ci_high=0.011)}
        with open(folder / 'eval' / 'summary.json', 'w') as f:
            json.dump({'live_adaptive': {'nested_four_stage': {'comparisons': comparisons}}}, f)

    def test_one_checkpoint_row(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'toy' / 'ckpt1'
            self.write_run(folder)
            row = summarize_results.checkpoint_row(folder)
        self.assertAlmostEqual(row['fixed_128_acc'], 100.0)
        self.assertAlmostEqual(row['adaptive_acc'], 100 * (1 - 1 / 24))
        self.assertAlmostEqual(row['gap_vs_fixed_128_pp'], -0.4)
        self.assertTrue(row['within_target'])
        self.assertFalse(row['within_target_whole_ci'])
        self.assertAlmostEqual(row['pattern_passes'], 0.25 * 240 + 0.75 * 16)
        self.assertAlmostEqual(row['stop_at_128_pct'], 25.0)
        self.assertAlmostEqual(row['live_ms_per_node'], 1000 * 2.4 / 24)

    def test_smoke_runs_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'toy' / 'ckpt1'
            self.write_run(folder, smoke=True)
            with self.assertRaises(ValueError):
                summarize_results.checkpoint_row(folder)


class Handoff(unittest.TestCase):
    SHA = 'a' * 64

    def write_run(self, folder, smoke=False, sha=SHA, rule='four', dataset='wikics'):
        folder.mkdir(parents=True, exist_ok=True)
        node_ids = np.arange(6)
        split = np.array(['val'] * 2 + ['test'] * 4)
        meta = dict(dataset=dataset, checkpoint_sha256=sha, nested=True, smoke_test=smoke)
        np.savez(folder / 'final_predictions.npz', probabilities=np.full((4, 3, 6, 2), 0.5), labels=np.zeros(6),
                 node_ids=node_ids, node_split=split, k_values=np.array([16, 32, 64, 128]), seed=71000, **meta)
        np.savez(folder / 'adaptive_decisions.npz', node_ids=node_ids[2:], final_k=np.full((3, 4), 16),
                 predicted=np.zeros((3, 4)), pattern_passes=np.full((3, 4), 16), threshold=0.8, target=48.,
                 rule=rule, seed=171000, **meta)

    def export(self, root, items=('wikics:1',)):
        runs = root / 'runs'
        row = dict(sha256=self.SHA, seed='0', split_index='0')
        return export_handoff.export(items, root / 'handoff',
                                     runs_folder=lambda d, c: runs / f'{d}{c}',
                                     registry_lookup=lambda d, c: row)

    def test_copies_into_labelled_folders(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_run(root / 'runs' / 'wikics1')
            entry = self.export(root)[0]
            folder = root / 'handoff' / 'wikics_ckpt1'
            self.assertTrue((folder / 'final_predictions.npz').is_file())
            self.assertTrue((folder / 'adaptive_decisions.npz').is_file())
        self.assertEqual(entry['folder'], 'wikics_ckpt1')
        self.assertEqual((entry['val_nodes'], entry['test_nodes'], entry['repetitions']), (2, 4, 3))
        self.assertEqual(entry['threshold'], 0.8)

    def test_exporting_twice_is_harmless(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_run(root / 'runs' / 'wikics1')
            first = self.export(root)[0]['files']
            self.assertEqual(self.export(root)[0]['files'], first)

    def test_a_changed_file_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.write_run(root / 'runs' / 'wikics1')
            self.export(root)
            np.savez(root / 'runs' / 'wikics1' / 'adaptive_decisions.npz', node_ids=np.arange(2, 6),
                     final_k=np.full((3, 4), 32), predicted=np.zeros((3, 4)), pattern_passes=np.full((3, 4), 48),
                     threshold=0.9, target=48., rule='four', seed=171000, dataset='wikics',
                     checkpoint_sha256=self.SHA, nested=True, smoke_test=False)
            with self.assertRaises(FileExistsError):
                self.export(root)

    def test_bad_runs_are_refused(self):
        cases = [dict(smoke=True), dict(sha='b' * 64), dict(rule='two'), dict(dataset='computers')]
        for case in cases:
            with self.subTest(**case), tempfile.TemporaryDirectory() as d:
                root = Path(d)
                self.write_run(root / 'runs' / 'wikics1', **case)
                with self.assertRaises(ValueError):
                    self.export(root)

    def test_missing_run_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                self.export(Path(d))


class Config(unittest.TestCase):
    def test_training_params_match_main_with_use_params(self):
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest('pyyaml is not installed')
        common.use_gpm()
        import train_checkpoint
        params = train_checkpoint.gpm_params('wikics')
        self.assertEqual(params['split'], 'public')
        self.assertEqual(params['dropout'], 0.5)
        self.assertEqual(params['node_pe'], 'lap')
        self.assertEqual(params['split_repeat'], 1)
        self.assertFalse(params['inference'])
        self.assertTrue(Path(params['data_path']).resolve() == (common.ROOT / 'GPM' / 'data').resolve())


if __name__ == '__main__':
    unittest.main()
