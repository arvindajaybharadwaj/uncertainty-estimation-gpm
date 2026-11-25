"""Run the encoder-caching pipeline for one registered checkpoint.

Steps, each skipped if its output already exists:
  1. predictions  predictions.py: per-length layout, K = 16, 32, 64, 128 and
                  fixed K = 48, for validation and test nodes.
  2. live         live_cached.py: threshold from step 1's validation nodes,
                  then cached adaptive inference on test nodes.
  3. evaluate     accuracy_efficiency_eval/evaluate.py with the cached decisions,
                  as step 3 of cross_dataset_validation/run_checkpoint.py.
  4. compare      evaluate_cached.py: paired comparisons with the original run.

Needs the original run of the same checkpoint in cross_dataset_validation/runs
(runs_smoke with --smoke), which is read but never written. Outputs go to
encoder_caching/runs/<dataset>/ckpt<id>/, or runs_smoke/ with --smoke. A smoke
run also checks every cached stage against a fresh pass (live_cached.py --check).

Example, from the repository root:
    python encoder_caching/run_checkpoint.py --dataset wikics --checkpoint-id 1 --smoke
    python encoder_caching/run_checkpoint.py --dataset wikics --checkpoint-id 1
"""
import argparse
import subprocess
import sys

import paths

STEPS = ('predictions', 'live', 'evaluate', 'compare')


def commands(dataset, checkpoint_id, smoke):
    """The four steps as (name, output, command). Kept separate so it can be tested."""
    python = sys.executable
    folder = paths.folder(dataset, checkpoint_id, smoke)
    common = ['--dataset', dataset, '--checkpoint-id', str(checkpoint_id)] + (['--smoke'] if smoke else [])
    evaluation = folder / 'eval'
    return [
        ('predictions', folder / 'predictions.npz', [python, str(paths.HERE / 'predictions.py')] + common),
        ('live', folder / 'decisions.npz',
         [python, str(paths.HERE / 'live_cached.py')] + common + (['--check'] if smoke else [])),
        ('evaluate', evaluation / 'summary.json', [
            python, str(paths.EVAL_DIR / 'evaluate.py'), '--predictions', str(folder / 'predictions.npz'),
            '--output', str(evaluation), '--adaptive-decisions', f'cached_four_stage={folder / "decisions.npz"}']),
        ('compare', folder / 'comparison.json', [python, str(paths.HERE / 'evaluate_cached.py')] + common),
    ]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=paths.DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--smoke', action='store_true', help='32 nodes per split and 2 repetitions, to test the pipeline')
    a.add_argument('--only', choices=STEPS, help='run a single step')
    args = a.parse_args()

    paths.verified_checkpoint(args.dataset, args.checkpoint_id)
    folder = paths.folder(args.dataset, args.checkpoint_id, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)
    for name, output, command in commands(args.dataset, args.checkpoint_id, args.smoke):
        if args.only and name != args.only:
            continue
        if output.exists():
            print(f'[{name}] already done: {output}')
            continue
        print(f'[{name}] running', flush=True)
        subprocess.run(command, cwd=paths.EVAL_DIR if name == 'evaluate' else paths.HERE, check=True)
    print(f'Done. Outputs in {folder}')


if __name__ == '__main__':
    main()
