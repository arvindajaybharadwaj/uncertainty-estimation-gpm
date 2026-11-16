"""Run the full adaptive-inference pipeline for one registered checkpoint.

Steps, each skipped if its output already exists:
  1. predictions  final_inference.py --stage predictions --sampling nested:
                  class probabilities at K = 16, 32, 64, 128 for validation and
                  test nodes, with nested pattern sets.
  2. live         final_inference.py --stage live --rule four: threshold frozen
                  from the validation nodes in step 1 at the cost target, then
                  real adaptive inference on test nodes.
  3. evaluate     accuracy_efficiency_eval/evaluate.py with the live decisions.
  4. plot         accuracy_efficiency_eval/plot_results.py.

Outputs go to cross_dataset_validation/runs/<dataset>/ckpt<id>/, or runs_smoke/
with --smoke. The same seeds are used for every checkpoint and dataset.

Example, from the repository root:
    python cross_dataset_validation/run_checkpoint.py --dataset wikics --checkpoint-id 1 --smoke
    python cross_dataset_validation/run_checkpoint.py --dataset wikics --checkpoint-id 1
"""
import argparse
import subprocess
import sys

from common import DATASETS, EVAL_DIR, GPM_DIR, HERE, checkpoint_file, find_checkpoint, run_folder, sha256

PREDICTION_SEED = 71000   # final_inference.py default
LIVE_SEED = 171000        # as in GPM/final_modal.py
TARGET = 48               # pattern-passes per node, as in the Computers runs


def commands(dataset, checkpoint, folder, repeats, batch_size, target, smoke):
    """The four steps as (name, output, command). Kept separate so it can be tested."""
    python = sys.executable
    predictions = folder / 'final_predictions.npz'
    decisions = folder / 'adaptive_decisions.npz'
    evaluation = folder / 'eval'
    common = ['--gpm-dir', str(GPM_DIR), '--dataset', dataset, '--checkpoint', str(checkpoint),
              '--repeats', str(repeats), '--batch-size', str(batch_size), '--sampling', 'nested']
    if smoke:
        common += ['--max-nodes-per-split', '32']
    inference = [python, str(HERE / 'run_inference.py')]
    return [
        ('predictions', predictions, inference + common + [
            '--stage', 'predictions', '--seed', str(PREDICTION_SEED), '--output', str(predictions)]),
        ('live', decisions, inference + common + [
            '--stage', 'live', '--rule', 'four', '--calibration', str(predictions), '--target', str(target),
            '--seed', str(LIVE_SEED), '--output', str(decisions)]),
        ('evaluate', evaluation / 'summary.json', [
            python, str(EVAL_DIR / 'evaluate.py'), '--predictions', str(predictions), '--output', str(evaluation),
            '--adaptive-decisions', f'nested_four_stage={decisions}']),
        ('plot', evaluation / 'accuracy_vs_cost.png', [
            python, str(EVAL_DIR / 'plot_results.py'), '--results', str(evaluation)]),
    ]


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--repeats', type=int, default=10, help='sampling repetitions; 10 for final results')
    a.add_argument('--batch-size', type=int, default=256)
    a.add_argument('--target', type=float, default=TARGET)
    a.add_argument('--smoke', action='store_true', help='32 nodes per split and 2 repetitions, to test the pipeline')
    a.add_argument('--only', choices=['predictions', 'live', 'evaluate', 'plot'], help='run a single step')
    args = a.parse_args()

    row = find_checkpoint(args.dataset, args.checkpoint_id)
    checkpoint = checkpoint_file(row)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256(checkpoint) != row['sha256']:
        raise ValueError(f'{checkpoint} does not match the hash in checkpoints.csv.')
    repeats = 2 if args.smoke else args.repeats
    folder = run_folder(args.dataset, args.checkpoint_id, args.smoke)
    folder.mkdir(parents=True, exist_ok=True)

    for name, output, command in commands(args.dataset, checkpoint, folder, repeats, args.batch_size,
                                          args.target, args.smoke):
        if args.only and name != args.only:
            continue
        if output.exists():
            print(f'[{name}] already done: {output}')
            continue
        print(f'[{name}] running', flush=True)
        subprocess.run(command, cwd=EVAL_DIR if name in ('evaluate', 'plot') else HERE, check=True)
    print(f'Done. Outputs in {folder}')


if __name__ == '__main__':
    main()
