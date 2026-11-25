"""Paths and settings shared by the encoder-caching scripts.

The helpers in cross_dataset_validation/common.py are reused unchanged, so the
checkpoints, data and GPM imports are handled exactly as in the original runs.
Nothing in cross_dataset_validation/ is written to.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CROSS = ROOT / 'cross_dataset_validation'
sys.path.insert(0, str(CROSS))

from common import (DATASETS, EVAL_DIR, GPM_DIR, checkpoint_file, environment, find_checkpoint,  # noqa: E402,F401
                    run_folder, sha256, use_gpm)

RUNS = HERE / 'runs'
SMOKE_RUNS = HERE / 'runs_smoke'
RESULTS = HERE / 'results'

PREDICTION_SEED = 71000          # as in cross_dataset_validation/run_checkpoint.py
LIVE_SEED = 171000
TIMING_SEED = 271000             # as in time_methods.py and fixed_budget_timing.py
TARGET = 48.                     # pattern-passes per node
REPEATS = 10
SMOKE_REPEATS = 2
SMOKE_NODES = 32                 # per split, as run_checkpoint.py --smoke
BATCH_SIZE = {'cora_full': 32}   # batch size of the original runs; 256 otherwise
CHECK_TOLERANCE = 1e-4           # as in fixed_budget_baseline.py
BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261105 + 9    # the seed evaluate.py uses for live comparisons


def batch_size(dataset, given=None):
    return given or BATCH_SIZE.get(dataset, 256)


def folder(dataset, checkpoint_id, smoke=False):
    """Output folder of the encoder-caching run."""
    return (SMOKE_RUNS if smoke else RUNS) / dataset / f'ckpt{checkpoint_id}'


def original_folder(dataset, checkpoint_id, smoke=False):
    """Folder of the original run in cross_dataset_validation, read only."""
    return run_folder(dataset, checkpoint_id, smoke)


def verified_checkpoint(dataset, checkpoint_id):
    row = find_checkpoint(dataset, checkpoint_id)
    path = checkpoint_file(row)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != row['sha256']:
        raise ValueError(f'{path} does not match the hash in checkpoints.csv.')
    return path, row['sha256']


def refuse_existing(*paths):
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing:
        raise FileExistsError(f'Refusing to overwrite {", ".join(existing)}')
