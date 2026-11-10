"""Paths and helpers shared by the cross-dataset validation scripts.

GPM's own files are not changed. Two of its imports are handled here so the
scripts also run on Windows:
  - model/random_walk.py imports dgl's node2vec sampler but never calls it.
    A stub is installed when dgl is missing, as accuracy_efficiency_eval does.
  - main.py imports utils/sys.py, which needs the Unix-only `resource` module.
    These scripts never import main.py, so that import is avoided entirely.
"""
import csv
import hashlib
import subprocess
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GPM_DIR = ROOT / 'GPM' / 'GPM'
EVAL_DIR = ROOT / 'accuracy_efficiency_eval'
FINAL_INFERENCE = ROOT / 'GPM' / 'final_inference.py'
HERE = Path(__file__).resolve().parent
REGISTRY = HERE / 'checkpoints.csv'
RUNS = HERE / 'runs'
SMOKE_RUNS = HERE / 'runs_smoke'

DATASETS = ('computers', 'wikics', 'cora_full')
KS = (16, 32, 64, 128)
REGISTRY_FIELDS = ('dataset', 'checkpoint_id', 'seed', 'split_index', 'path', 'sha256', 'best_epoch',
                   'val_acc', 'test_acc', 'source', 'torch', 'gpu', 'git_commit', 'notes')


def install_dgl_stub_if_missing():
    """GPM imports dgl's node2vec sampler but never calls it, so allow it to be absent."""
    try:
        import dgl  # noqa: F401
    except Exception:
        dgl = types.ModuleType('dgl')
        sampling = types.ModuleType('dgl.sampling')

        def unavailable(*args, **kwargs):
            raise RuntimeError('dgl is not installed, and this call needs it.')

        sampling.node2vec_random_walk = unavailable
        dgl.sampling = sampling
        sys.modules['dgl'] = dgl
        sys.modules['dgl.sampling'] = sampling


def use_gpm():
    """Make GPM's packages importable, as if running from GPM/GPM."""
    install_dgl_stub_if_missing()
    if str(GPM_DIR) not in sys.path:
        sys.path.insert(0, str(GPM_DIR))


def sha256(path):
    """Same digest as checkpoint_hash in GPM/final_inference.py."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def relative(path):
    """Store paths relative to the repository so the registry works on any machine."""
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_registry(path=REGISTRY):
    if not Path(path).exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def find_checkpoint(dataset, checkpoint_id, path=REGISTRY):
    for row in read_registry(path):
        if row['dataset'] == dataset and row['checkpoint_id'] == str(checkpoint_id):
            return row
    raise KeyError(f'No checkpoint {checkpoint_id} for {dataset} in {path}. Register it first.')


def checkpoint_file(row):
    path = Path(row['path'])
    return path if path.is_absolute() else ROOT / path


def add_to_registry(row, path=REGISTRY):
    """Append one checkpoint. Refuses a second row with the same dataset and checkpoint ID."""
    missing = set(REGISTRY_FIELDS) - set(row)
    if missing:
        raise ValueError(f'Registry row is missing {sorted(missing)}')
    rows = read_registry(path)
    if any(r['dataset'] == row['dataset'] and r['checkpoint_id'] == str(row['checkpoint_id']) for r in rows):
        raise ValueError(f"{row['dataset']} checkpoint {row['checkpoint_id']} is already registered.")
    new = not Path(path).exists()
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REGISTRY_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row[k] for k in REGISTRY_FIELDS})


def run_folder(dataset, checkpoint_id, smoke=False):
    return (SMOKE_RUNS if smoke else RUNS) / dataset / f'ckpt{checkpoint_id}'


def environment():
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    return dict(torch=torch.__version__, gpu=gpu, git_commit=git_commit())
