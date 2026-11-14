"""Check and copy each run's two output files into clearly labelled folders for Akhil.

For every requested checkpoint, copies
    runs/<dataset>/ckpt<id>/final_predictions.npz
    runs/<dataset>/ckpt<id>/adaptive_decisions.npz
to
    handoff/<dataset>_ckpt<id>/   e.g. handoff/wikics_ckpt1/
and writes handoff/MANIFEST.json, listing each folder's checkpoint, seed, node
counts, threshold and file hashes.

Before copying, it checks that both files come from a full (not smoke) nested
run of the registered checkpoint, and that the decisions cover exactly the test
nodes of the predictions. The folder layout matches what
accuracy_efficiency_eval/evaluate.py expects.

Default: WikiCS checkpoints 1-3 and Computers checkpoints 2-3. Computers
checkpoint 1 is left out because Akhil already has Arvind's runs of it.

Example, from the repository root:
    python cross_dataset_validation/export_handoff.py --zip
    python cross_dataset_validation/export_handoff.py wikics:1 wikics:2
    python cross_dataset_validation/export_handoff.py cora_full:1 cora_full:2 cora_full:3 \
        --output cross_dataset_validation/handoff/corafull_checkpoints --zip
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from common import HERE, KS, environment, find_checkpoint, run_folder, sha256

FILES = ('final_predictions.npz', 'adaptive_decisions.npz')
DEFAULT = ('wikics:1', 'wikics:2', 'wikics:3', 'computers:2', 'computers:3')
HANDOFF = HERE / 'handoff'


def label(dataset, checkpoint_id):
    return f'{dataset}_ckpt{checkpoint_id}'


def check_pair(folder, dataset, checkpoint_sha256):
    """Validate one run's two files. Returns what the manifest records about them."""
    predictions, decisions = (folder / name for name in FILES)
    for path in (predictions, decisions):
        if not path.is_file():
            raise FileNotFoundError(f'{path} is missing. Run run_checkpoint.py for this checkpoint first.')
    with np.load(predictions, allow_pickle=False) as p, np.load(decisions, allow_pickle=False) as d:
        for name, z in (('predictions', p), ('decisions', d)):
            if bool(z['smoke_test']):
                raise ValueError(f'{folder}: the {name} file is from a smoke test.')
            if not bool(z['nested']):
                raise ValueError(f'{folder}: the {name} file is not from nested sampling.')
            if str(z['dataset']) != dataset:
                raise ValueError(f"{folder}: the {name} file is for {z['dataset']}, not {dataset}.")
            if str(z['checkpoint_sha256']) != checkpoint_sha256:
                raise ValueError(f'{folder}: the {name} file was made with a different checkpoint.')
        if [int(k) for k in p['k_values']] != list(KS):
            raise ValueError(f"{folder}: expected K = {list(KS)}, found {p['k_values'].tolist()}.")
        split = p['node_split'].astype(str)
        test_ids = p['node_ids'][split == 'test']
        if not (split == 'val').any() or not len(test_ids):
            raise ValueError(f'{folder}: predictions need both validation and test nodes.')
        if str(d['rule']) != 'four':
            raise ValueError(f"{folder}: decisions use the {d['rule']}-stage rule, not four-stage.")
        if not np.array_equal(d['node_ids'], test_ids):
            raise ValueError(f'{folder}: decision nodes differ from the test nodes in the predictions.')
        if d['final_k'].shape[0] != p['probabilities'].shape[1]:
            raise ValueError(f'{folder}: predictions and decisions have different repetition counts.')
        return dict(repetitions=int(p['probabilities'].shape[1]), val_nodes=int((split == 'val').sum()),
                    test_nodes=len(test_ids), classes=int(p['probabilities'].shape[-1]),
                    threshold=float(d['threshold']), target_pattern_passes=float(d['target']),
                    mean_pattern_passes=float(d['pattern_passes'].mean()),
                    prediction_seed=int(p['seed']), live_seed=int(d['seed']))


def copy_file(source, destination):
    """Copy, or skip if an identical file is already there. Never replaces a different file."""
    if destination.exists():
        if sha256(destination) == sha256(source):
            return
        raise FileExistsError(f'{destination} exists and differs from {source}. Delete it to replace it.')
    shutil.copy2(source, destination)


def export(items, out=HANDOFF, runs_folder=run_folder, registry_lookup=find_checkpoint):
    entries = []
    for item in items:
        dataset, checkpoint_id = item.split(':', 1)
        row = registry_lookup(dataset, checkpoint_id)
        source = runs_folder(dataset, checkpoint_id)
        info = check_pair(source, dataset, row['sha256'])
        target = out / label(dataset, checkpoint_id)
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            copy_file(source / name, target / name)
        entries.append(dict(folder=label(dataset, checkpoint_id), dataset=dataset, checkpoint_id=int(checkpoint_id),
                            training_seed=int(row['seed']), split_index=int(row['split_index']),
                            checkpoint_sha256=row['sha256'], **info,
                            files={name: sha256(target / name) for name in FILES}))
        print(f'{label(dataset, checkpoint_id)}: copied')
    return entries


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('checkpoints', nargs='*', default=list(DEFAULT), help='dataset:checkpoint_id')
    a.add_argument('--output', default=str(HANDOFF))
    a.add_argument('--zip', action='store_true',
                   help='also write <output folder name>.zip, e.g. handoff.zip, containing every folder')
    args = a.parse_args()
    out = Path(args.output)
    entries = export(args.checkpoints, out)
    manifest = dict(
        description='Nested four-stage adaptive inference. Thresholds frozen from validation nodes; '
                    'decisions are on test nodes. One folder per dataset and checkpoint.',
        usage='python evaluate.py --predictions <folder>/final_predictions.npz --output <results> '
              '--adaptive-decisions nested_four_stage=<folder>/adaptive_decisions.npz',
        environment=environment(), checkpoints=entries)
    with open(out / 'MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print('Wrote', out / 'MANIFEST.json')
    if args.zip:
        archive = out / f'{out.name}.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(out / 'MANIFEST.json', 'MANIFEST.json')
            for entry in entries:
                for name in FILES:
                    z.write(out / entry['folder'] / name, f"{entry['folder']}/{name}")
        print('Wrote', archive)


if __name__ == '__main__':
    main()
