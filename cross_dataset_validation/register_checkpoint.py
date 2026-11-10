"""Add a checkpoint trained elsewhere, such as Arvind's epoch_380.pt, to checkpoints.csv.

Checkpoints from train_checkpoint.py register themselves. Use this only for
outside checkpoints. The hash is computed here; the other fields are what you
know about how it was trained.

Example, from the repository root:
    python cross_dataset_validation/register_checkpoint.py --dataset computers --checkpoint-id 1 \
        --seed 0 --path GPM/model/computers/epoch_380.pt --best-epoch 380 \
        --source "Arvind, Modal L40S" --notes "trained with main.py --use_params --split_repeat 1"
"""
import argparse
from pathlib import Path

from common import DATASETS, add_to_registry, git_commit, relative, sha256


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=DATASETS)
    a.add_argument('--checkpoint-id', type=int, required=True)
    a.add_argument('--seed', type=int, required=True, help='training seed it was trained with')
    a.add_argument('--path', required=True)
    a.add_argument('--split-index', type=int, default=0)
    a.add_argument('--best-epoch', default='')
    a.add_argument('--val-acc', default='')
    a.add_argument('--test-acc', default='')
    a.add_argument('--source', required=True, help='who trained it and where')
    a.add_argument('--torch', default='unknown', help='torch version used for training, if known')
    a.add_argument('--gpu', default='unknown', help='GPU used for training, if known')
    a.add_argument('--notes', default='')
    args = a.parse_args()
    if args.split_index != 0:
        raise ValueError('final_inference.py evaluates on split 0, so other splits cannot be used.')
    path = Path(args.path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256(path)
    add_to_registry(dict(dataset=args.dataset, checkpoint_id=args.checkpoint_id, seed=args.seed,
                         split_index=args.split_index, path=relative(path), sha256=digest,
                         best_epoch=args.best_epoch, val_acc=args.val_acc, test_acc=args.test_acc,
                         source=args.source, torch=args.torch, gpu=args.gpu, git_commit=git_commit(),
                         notes=args.notes))
    print(f'Registered {args.dataset} checkpoint {args.checkpoint_id}: {digest}')


if __name__ == '__main__':
    main()
