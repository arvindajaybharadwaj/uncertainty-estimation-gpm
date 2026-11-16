"""Train one GPM checkpoint on split 0 with a chosen training seed, then register it.

This follows GPM/GPM/main.py's run() step for step, with two differences:
  - It always trains on split 0, the split final_inference.py evaluates on.
    main.py trains run i on split i with seed i, so it cannot give three seeds
    on one split.
  - It keeps the weights from the evaluation with the best validation accuracy,
    and records that epoch. Test accuracy is recorded but never used to choose.

Configuration matches `python GPM/main.py --dataset <name> --use_params --split_repeat 1`,
which is how the Computers checkpoint epoch_380.pt was trained (seed 0).

Example, from the repository root:
    python cross_dataset_validation/train_checkpoint.py --dataset wikics --seed 0
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import yaml

from common import DATASETS, GPM_DIR, add_to_registry, environment, relative, sha256, use_gpm


def gpm_params(dataset):
    """Rebuild the params dict main.py would create for this dataset with --use_params."""
    from utils.args import get_args
    argv = sys.argv
    try:
        sys.argv = ['main.py', '--dataset', dataset, '--use_params', '--debug', '--split_repeat', '1']
        params = get_args()
    finally:
        sys.argv = argv
    base = GPM_DIR
    params['data_path'] = str(base / '..' / 'data')
    params['pattern_path'] = str(base / '..' / 'patterns')
    params['save_path'] = str(base / '..' / 'model')
    with open(base / '..' / 'config' / 'data.yaml') as f:
        data_config = yaml.safe_load(f)[dataset]
    params['task'] = data_config['task']
    params['metric'] = data_config['metric']
    params['num_tasks'] = data_config.get('num_tasks', None)
    with open(base / '..' / 'config' / 'main.yaml') as f:
        params.update(yaml.safe_load(f)[params['task']][dataset])
    if params['no_node_pe']:
        params['node_pe'] = 'none'
    if params['no_ap']:
        params['pe_encoder'] = 'none'
    return params


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=DATASETS)
    a.add_argument('--seed', type=int, required=True, help='training seed; 0, 1 and 2 for the three checkpoints')
    a.add_argument('--checkpoint-id', type=int, help='ID in checkpoints.csv; defaults to seed + 1')
    a.add_argument('--epochs', type=int, help='override the epoch limit, for a quick test only')
    a.add_argument('--eval-batch-size', type=int,
                   help='nodes per evaluation batch; defaults to the training batch size. Evaluation uses all '
                        '128 patterns per node, so datasets with many features (CoraFull) need a smaller one. '
                        'It changes memory use only, not the results.')
    a.add_argument('--no-register', action='store_true', help='train without adding to checkpoints.csv')
    args = a.parse_args()
    checkpoint_id = args.seed + 1 if args.checkpoint_id is None else args.checkpoint_id

    use_gpm()
    import torch
    from data.pyg_data_loader import load_data
    from model.model import Model
    from task.node import preprocess_node, train_node, eval_node
    from utils.early_stop import EarlyStopping
    from utils.scheduler import get_scheduler
    from utils.utils import seed_everything

    params = gpm_params(args.dataset)
    if params['task'] != 'node':
        raise ValueError('Only node classification is supported.')
    if args.epochs is not None:
        params['epochs'] = args.epochs
    out_dir = Path(params['save_path']) / args.dataset / f'seed{args.seed}'
    out_file = out_dir / 'best.pt'
    if out_file.exists():
        raise FileExistsError(f'Refusing to overwrite {out_file}')

    # From here to the training loop mirrors main.run().
    seed_everything(42)  # Make sure the split is the same for each run
    assert not (params['node_pe'] != 'none' and params['node_pe_dim'] == 0)
    if params['node_pe'] == 'none':
        params['node_pe_dim'] = 0
    device = torch.device(f"cuda:{params['gpu']}") if torch.cuda.is_available() else torch.device('cpu')
    params['device'] = device
    print('Use Device:', device)

    graph, splits = load_data(params)
    params['input_dim'] = graph.x.size(1)
    params['edge_dim'] = graph.edge_attr.size(1) if graph.edge_attr is not None else 0
    params['output_dim'] = params['num_tasks'] if params.get('num_tasks') is not None else graph.y.max().item() + 1
    split = splits[0]

    start = time.time()
    params['pattern_set'] = preprocess_node(graph, params)
    print(f'Preprocessing time: {time.time() - start:.2f}s')

    seed_everything(args.seed)
    model = Model(params=params).to(device)
    stopper = EarlyStopping(patience=params['early_stop'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'],
                                  betas=(params['opt_beta1'], params['opt_beta2']), eps=params['opt_eps'])
    scheduler = get_scheduler(optimizer, params)
    eval_params = dict(params, batch_size=args.eval_batch_size or params['batch_size'])

    best = dict(epoch=None, val=float('-inf'), test=None)
    best_state, history = None, []
    start = time.time()
    for epoch in range(1, params['epochs'] + 1):
        loss = train_node(graph, model, optimizer, split=split, scheduler=scheduler, params=params)
        if epoch % params['eval_every'] != 0:
            continue
        result = eval_node(graph, model, split=split, params=eval_params)
        history.append(dict(epoch=epoch, loss=loss['train'], val=result['val'], test=result['test']))
        print(f"Epoch {epoch}: loss {loss['train']:.4f}, val {result['val']:.2f}, test {result['test']:.2f}", flush=True)
        if result['val'] > best['val']:  # the same comparison EarlyStopping makes
            best = dict(epoch=epoch, val=result['val'], test=result['test'])
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if stopper(result):
            print('Early Stopping at Epoch:', epoch)
            break
    seconds = time.time() - start
    if best_state is None:
        raise RuntimeError('No evaluation ran, so there is no checkpoint to save. Use more epochs.')

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_file)
    env = environment()
    log = dict(dataset=args.dataset, seed=args.seed, split_index=0, best=best, epochs_run=history[-1]['epoch'],
               training_seconds=seconds, eval_batch_size=eval_params['batch_size'], history=history, checkpoint=relative(out_file), sha256=sha256(out_file),
               **env, params={k: v for k, v in params.items()
                              if k not in ('pattern_set', 'device') and not hasattr(v, 'shape')})
    with open(out_dir / 'train_log.json', 'w') as f:
        json.dump(log, f, indent=2, default=str)
    print(f"Best epoch {best['epoch']}: val {best['val']:.2f}, test {best['test']:.2f}. Saved {out_file}")

    if not args.no_register:
        add_to_registry(dict(dataset=args.dataset, checkpoint_id=checkpoint_id, seed=args.seed, split_index=0,
                             path=relative(out_file), sha256=log['sha256'], best_epoch=best['epoch'],
                             val_acc=round(best['val'], 4), test_acc=round(best['test'], 4),
                             source='train_checkpoint.py', notes='', **env))
        print(f'Registered as {args.dataset} checkpoint {checkpoint_id}.')


if __name__ == '__main__':
    main()
