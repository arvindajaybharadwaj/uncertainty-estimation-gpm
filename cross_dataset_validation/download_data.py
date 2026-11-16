"""Download and preprocess one dataset with GPM's own loader, then print its split.

PyTorch Geometric downloads the raw files into GPM/data on first use. Loading
here uses the same tuned settings as training and inference, so the processed
files and Laplacian positional encodings match what the other scripts expect.

Example, from the repository root:
    python cross_dataset_validation/download_data.py --dataset wikics
"""
import argparse

from common import DATASETS, use_gpm


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--dataset', required=True, choices=DATASETS)
    args = a.parse_args()

    use_gpm()
    from data.pyg_data_loader import load_data
    from sampling_experiment import load_baseline_params
    from utils.utils import seed_everything

    seed_everything(42)  # the split is drawn after this seed, as in main.py
    params = load_baseline_params(args.dataset)
    params['device'] = 'cpu'
    graph, splits = load_data(params)
    split = splits[0]
    print(f'{args.dataset}: {graph.num_nodes:,} nodes, {graph.num_edges:,} edges, '
          f'{graph.x.size(1)} features, {int(graph.y.max()) + 1} classes')
    print(f"Split 0 ({params['split']}): train {int(split['train'].sum()):,}, "
          f"val {int(split['val'].sum()):,}, test {int(split['test'].sum()):,}")
    print(f"Data folder: {params['data_path']}")


if __name__ == '__main__':
    main()
