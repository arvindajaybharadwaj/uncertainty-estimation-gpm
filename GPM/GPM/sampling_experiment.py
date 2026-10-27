import argparse
import copy
import os
import os.path as osp
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch_geometric.data import Data, InMemoryDataset

from data.pyg_data_loader import load_data, mol_graphs
from model.model import Model
from model.random_walk import get_patterns
from utils.utils import seed_everything, mask2idx


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="computers")
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[4, 8, 16, 32, 64, 128],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="../results/computers_sampling.npz",
    )

    parser.add_argument(
        "--base_seed",
        type=int,
        default=10000,
    )

    # useful for a quick smoke test
    parser.add_argument(
        "--max_test_nodes",
        type=int,
        default=None,
    )

    return parser.parse_args()


def set_sampling_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_baseline_params(dataset):
    """
    Reconstruct the same configuration used by:
        python GPM/main.py --dataset computers --use_params
    """

    base_dir = osp.dirname(__file__)
    config_dir = osp.join(base_dir, "..", "config")

    # These are the defaults from args.py.
    params = {
        "gpu": 0,
        "dataset": dataset,

        "epochs": 1000,
        "lr": 1e-2,
        "batch_size": 256,
        "split": "public",
        "split_repeat": 3,
        "eval_every": 10,

        "pre_sample_pattern_num": 128,
        "pre_sample_batch_size": 8192,
        "num_patterns": 16,
        "pattern_size": 8,
        "multiscale": [2, 4, 6, 8],
        "p": 1.0,
        "q": 1.0,

        "hidden_dim": 256,
        "heads": 4,
        "num_layers": 1,
        "norm_first": False,

        "node_pe": "rw",
        "node_pe_dim": 8,
        "pe_encoder": "gru",
        "pe_weight": 1.0,

        "pattern_encoder": "transformer",
        "pattern_encoder_layers": 1,
        "pattern_encoder_heads": 1,

        "use_vq": False,
        "codebook_size": 8196,
        "use_cls_token": False,
        "use_attn_fusion": False,

        "dropout": 0.1,
        "weight_decay": 0.0,
        "early_stop": 10,
        "label_smoothing": 0.05,
        "grad_clip": 1.0,
        "scheduler": "warmup",
        "warmup_steps": 100,
        "eta_min": 1e-6,

        "opt_beta1": 0.9,
        "opt_beta2": 0.999,
        "opt_eps": 1e-8,

        "pretrain_data": "none",
        "pretrain_epoch": 0,
        "linear_probe": False,
        "reset_head": False,
        "inference": True,
    }

    # Dataset/task metadata.
    with open(osp.join(config_dir, "data.yaml"), "r") as f:
        data_config = yaml.safe_load(f)

    params["task"] = data_config[dataset]["task"]
    params["metric"] = data_config[dataset]["metric"]
    params["num_tasks"] = data_config[dataset].get("num_tasks", None)

    # This reproduces --use_params.
    with open(osp.join(config_dir, "main.yaml"), "r") as f:
        tuned = yaml.safe_load(f)

    params.update(tuned[params["task"]][dataset])

    params["data_path"] = osp.join(base_dir, "..", "data")
    params["pattern_path"] = osp.join(base_dir, "..", "patterns")
    params["save_path"] = osp.join(base_dir, "..", "model")

    return params


def main():
    args = parse_args()

    # ---------------------------------------------------
    # Reproduce the same dataset split as baseline run 1
    # ---------------------------------------------------

    seed_everything(42)

    params = load_baseline_params(args.dataset)

    device = (
        torch.device("cuda:0")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    params["device"] = device

    print("Device:", device)

    graph, splits = load_data(params)

    # Same first split that baseline run 1 used.
    if splits is None:
        raise RuntimeError("Expected node-classification splits.")

    split = splits[0]

    # ---------------------------------------------------
    # Infer dimensions exactly as main.py does
    # ---------------------------------------------------

    if isinstance(graph, Data):
        params["input_dim"] = graph.x.size(1)
        params["edge_dim"] = (
            graph.edge_attr.size(1)
            if graph.edge_attr is not None
            else 0
        )

    elif isinstance(graph, InMemoryDataset):
        params["input_dim"] = graph._data.x_feat.size(1)
        params["edge_dim"] = (
            graph._data.e_feat.size(1)
            if graph._data.edge_attr is not None
            else 0
        )

    if params.get("num_tasks") is not None:
        params["output_dim"] = params["num_tasks"]
    else:
        params["output_dim"] = graph.y.max().item() + 1

    # ---------------------------------------------------
    # Load and freeze model
    # ---------------------------------------------------

    model = Model(params=params).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    print("Loaded checkpoint:", args.checkpoint)

    # ---------------------------------------------------
    # Test nodes
    # ---------------------------------------------------

    test_idx = mask2idx(split["test"])

    if args.max_test_nodes is not None:
        test_idx = test_idx[:args.max_test_nodes]

    labels = graph.y[test_idx].cpu().numpy()

    n_test = len(test_idx)
    n_classes = params["output_dim"]

    print("Test nodes:", n_test)
    print("Classes:", n_classes)
    print("K values:", args.k_values)
    print("Repeats:", args.repeats)

    # Shape:
    # [num_K, repeats, test_nodes, classes]
    probabilities = np.full(
        (
            len(args.k_values),
            args.repeats,
            n_test,
            n_classes,
        ),
        np.nan,
        dtype=np.float32,
    )

    # ---------------------------------------------------
    # Core stochastic experiment
    # ---------------------------------------------------

    output = osp.abspath(
        osp.join(
            osp.dirname(__file__),
            args.output,
        )
    )
    
    os.makedirs(
            osp.dirname(output),
            exist_ok=True,
    )

    with torch.no_grad():

        experiment_start = time.time()
        for k_idx, K in enumerate(args.k_values):

            print(f"\n===== K = {K} =====")

            for r in range(args.repeats):

                # deterministic but different random-walk sample
                # for every (K, repetition)
                sample_seed = (
                    args.base_seed
                    + K * 1000
                    + r
                )

                set_sampling_seed(sample_seed)

                all_batch_probs = []

                num_batches = (
                    n_test + args.batch_size - 1
                ) // args.batch_size

                for b in range(num_batches):

                    global_nodes = test_idx[
                        b * args.batch_size:
                        (b + 1) * args.batch_size
                    ]

                    if len(global_nodes) == 0:
                        continue

                    # ------------------------------------
                    # Generate FRESH random walks
                    # ------------------------------------

                    run_params = copy.copy(params)

                    run_params["pre_sample_pattern_num"] = K

                    patterns, eids = get_patterns(
                        graph,
                        run_params,
                        nodes=global_nodes,
                    )

                    # Important:
                    # pattern columns are now LOCAL to this
                    # particular batch.
                    run_params["pattern_set"] = {
                        "pattern": patterns,
                        "eid": eids,
                    }

                    local_nodes = torch.arange(
                        len(global_nodes),
                        dtype=torch.long,
                    )

                    logits, _, _, _ = model(
                        graph,
                        local_nodes,
                        run_params,
                        mode="eval",
                    )

                    probs = F.softmax(
                        logits,
                        dim=-1,
                    )

                    all_batch_probs.append(
                        probs.cpu()
                    )

                repeat_probs = torch.cat(
                    all_batch_probs,
                    dim=0,
                ).numpy()

                probabilities[k_idx, r] = repeat_probs

                predicted = repeat_probs.argmax(axis=1)

                accuracy = (
                    predicted == labels
                ).mean()

                elapsed = time.time() - experiment_start
                print(
                    f"K={K:3d}, "
                    f"repeat={r + 1:02d}/{args.repeats}, "
                    f"accuracy={accuracy:.4f}",
                    f"elapsed={elapsed / 60:.1f} min"
                )

            np.savez_compressed(
                output,
                probabilities=probabilities,
                labels=labels,
                node_ids=test_idx.cpu().numpy(),
                k_values=np.asarray(args.k_values),
                repeats=args.repeats,
                checkpoint=args.checkpoint,
                completed_k_index=k_idx,
            )

    print("\nExperiment complete.")
    print("Saved:", output)
    print("Probability tensor:", probabilities.shape)


if __name__ == "__main__":
    main()