import os
import subprocess

import modal


APP_NAME = "gpm-computers"
REPO = "/workspace/GPM"
GPM_PYTHON = "/opt/conda/envs/GPM/bin/python"
GPM_PIP = "/opt/conda/envs/GPM/bin/pip"


# ---------------------------------------------------------
# Persistent storage
# ---------------------------------------------------------

data_volume = modal.Volume.from_name(
    "gpm-data",
    create_if_missing=True,
)

patterns_volume = modal.Volume.from_name(
    "gpm-patterns",
    create_if_missing=True,
)

models_volume = modal.Volume.from_name(
    "gpm-models",
    create_if_missing=True,
)

results_volume = modal.Volume.from_name(
    "gpm-results",
    create_if_missing=True,
)

# ---------------------------------------------------------
# Build the environment
# ---------------------------------------------------------

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-devel-ubuntu22.04",
        add_python="3.11",
    )

    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "bzip2",
    )

    # Copy original GPM environment.yml
    .add_local_file(
        "environment.yml",
        "/tmp/environment.yml",
        copy=True,
    )

    # Install Miniforge
    .run_commands(
        "curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh"
    )

    .run_commands(
        "bash /tmp/miniforge.sh -b -p /opt/conda"
    )

    # Show Conda info first, useful for debugging
    .run_commands(
        "/opt/conda/bin/conda --version"
    )

    # Accept Anaconda channel ToS if needed
    .run_commands(
        "/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true"
    )
    .run_commands(
        "/opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true"
    )

    # Create the repo's original GPM environment
    .run_commands(
        "/opt/conda/bin/conda env create -f /tmp/environment.yml"
    )

    # DGL exactly as requested by GPM README
    .run_commands(
        f"{GPM_PIP} install 'dgl==2.4.0+cu121' -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html"
    )

    # PyG CUDA extensions
    .run_commands(
        f"{GPM_PIP} install "
        "pyg_lib "
        "torch_scatter "
        "torch_sparse "
        "torch_cluster "
        "torch_spline_conv "
        "-f https://data.pyg.org/whl/torch-2.4.0+cu121.html"
    )

    # Sanity-check installed libraries
    .run_commands(
        f'{GPM_PYTHON} -c "import torch; import torch_geometric; import dgl; '
        'print(\'torch:\', torch.__version__); '
        'print(\'torch_geometric:\', torch_geometric.__version__); '
        'print(\'dgl:\', dgl.__version__)"'
    )

    # Copy repo only after dependencies have been installed.
    .add_local_dir(
        ".",
        REPO,
        copy=True,
        ignore=[
            ".git/**",
            ".modal-venv/**",
            ".venv/**",
            "__pycache__/**",
            "**/__pycache__/**",
            "data/**",
            "patterns/**",
        ],
    )
)


app = modal.App(APP_NAME)


VOLUMES = {
    # These match the paths that GPM/main.py itself uses.
    f"{REPO}/data": data_volume,
    f"{REPO}/patterns": patterns_volume,
    f"{REPO}/model": models_volume,
    f"{REPO}/results": results_volume,
}


def run_command(command):
    print("\nRunning:")
    print(" ".join(command))
    print()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    subprocess.run(
        command,
        cwd=REPO,
        env=env,
        check=True,
    )


# ---------------------------------------------------------
# Step 1: verify GPU + packages
# ---------------------------------------------------------

@app.function(
    image=image,
    gpu="L40S",
    timeout=30 * 60,
    volumes=VOLUMES,
)
def verify():
    subprocess.run(["nvidia-smi"], check=True)

    run_command(
        [
            GPM_PYTHON,
            "-c",
            """
import torch
import torch_geometric
import torch_scatter
import dgl

print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("DGL:", dgl.__version__)
print("CUDA runtime seen by torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
        ]
    )


# ---------------------------------------------------------
# Step 2: reproduce official Computers baseline
# ---------------------------------------------------------

@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
)
def baseline():
    run_command(
        [
            GPM_PYTHON,
            "GPM/main.py",
            "--dataset",
            "computers",
            "--use_params",

            # In the current repo this only disables wandb online logging.
            "--debug",
        ]
    )

    # Persist downloaded data and generated random-walk patterns.
    data_volume.commit()
    patterns_volume.commit()
    models_volume.commit()


# ---------------------------------------------------------
# Step 3: train ONE model and save checkpoints
# This will be the model we later freeze for the U_sampling experiment.
# ---------------------------------------------------------

@app.function(
    image=image,
    gpu="L40S",
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
)
def train_experiment_model():
    run_command(
        [
            GPM_PYTHON,
            "GPM/main.py",
            "--dataset",
            "computers",
            "--use_params",
            "--debug",

            # We need one frozen trained model, not three repetitions.
            "--split_repeat",
            "1",

            # Evaluation occurs every 10 epochs by default.
            # Save checkpoints at the same cadence.
            "--save_every",
            "10",
        ]
    )

    data_volume.commit()
    patterns_volume.commit()
    models_volume.commit()

# smoke test

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    volumes=VOLUMES,
)
def sampling_smoke():

    run_command(
        [
            GPM_PYTHON,
            "GPM/sampling_experiment.py",

            "--dataset",
            "computers",

            "--checkpoint",
            "/workspace/GPM/model/computers/epoch_380.pt",

            "--k_values",
            "16",
            "128",

            "--repeats",
            "3",

            "--max_test_nodes",
            "100",

            "--output",
            "../results/sampling_smoke.npz",
        ]
    )

    results_volume.commit()

# actual experiment

@app.function(
    image=image,
    gpu="L40S",
    timeout=6 * 60 * 60,
    volumes=VOLUMES,
)
def sampling_full():

    run_command(
        [
            GPM_PYTHON,
            "GPM/sampling_experiment.py",

            "--dataset",
            "computers",

            "--checkpoint",
            "/workspace/GPM/model/computers/epoch_380.pt",

            "--k_values",
            "4",
            "8",
            "16",
            "32",
            "64",
            "128",

            "--repeats",
            "50",

            "--batch_size",
            "256",

            "--output",
            "../results/computers_sampling_full.npz",
        ]
    )

    results_volume.commit()