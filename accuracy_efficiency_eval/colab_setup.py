"""Install what GPM needs on Colab. Run before importing torch in the notebook.

Tries the torch Colab already has. If PyG wheels do not exist for it, falls
back to GPM's original torch 2.4.0 build. dgl is left out on purpose: GPM
imports it but never calls it, and measure_timing.py handles its absence.
"""
import subprocess
import sys

COMMON = ['ogb==1.3.6', 'einops==0.8.0', 'torchmetrics==1.6.0', 'wandb', 'googledrivedownloader==0.4', 'psutil', 'pyyaml']
CHECK = ('import torch, torch_geometric, torch_scatter, torch_sparse, torch_cluster, ogb, einops, torchmetrics, wandb, '
         'google_drive_downloader, psutil, yaml; print("torch", torch.__version__, "| PyG", torch_geometric.__version__, '
         '"| GPU available:", torch.cuda.is_available())')
PYG = ['torch_scatter', 'torch_sparse', 'torch_cluster']


def wheel_url(version):
    """'2.4.0+cu121' -> PyG wheel index for that build."""
    if '+' not in version:
        raise ValueError(f'Cannot read the torch build from {version!r}.')
    return f'https://data.pyg.org/whl/torch-{version}.html'


def pip(*args):
    print('pip install', ' '.join(args), flush=True)
    return subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *args]).returncode == 0


def python(code):
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)


def works():
    result = python(CHECK)
    if result.returncode == 0:
        print(result.stdout.strip())
        return True
    print(result.stderr.strip()[-600:])
    return False


def use_current_torch():
    version = python('import torch; print(torch.__version__)').stdout.strip()
    print('Colab torch:', version or 'not found', flush=True)
    if not version:
        return False
    return pip('--only-binary=:all:', *PYG, '-f', wheel_url(version)) and pip('torch_geometric')


def use_original_torch():
    return (pip('torch==2.4.0', '--index-url', 'https://download.pytorch.org/whl/cu121')
            and pip('--only-binary=:all:', *PYG, '-f', wheel_url('2.4.0+cu121'))
            and pip('torch-geometric==2.6.1'))


def main(attempts=(use_current_torch, use_original_torch)):
    if not pip(*COMMON):
        sys.exit('Could not install the common packages. Paste the output above to Claude.')
    for attempt in attempts:
        if attempt() and works():
            print('SETUP OK')
            return
    sys.exit('SETUP FAILED. Paste the output above to Claude.')


if __name__ == '__main__':
    main()
