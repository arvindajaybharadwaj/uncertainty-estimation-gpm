"""Run from GPM/: modal run final_modal.py --smoke"""
import modal
import os
import subprocess

# The existing image definition is needed on the local client only.
# Modal imports this module again inside the worker container.
image = modal.Image.debian_slim()
if modal.is_local():
    import modal_gpm
    image = modal_gpm.image.add_local_file(
        'final_inference.py', '/workspace/final_inference.py', copy=True)

results_volume = modal.Volume.from_name('gpm-results')
volumes = {
    '/workspace/GPM/data': modal.Volume.from_name('gpm-data'),
    '/workspace/GPM/patterns': modal.Volume.from_name('gpm-patterns'),
    '/workspace/GPM/model': modal.Volume.from_name('gpm-models'),
    '/workspace/GPM/results': results_volume,
}


def run_command(command):
    subprocess.run(command, cwd='/workspace/GPM',
                   env=dict(os.environ, PYTHONUNBUFFERED='1'), check=True)

app = modal.App('gpm-final-evaluation')


@app.function(image=image, gpu='L40S', timeout=24 * 60 * 60, volumes=volumes)
def evaluate(smoke: bool = True, repeats: int = 50):
    prefix = '/workspace/GPM/results/final_smoke' if smoke else '/workspace/GPM/results/final_evaluation'
    common = ['/opt/conda/envs/GPM/bin/python', '/workspace/final_inference.py',
              '--gpm-dir', '/workspace/GPM/GPM', '--dataset', 'computers',
              '--checkpoint', '/workspace/GPM/model/computers/epoch_380.pt',
              '--repeats', '2' if smoke else str(repeats), '--batch-size', '256']
    if smoke:
        common += ['--max-nodes-per-split', '32']
    for sampling in ('nested', 'fresh'):
        prediction = f'{prefix}/{sampling}/final_predictions.npz'
        run_command(common + ['--stage', 'predictions', '--sampling', sampling,
                                   '--output', prediction])
        results_volume.commit()
        rules = ('four',) if sampling == 'nested' else ('four', 'two')
        for rule in rules:
            name = 'adaptive_decisions.npz' if rule == 'four' else 'adaptive_decisions_two_stage.npz'
            run_command(common + ['--stage', 'live', '--sampling', sampling,
                                       '--rule', rule, '--calibration', prediction,
                                       '--target', '48', '--seed', '171000',
                                       '--output', f'{prefix}/{sampling}/{name}'])
            results_volume.commit()


@app.local_entrypoint()
def main(smoke: bool = True, repeats: int = 50):
    evaluate.remote(smoke=smoke, repeats=repeats)
