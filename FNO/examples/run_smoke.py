"""Run the documented artificial example on CPU from any working directory."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    commands = [
        ['examples/make_example_data.py'],
        ['-m', 'src.train', 'experiment=smoke', 'hydra.run.dir=logs/smoke/run'],
        ['-m', 'src.predict', '--checkpoint', 'logs/smoke/run/checkpoints/best_epoch000.ckpt',
         '--parameters', 'data/parameters.npz', '--output', 'results/smoke/prediction.npz'],
    ]
    for method in ['ga', 'lbfgs']:
        commands.append(['-m', 'src.inverse', '--checkpoint', 'logs/smoke/run/checkpoints/best_epoch000.ckpt',
            '--target', 'data/target.npz', '--method', method, '--population', '4', '--generations', '2',
            '--n-starts', '3', '--chunk-size', '1', '--max-iter', '2', '--zero-first-n', '0',
            '--output-dir', f'results/smoke/{method}'])
    for command in commands:
        subprocess.run([sys.executable, *command], cwd=ROOT, env=env, check=True)
    print('Artificial train/save/load/predict/GA/L-BFGS workflow completed.')


if __name__ == '__main__':
    main()
