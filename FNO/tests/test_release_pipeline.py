"""One small real train/save/relocate/load/predict/GA/L-BFGS integration run."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

from src.load_model import load_model

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    result = subprocess.run([sys.executable, *map(str, arguments)], cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_source_checkout_pipeline(tmp_path):
    data = tmp_path / 'data'
    run('examples/make_example_data.py', '--output', data)
    training = tmp_path / 'training'
    run('-m', 'src.train', 'experiment=smoke', f'data.h5_file_path={data.as_posix()}/example.h5', f'hydra.run.dir={training.as_posix()}')
    relocated = tmp_path / 'relocated'
    shutil.copytree(training, relocated)
    saved = relocated / '.hydra/config.yaml'
    cfg = OmegaConf.load(saved)
    cfg.data.paras_cfg_path = '/missing/research/machine/para_info.yaml'
    OmegaConf.save(cfg, saved)
    checkpoint = relocated / 'checkpoints/best_epoch000.ckpt'
    model, _, params = load_model(checkpoint)
    assert model.model.n_time_full == 12 and float(params.field_norm.par) == .1
    prediction = tmp_path / 'prediction.npz'
    run('-m', 'src.predict', '--checkpoint', checkpoint, '--parameters', data / 'parameters.npz', '--output', prediction)
    with np.load(prediction, allow_pickle=False) as output:
        assert output['bul'].shape == (2, 12, 17)
        assert output['par'].shape == (2, 12, 34)
        assert np.isfinite(output['outlet']).all()
    for method in ['ga', 'lbfgs']:
        out = tmp_path / method
        run('-m', 'src.inverse', '--checkpoint', checkpoint, '--target', data / 'target.npz',
            '--method', method, '--population', '4', '--generations', '2', '--n-starts', '3',
            '--chunk-size', '1', '--max-iter', '2', '--zero-first-n', '0', '--output-dir', out)
        assert (out / 'parameters.csv').is_file()
        assert (out / 'run.json').is_file()
        with np.load(out / 'predictions.npz', allow_pickle=False) as output:
            assert output['outlet'].shape[1:] == (12, 3)
            assert np.isfinite(output['outlet']).all()
