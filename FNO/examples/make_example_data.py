"""Generate artificial Gaussian profiles for software checks, without a PDE solver.

These profiles are NOT GRM solutions and cannot reproduce manuscript results.
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf


def generate(output, n_samples=20, n_time=12, n_z=17, n_r=2, seed=42):
    if n_samples < 10 or min(n_time, n_z, n_r) < 1:
        raise ValueError('Use >=10 samples for nonempty splits and positive grid sizes.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / 'data/para_info.yaml')
    rng = np.random.default_rng(seed)
    parameters = {}
    for key, spec in cfg.paras.items():
        if 'choices' in spec:
            value = rng.choice(list(spec.choices), n_samples)
        else:
            value = rng.uniform(spec.min, spec.max, n_samples)
            if spec.log_scale:
                value = 10.0 ** value
        parameters[key] = value.astype(np.float64)
    time = np.arange(1, n_time + 1)[None, :, None]
    z = np.linspace(0, 1, n_z)[None, None, :]
    delay = n_time * (.15 + .2 * parameters['sa'][:, None, None] / 10)
    bul = np.exp(-.5 * ((time - delay - .4 * n_time * z) / max(1, .1 * n_time)) ** 2).astype(np.float32)
    radial = np.linspace(.7, 1., n_r)[None, None, None, :]
    par = (bul[..., None] * radial / float(cfg.field_norm.par)).reshape(n_samples, n_time, -1).astype(np.float32)
    sol = (.5 * bul[..., None] * radial / float(cfg.field_norm.sol)).reshape(n_samples, n_time, -1).astype(np.float32)
    with h5py.File(output / 'example.h5', 'w') as data:
        data.attrs['description'] = 'Artificial Gaussian fields for software smoke tests; not GRM/FEM data.'
        data.attrs['seed'] = seed
        for key, value in {**parameters, 'bul': bul, 'par': par, 'sol': sol}.items():
            data.create_dataset(key, data=value)
    # Small prediction batch; inversion uses three distinct operating programs
    # and one shared synthetic physical parameter set.
    np.savez_compressed(output / 'parameters.npz', **{key: value[:2] for key, value in parameters.items()})
    end_con = np.array([.95, .95, .95])
    end_time = np.array([120., 210., 270.])
    time_axis = np.arange(1, n_time + 1)[:, None]
    centers = n_time * np.array([.45, .55, .65])[None, :]
    outlet = np.exp(-.5 * ((time_axis - centers) / max(1, .1 * n_time)) ** 2).astype(np.float32)
    np.savez_compressed(output / 'target.npz', end_con=end_con, end_time=end_time, outlet=outlet)
    print(f'Wrote artificial data to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data')
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--n-time', type=int, default=12)
    parser.add_argument('--n-z', type=int, default=17)
    parser.add_argument('--n-r', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    generate(args.output, args.samples, args.n_time, args.n_z, args.n_r, args.seed)


if __name__ == '__main__':
    main()
