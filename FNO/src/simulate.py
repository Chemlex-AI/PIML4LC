"""Evaluate user-supplied physical parameters with an optional reference solver."""
import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

from src.io_utils import load_parameters, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['comsol', 'cadet'], default='comsol')
    parser.add_argument('--parameters', required=True)
    parser.add_argument('--parameter-config', default=str(Path(__file__).resolve().parents[1] / 'data/para_info.yaml'))
    parser.add_argument('--output', default='data/train.h5')
    parser.add_argument('--model', help='Path to the licensed COMSOL GRM.mph')
    parser.add_argument('--n-time', type=int, default=360)
    parser.add_argument('--cores', type=int, default=1)
    args = parser.parse_args()
    if min(args.n_time, args.cores) <= 0:
        parser.error('--n-time and --cores must be positive.')
    if args.model:
        os.environ['PIML4LC_COMSOL_MODEL'] = str(Path(args.model).resolve())
    if args.backend == 'comsol':
        from src.models.simulator.comsol import gradient_solve, reset_client
    else:
        from src.models.simulator.cadet import gradient_solve, reset_client
    parameters = load_parameters(args.parameters, OmegaConf.load(args.parameter_config))
    output = output_path(args.output)
    # Exclusive creation prevents accidental replacement of a reference corpus.
    try:
        with h5py.File(output, 'x') as data:
            data.attrs['backend'] = args.backend
            data.attrs['n_time'] = args.n_time
            for key, value in parameters.solver_kwargs().items():
                data.create_dataset(key, data=value.numpy())
            for sample in range(len(parameters)):
                fields = gradient_solve(**parameters.solver_kwargs(sample), n_time=args.n_time, n_cores=args.cores, clip=False, bul_out_only=False)
                if fields is None:
                    raise RuntimeError(f'{args.backend} failed at sample {sample}; discard incomplete output.')
                for key in ['bul', 'par', 'sol']:
                    value = np.asarray(fields[key])
                    if sample == 0:
                        data.create_dataset(key, shape=(len(parameters), *value.shape), dtype='f4')
                    data[key][sample] = value
                data.attrs['completed_samples'] = sample + 1
    finally:
        reset_client()
    print(f'Saved reference fields to {output}; match the training grid to their shapes.')


if __name__ == '__main__':
    main()
