"""Predict fields and outlet curves from physical parameter vectors."""
import argparse
import numpy as np
import torch

from src.io_utils import choose_device, load_parameters, output_path
from src.load_model import load_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--parameters', required=True, help='NPZ: one vector per physical parameter')
    parser.add_argument('--output', default='results/prediction.npz')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--config', default=None)
    parser.add_argument('--parameter-config', default=None)
    args = parser.parse_args()
    model, _, config = load_model(args.checkpoint, choose_device(args.device), args.config, args.parameter_config)
    batch = load_parameters(args.parameters, config)
    static, rates = batch.solver_input(model.model.n_time, config.paras, model.model.time_downsample)
    device = next(model.parameters()).device
    with torch.inference_mode():
        fields = model.model(static.float().to(device), rates.float().to(device))
    if isinstance(fields, tuple):
        values = {key: field.cpu().numpy() / float(config.field_norm[key]) for key, field in zip(['bul', 'par', 'sol'], fields)}
    else:
        values = {'bul': fields.squeeze(-1).cpu().numpy()}
    # Export raw fields; outlet follows the nonnegative inference convention.
    values['outlet'] = np.maximum(values['bul'][..., -1], 0)
    values['time_s'] = (np.arange(model.model.n_time) + 1) * model.model.time_downsample
    np.savez_compressed(output_path(args.output), **values)
    print(f'Saved predictions to {args.output}')


if __name__ == '__main__':
    main()
