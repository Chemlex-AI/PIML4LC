"""Estimate nine shared physical parameters from one or more gradient curves."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.batch import PARA_ORDER, PDEBatch
from src.io_utils import choose_device, load_target
from src.load_model import load_model
from src.models.inverse.ga import SimpleGA
from src.models.inverse.gd import LBFGSInverse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--target', required=True, help='NPZ: outlet (time, experiments), end_con, end_time')
    parser.add_argument('--method', choices=['lbfgs', 'ga'], default='ga')
    parser.add_argument('--output-dir', default='results/inverse')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--config', default=None)
    parser.add_argument('--parameter-config', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-starts', type=int, default=8)
    parser.add_argument('--max-iter', type=int, default=30)
    parser.add_argument('--population', type=int, default=1024)
    parser.add_argument('--generations', type=int, default=15)
    parser.add_argument('--top-n', type=int, default=128)
    parser.add_argument('--unique-eps', type=float, default=0.1)
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--zero-first-n', type=int, default=10)
    parser.add_argument('--target-peak-weight', type=float, default=20.0)
    parser.add_argument('--peak-position-weight', type=float, default=5.0)
    parser.add_argument('--peak-height-weight', type=float, default=1.0)
    args = parser.parse_args()
    if min(args.n_starts, args.max_iter, args.chunk_size, args.generations, args.top_n) <= 0 or args.population < 2 or args.unique_eps < 0:
        parser.error('Counts must be positive; population must be at least two.')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    solver, _, config = load_model(args.checkpoint, choose_device(args.device), args.config, args.parameter_config)
    end_con, end_time, target = load_target(args.target, solver.model.n_time_full, config)
    if not 0 <= args.zero_first_n < len(target):
        parser.error('--zero-first-n must be in [0, number of time points).')
    for parameter in solver.parameters():
        parameter.requires_grad_(False)
    if args.method == 'lbfgs':
        algorithm = LBFGSInverse(
            solver, config.paras, zero_first_n=args.zero_first_n,
            max_iter=args.max_iter, lr=0.5, history_size=20,
            tolerance_grad=1e-7, tolerance_change=1e-9,
            target_peak_weight=args.target_peak_weight,
            peak_position_weight=args.peak_position_weight,
            peak_height_weight=args.peak_height_weight,
        )
        algorithm.pred_chunk_size = args.chunk_size
        result = algorithm.optimize(end_con, end_time, target, n_starts=args.n_starts, seed=args.seed)
        parameters, losses = result['all_paras'], result['losses']
        predictions, history = result['model_pred'], result['history']
    else:
        algorithm = SimpleGA(
            solver, config.paras, zero_first_n=args.zero_first_n,
            loss_weights={'mse': .05, 'penalty_on_0': 1., 'peak_pos': 1., 'peak_v': 1., 'tv_penalty': 1.},
            sol_per_pop=args.population, num_generations=args.generations,
            mutation_probability=.2, max_copies=5,
        )
        algorithm.pred_chunk_size = args.chunk_size
        parameters, losses, _, history = algorithm.optimize(end_con, end_time, target, top_n=1.0)
        physical = {key: parameters[:, i] for i, key in enumerate(PARA_ORDER) if key not in ['end_con', 'end_time']}
        normalized = algorithm.pde_processor.dict_norm_01(physical)
        points = torch.stack(list(normalized.values()), dim=-1).cpu().numpy()
        keep = algorithm.unique_indices(points, args.unique_eps)[:args.top_n]
        parameters, losses = parameters[keep], losses[keep]
        predictions = algorithm.model_pred(PDEBatch(**dict(zip(PARA_ORDER, parameters.T))), end_con=end_con, end_time=end_time)
        predictions[:, :args.zero_first_n] = 0
        history = [value.tolist() for value in history]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'parameters.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['rank', 'loss', *PARA_ORDER])
        writer.writerows([i + 1, float(loss), *row.tolist()] for i, (loss, row) in enumerate(zip(losses, parameters)))
    np.savez_compressed(out / 'predictions.npz', outlet=predictions, target=target, end_con=end_con, end_time=end_time)
    (out / 'run.json').write_text(json.dumps({'arguments': vars(args), 'history': history}, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8, 4))
    time = np.arange(1, len(target) + 1)
    for exp in range(target.shape[1]):
        line, = ax.plot(time, target[:, exp], label=f'target {exp + 1}')
        ax.plot(time, predictions[0, :, exp], '--', color=line.get_color(), label=f'fit {exp + 1}')
    ax.set(xlabel='Time (s)', ylabel='Dimensionless outlet concentration')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'overlay.png', dpi=160)
    plt.close(fig)
    print(f'Saved ranked physical parameters and curves to {out}')


if __name__ == '__main__':
    main()
