"""Re-rank FNO candidates with COMSOL gradient runs, then test isocratic transfer.

Requires a licensed COMSOL installation and the compatible GRM model. Candidate
selection uses gradient curves only. Isocratic data never influence selection.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from src.models.inverse.gd import PHYSICAL_KEYS


def curve_scores(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if prediction.shape != target.shape or target.ndim != 2:
        raise ValueError('Prediction and target must have equal (time, experiments) shapes.')
    valid = target.max(axis=0) > 0
    if not valid.any() or not np.isfinite(prediction).all():
        raise ValueError('Need finite predictions and at least one observed peak.')
    delta = prediction[:, valid].argmax(axis=0) - target[:, valid].argmax(axis=0)
    scale = target[:, valid].max(axis=0) + 1e-5
    return {
        'retention_time_rmse_s': float(np.sqrt(np.mean(delta ** 2))),
        'normalized_profile_mse': float(np.mean(((prediction[:, valid] - target[:, valid]) / scale) ** 2)),
        'peak_height_ratio': (prediction[:, valid].max(axis=0) / scale).tolist(),
        'peak_time_difference_s': delta.tolist(),
        'included_experiments': np.flatnonzero(valid).tolist(),
    }


def read_target(path, isocratic=False):
    with np.load(path, allow_pickle=False) as data:
        target = np.asarray(data['outlet'], dtype=np.float64)
        operating = np.asarray(data['mobile_fraction' if isocratic else 'end_con'], dtype=np.float64)
        duration = None if isocratic else np.asarray(data['end_time'], dtype=np.float64)
    if target.ndim != 2 or operating.shape != (target.shape[1],) or not operating.size:
        raise ValueError('Target requires outlet (time, experiments) and matching operating vectors.')
    if not np.isfinite(target).all() or (target < 0).any() or not np.isfinite(operating).all():
        raise ValueError('Target and operating vectors must be finite; outlet must be nonnegative.')
    if (operating < 0).any() or (operating > 1).any():
        raise ValueError('Mobile phase fractions must be in [0, 1].')
    if not isocratic and (duration.shape != operating.shape or not np.isfinite(duration).all() or (duration <= 0).any()):
        raise ValueError('end_time must contain one finite positive duration per gradient.')
    return target, operating, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True, help='parameters.csv from src.inverse')
    parser.add_argument('--gradient-target', required=True)
    parser.add_argument('--isocratic-target', help='NPZ: outlet (time, experiments), mobile_fraction')
    parser.add_argument('--isocratic-kd-scale', type=float, default=1., help='Explicit protocol adjustment; manuscript uses 2.')
    parser.add_argument('--top-k', type=int, default=8)
    parser.add_argument('--cores', type=int, default=1)
    parser.add_argument('--model', help='COMSOL GRM.mph path')
    parser.add_argument('--output-dir', default='results/comsol_verification')
    args = parser.parse_args()
    if min(args.top_k, args.cores) <= 0 or not np.isfinite(args.isocratic_kd_scale) or args.isocratic_kd_scale <= 0:
        parser.error('Counts and desorption scale must be finite and positive.')
    if args.model:
        os.environ['PIML4LC_COMSOL_MODEL'] = str(Path(args.model).resolve())
    from src.models.simulator.comsol import gradient_solve, isocratic_solve, reset_client
    target, end_con, end_time = read_target(args.gradient_target)
    with open(args.candidates, newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))[:args.top_k]
    if not rows:
        raise ValueError('No candidate parameter rows.')
    records, predictions = [], []
    try:
        for index, row in enumerate(rows):
            physical = {key: float(row[key]) for key in PHYSICAL_KEYS}
            curves = [gradient_solve(**physical, end_con=float(con), end_time=int(duration),
                      n_time=len(target), n_cores=args.cores, clip=False, bul_out_only=True)
                      for con, duration in zip(end_con, end_time)]
            if any(curve is None for curve in curves):
                raise RuntimeError(f'COMSOL failed for candidate {index + 1}.')
            prediction = np.stack(curves, axis=-1)
            predictions.append(prediction)
            records.append({'fno_rank': int(row.get('rank', index + 1)), 'physical_parameters': physical,
                            **curve_scores(prediction, target)})
        selected = min(range(len(records)), key=lambda i: (records[i]['retention_time_rmse_s'], records[i]['fno_rank']))
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / 'gradient_predictions.npz', outlet=np.stack(predictions), target=target)
        report = {'selection': 'minimum gradient retention-time RMSE; ties use FNO rank',
                  'selected_fno_rank': records[selected]['fno_rank'], 'candidates': records}
        if args.isocratic_target:
            iso_target, fractions, _ = read_target(args.isocratic_target, isocratic=True)
            physical = dict(records[selected]['physical_parameters'])
            physical['kd'] *= args.isocratic_kd_scale
            iso_prediction = np.stack([isocratic_solve(**physical, end_p=float(fraction), n_time=len(iso_target),
                n_cores=args.cores, clip=False, bul_out_only=True) for fraction in fractions], axis=-1)
            report['isocratic_kd_scale'] = args.isocratic_kd_scale
            report['isocratic'] = curve_scores(iso_prediction, iso_target)
            np.savez_compressed(out / 'isocratic_predictions.npz', outlet=iso_prediction, target=iso_target)
        (out / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    finally:
        reset_client()
    print(f'Saved COMSOL candidate selection and independent validation to {args.output_dir}')


if __name__ == '__main__':
    main()
