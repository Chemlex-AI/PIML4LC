"""Portable numerical inputs shared by prediction and inverse entry points."""
from pathlib import Path

import numpy as np
import torch

from src.data.batch import PARA_ORDER, PDEBatch
from src.models.pde_processor import PDEProcessor


def choose_device(requested='auto'):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else torch.device(requested)


def load_parameters(path, config):
    with np.load(path, allow_pickle=False) as data:
        missing = set(PARA_ORDER) - set(data.files)
        if missing:
            raise ValueError(f'Missing physical parameters: {sorted(missing)}')
        values = {key: np.asarray(data[key], dtype=np.float64) for key in PARA_ORDER}
    shapes = {value.shape for value in values.values()}
    if len(shapes) != 1 or any(value.ndim != 1 or value.size == 0 for value in values.values()):
        raise ValueError('Every parameter must be a nonempty vector of the same length.')
    processor = PDEProcessor(config.paras)
    for key, value in values.items():
        lo, hi = processor.physical_bounds(key)
        if not np.isfinite(value).all() or (value < lo).any() or (value > hi).any():
            raise ValueError(f'{key} must be finite and within [{lo}, {hi}] in physical units.')
    return PDEBatch(**{key: torch.as_tensor(value, dtype=torch.float32) for key, value in values.items()})


def load_target(path, n_time, config):
    with np.load(path, allow_pickle=False) as data:
        end_con = np.asarray(data['end_con'], dtype=np.float64)
        end_time = np.asarray(data['end_time'], dtype=np.float64)
        target = np.asarray(data['outlet'], dtype=np.float32)
    if end_con.ndim != 1 or not end_con.size or end_time.shape != end_con.shape:
        raise ValueError('end_con and end_time must be equal-length nonempty vectors.')
    if target.shape != (n_time, end_con.size):
        raise ValueError(f'outlet must have shape ({n_time}, {end_con.size}); got {target.shape}.')
    if not np.isfinite(target).all() or (target < 0).any():
        raise ValueError('outlet must contain finite nonnegative dimensionless concentrations.')
    processor = PDEProcessor(config.paras)
    for key, value in [('end_con', end_con), ('end_time', end_time)]:
        lo, hi = processor.physical_bounds(key)
        if not np.isfinite(value).all() or (value < lo).any() or (value > hi).any():
            raise ValueError(f'{key} is outside the trained operating range [{lo}, {hi}].')
    return end_con, end_time, target


def output_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
