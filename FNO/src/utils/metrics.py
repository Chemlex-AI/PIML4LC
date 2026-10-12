from __future__ import annotations

import numpy as np

from numpy import ndarray

__all__ = [
    'normalize_maxabs_error',
    'relative_error',
    'max_relative_error',
]

def normalize_maxabs_error(y_pred: ndarray, y_gt: ndarray, eps: float = 1e-10) -> ndarray:
    # shape == (batch_size, n_time)
    gt_max = np.max(y_gt, axis=1, keepdims=True)
    gt_max = np.max(gt_max, axis=1, keepdims=True)
    pred_normalized = y_pred / (eps + gt_max)
    gt_normalized = y_gt / (eps + gt_max)

    return np.abs(pred_normalized - gt_normalized).mean(-1).mean(-1)

def relative_error(y_pred: ndarray, y_true: ndarray, lb: float = 1e-4, epsilon: float = 1e-8) -> ndarray:
    mask = y_true > lb

    relative_err = np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon)
    relative_err = relative_err * mask

    valid_entry_count = np.sum(mask, -1)
    relative_err = np.sum(relative_err, -1) / np.maximum(valid_entry_count, 1)
    relative_err[valid_entry_count == 0] = None
    return relative_err

def max_relative_error(y_pred: ndarray, y_true: ndarray, lb: float = 1e-4, epsilon: float = 1e-8) -> ndarray:
    mask = y_true > lb

    relative_err = np.abs(y_pred - y_true) / (np.abs(y_true) + epsilon)
    relative_err = relative_err * mask

    valid_entry_count = np.sum(mask, -1)
    relative_err = np.max(relative_err, -1)
    relative_err[valid_entry_count == 0] = None
    return relative_err

