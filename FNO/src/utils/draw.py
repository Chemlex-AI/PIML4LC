"""Small plotting helpers for the optional GA progress display."""
import numpy as np
import matplotlib.pyplot as plt


def draw_loss(loss_track, ax=None, draw_all=False):
    ax = ax if ax is not None else plt.gca()
    values = np.asarray(loss_track)
    ax.plot(values if draw_all else values.mean(axis=1))
    ax.set(xlabel='Generation', ylabel='Loss', yscale='log')


def draw_cl(model_pred, gt_cl, ax=None, gt_bul=None):
    ax = ax if ax is not None else plt.gca()
    n = min(len(model_pred), len(gt_cl))
    for index in range(model_pred.shape[1]):
        line, = ax.plot(gt_cl[:n, index], label=f'target {index + 1}')
        ax.plot(model_pred[:n, index], '--', color=line.get_color(), label=f'fit {index + 1}')
    ax.legend()
