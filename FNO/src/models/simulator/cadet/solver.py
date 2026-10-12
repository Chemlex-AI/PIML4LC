from __future__ import annotations

import numpy as np

from numpy import ndarray
from typing import Union

from .cadet import CadetGradEluSolver, CadetIsocEluSolver

__all__ = [
    'gradient_solve',
    'isocratic_solve',
]


def _to_array(v):
    return np.array([v]) if np.isscalar(v) else np.asarray(v)


def output_reshape(
        result: Union[dict[str, ndarray], ndarray, None]
    ) -> Union[dict[str, ndarray], ndarray, None]:
    if result is None:
        return None
    elif isinstance(result, dict):
        bul = result['bul'][..., -1] # squeeze the component dimension
        par, sol = map(
            lambda x: x[..., -1].reshape(bul.shape[0], -1),
            (result['par'], result['sol'])
        )
        return {'bul': bul, 'par': par, 'sol': sol}
    elif isinstance(result, ndarray):
        return result[..., -1]
    else:
        raise ValueError()


def gradient_solve(
        *,
        end_con: float, end_time: int,
        sa: float, sd: float, sq: float,
        ka: float, kd: float, qmax: float,
        col_disp: float, fil_diff: float, par_diff: float,
        n_time: int = 360,
        clip: bool = False, bul_out_only: bool = True,
        **kwargs
    ) -> Union[dict[str, ndarray], ndarray, None]:
    result = CadetGradEluSolver()(
        end_con=end_con, end_time=end_time,
        sa=_to_array(sa), sd=_to_array(sd), sq=_to_array(sq),
        ka=_to_array(ka), kd=_to_array(kd), qmax=_to_array(qmax),
        col_disp=_to_array(col_disp), fil_diff=_to_array(fil_diff), par_diff=_to_array(par_diff),
        n_time=n_time,
        clip=clip, bul_out_only=bul_out_only,
    )
    return output_reshape(result)


def isocratic_solve(
        *,
        end_p: float,
        sa: float, sd: float, sq: float,
        ka: float, kd: float, qmax: float,
        col_disp: float, fil_diff: float, par_diff: float,
        n_time: int = 360,
        clip: bool = False, bul_out_only: bool = True,
        **kwargs
    ) -> Union[dict[str, ndarray], ndarray, None]:
    result = CadetIsocEluSolver()(
        end_p=_to_array(end_p),
        sa=_to_array(sa), sd=_to_array(sd), sq=_to_array(sq),
        ka=_to_array(ka), kd=_to_array(kd), qmax=_to_array(qmax),
        col_disp=_to_array(col_disp), fil_diff=_to_array(fil_diff), par_diff=_to_array(par_diff),
        n_time=n_time,
        clip=clip, bul_out_only=bul_out_only,
    )
    return output_reshape(result)
