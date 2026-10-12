from __future__ import annotations

from numpy import ndarray
from typing import Union

from .comsol import ComsolSolver

__all__ = [
    'gradient_solve',
    'isocratic_solve',
]

def gradient_solve(
        *, 
        end_con: float, end_time: int,
        sa: float, sd: float, sq: float,
        ka: float, kd: float, qmax: float,
        col_disp: float, fil_diff: float, par_diff: float,
        n_time: int = 360,
        clip: bool = False, bul_out_only: bool = True,
        lock = None, n_cores = None,
        **kwargs
    ) -> Union[dict[str, ndarray], ndarray, None]:
    return ComsolSolver()(
        start_con=0.05, end_con=end_con, end_time=end_time,
        sa=sa, sd=sd, sq=sq, ka=ka, kd=kd, qmax=qmax,
        col_disp=col_disp, fil_diff=fil_diff, par_diff=par_diff,
        n_time=n_time,
        clip=clip, bul_out_only=bul_out_only,
        lock=lock, n_cores=n_cores
    )

def isocratic_solve(
        *, 
        end_p: float,
        sa: float, sd: float, sq: float,
        ka: float, kd: float, qmax: float,
        col_disp: float, fil_diff: float, par_diff: float,
        n_time: int = 360,
        clip: bool = False, bul_out_only: bool = True,
        lock = None, n_cores = None,
        **kwargs
    ) -> Union[dict[str, ndarray], ndarray, None]:
    return ComsolSolver()(
        start_con=end_p, end_con=end_p, end_time=n_time,
        sa=sa, sd=sd, sq=sq, ka=ka, kd=kd, qmax=qmax,
        col_disp=col_disp, fil_diff=fil_diff, par_diff=par_diff,
        n_time=n_time,
        clip=clip, bul_out_only=bul_out_only,
        lock=lock, n_cores=n_cores
    )

