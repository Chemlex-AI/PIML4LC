from __future__ import annotations

import gc
import os
import mph
import tempfile

import numpy as np
import pandas as pd

from numpy import ndarray
from typing import Union

__all__ = ['ComsolSolver', 'reset_client']

_client = None

def reset_client():
    global _client
    if _client is not None:
        try:
            if hasattr(_client, 'disconnect'):
                _client.disconnect()
        except Exception:
            pass
        _client = None
    gc.collect()

def _get_or_create_client(cores: int):
    global _client
    
    if _client is not None:
        try:
            _ = _client.names()
            return _client
        except Exception:
            _client = None
    
    _client = mph.start(cores=cores)
    return _client


class ComsolSolver:
    def __init__(self, model_path: str | None = None):
        from pathlib import Path
        default = Path(__file__).resolve().parents[5] / 'COMSOL_models' / 'GRM.mph'
        self.model_path = str(model_path or os.environ.get('PIML4LC_COMSOL_MODEL', default))
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f'COMSOL model not found: {self.model_path}. Set PIML4LC_COMSOL_MODEL.')

    def __call__(
            self, *, 
            start_con: float, end_con: float,
            end_time: int,
            sa: float, sd: float, sq: float,
            ka: float, kd: float, qmax: float,
            col_disp: float, fil_diff: float, par_diff: float,
            n_time: int,
            clip: bool, bul_out_only: bool,
            lock = None, n_cores = None
        ) -> Union[dict[str, ndarray], ndarray, None]:

        if lock:
            lock.acquire()
        try:
            client = _get_or_create_client(cores=n_cores)
        finally:
            if lock:
                lock.release()
        
        model = None
        result = None
        try:
            model = client.load(self.model_path)
            
            model.parameter('phi0', f'{start_con}')
            model.parameter('phie', f'{end_con}')
            model.parameter('ts', f'{end_time} [s]')
            model.parameter('te', f'{n_time} [s]')
            model.parameter('D_1', f'{col_disp} [m^2/s]')
            model.parameter('Dp', f'{par_diff} [m^2/s]')
            model.parameter('kf', f'{fil_diff} [m/s]')
            model.parameter('ka', f'{ka}')
            model.parameter('kd', f'{kd}')
            model.parameter('qmax', f'{qmax}')
            model.parameter('sads', f'{sa}')
            model.parameter('sdep', f'{sd}')
            model.parameter('sq', f'{sq}')
            model.parameter('c01', '1.0')
            
            model.build()
            model.mesh()
            model.solve('Study 1')
            
            with tempfile.TemporaryDirectory() as tmpdir:
                cl_path = os.path.join(tmpdir, 'cl.csv')
                cp_path = os.path.join(tmpdir, 'cp.csv')
                q_path = os.path.join(tmpdir, 'q.csv')
                
                model.export('数据 1', cl_path)
                model.export('数据 2', cp_path)
                model.export('数据 3', q_path)
                
                bul = pd.read_csv(cl_path).to_numpy()[1:].astype(np.float32)
                par = pd.read_csv(cp_path).to_numpy()[2:].astype(np.float32)
                sol = pd.read_csv(q_path).to_numpy()[2:].astype(np.float32)
            
            result = {'bul': bul, 'par': par, 'sol': sol}
            
        except Exception as e:
            raise RuntimeError(f'COMSOL solve or named CSV export failed: {e}') from e
            
        finally:
            if model is not None:
                try:
                    model.clear()
                except Exception:
                    pass
                try:
                    client.remove(model)
                except Exception:
                    pass
            gc.collect()

        if result:
            if clip:
                result = {
                    k: np.clip(v, a_min=0.0, a_max=None)
                    for k, v in result.items()
                }
            return result['bul'][:, -1] if bul_out_only else result
        return None 
