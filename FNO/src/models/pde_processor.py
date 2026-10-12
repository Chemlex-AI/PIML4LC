from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from omegaconf import DictConfig, OmegaConf

from src.data.batch import OPER_PARA_ORDER, PARA_ORDER

__all__ = ['PDEProcessor']

class PDEProcessor():
    def __init__(self, paras: DictConfig | str):
        if isinstance(paras, str):
            paras = OmegaConf.load(paras)['paras']
        self.paras = paras

    def norm_01(self, x, a, b):
        return (x - a) / (b - a)

    def unnorm_01(self, x, a, b):
        return x * (b - a) + a

    def log10(self, x):
        if isinstance(x, Tensor):
            return torch.log10(x)
        return np.log10(x)

    def dict_norm_01(self, paras: dict) -> dict:
        normed = {}
        for para_k, para_v in paras.items():
            para_info = self.paras[para_k]
            if para_info.log_scale:
                v = self.log10(para_v)
            else:
                v = para_v
            normed[para_k] = self.norm_01(
                v,
                para_info['min'], para_info['max']
            )
        return normed

    def dict_unnorm_01(self, paras: dict) -> dict:
        unnormed = {}
        for para_k, para_v in paras.items():
            para_info = self.paras[para_k]
            v = self.unnorm_01(
                para_v,
                para_info['min'], para_info['max']
            )
            if para_info.log_scale:
                if isinstance(v, Tensor):
                    v = torch.pow(10.0, v)
                else:
                    v = np.power(10.0, v)
            unnormed[para_k] = v
        return unnormed

    def physical_bounds(self, para_k: str) -> tuple[float, float]:
        para_info = self.paras[para_k]
        if para_info.log_scale:
            return float(10.0 ** para_info.min), float(10.0 ** para_info.max)
        return float(para_info.min), float(para_info.max)

    def assert_physical_pde_paras(self, paras: dict) -> None:
        for para_k, para_v in paras.items():
            para_info = self.paras[para_k]
            if not para_info.pde_para:
                continue
            lo, hi = self.physical_bounds(para_k)
            if isinstance(para_v, Tensor):
                vals = para_v.detach().cpu().reshape(-1)
            else:
                vals = np.asarray(para_v, dtype=np.float64).reshape(-1)
            for i, raw in enumerate(vals):
                v = float(raw)
                if not (lo <= v <= hi):
                    raise ValueError(
                        f'{para_k}[{i}]={v} outside physical range [{lo}, {hi}]'
                    )
                if para_k == 'qmax' and v < 1.0:
                    raise ValueError(
                        f'{para_k}[{i}]={v} looks normalized; expected physical qmax>=10'
                    )
                if para_k in ('col_disp', 'par_diff') and v > 1e-6:
                    raise ValueError(
                        f'{para_k}[{i}]={v} looks normalized; expected <=1e-7'
                    )
                if para_k == 'fil_diff' and v > 0.15:
                    raise ValueError(
                        f'{para_k}[{i}]={v} looks normalized; expected <=0.1'
                    )

    @property
    def oper_para_order(self):
        return list(OPER_PARA_ORDER)

    @property
    def initializer_para_order(self):
        return [k for k in self.para_order if k not in self.oper_para_order]

    @property
    def para_order(self):
        return list(PARA_ORDER)
