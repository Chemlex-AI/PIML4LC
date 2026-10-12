from __future__ import annotations

import pickle

from dataclasses import dataclass, fields


import torch
import numpy as np

from torch import Tensor
from omegaconf import DictConfig
from omegaconf import OmegaConf

class BaseBatchClass:
    def __getitem__(self, key):
        return getattr(self, key)
    
    @property
    def keys(self) -> list[str]:
        return [field.name for field in fields(self)]
    
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            key: getattr(self, key).shape
            for key in self.keys
        }
    
    def get_shape(self, key: str) -> tuple[int, ...]:
        if hasattr(self, key):
            return getattr(self, key).shape
        else:
            raise KeyError(f"Key '{key}' not found in BatchDataClass")
    
    def summary(self) -> None:
        print("BatchDataClass Summary:")
        shapes = self.shapes()
        for key, shape in shapes.items():
            print(f"{key}: {shape}")
    

PARA_ORDER = [
    'end_con', 'end_time',
    'sa', 'sd', 'sq', 'ka', 'kd', 'qmax',
    'col_disp', 'fil_diff', 'par_diff',
]

OPER_PARA_ORDER = ['end_con', 'end_time']


@dataclass
class PDEBatch(BaseBatchClass):
    end_con: Tensor
    end_time: Tensor
    sa: Tensor
    sd: Tensor
    sq: Tensor
    ka: Tensor
    kd: Tensor
    qmax: Tensor
    col_disp: Tensor
    fil_diff: Tensor
    par_diff: Tensor

    def __post_init__(self):
        self._refresh_v()

    def _para_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.end_con, self.end_time,
            self.sa, self.sd, self.sq,
            self.ka, self.kd, self.qmax,
            self.col_disp, self.fil_diff, self.par_diff,
        )

    def _refresh_v(self):
        self.v = torch.stack(self._para_tensors(), -1)

    def __getitem__(self, idx):
        assert self.v.ndim == 2
        return self.v[idx]
    
    def __len__(self) -> int:
        return self.bsz
    
    def solver_input(
            self,
            n_time: int,
            paras_range: DictConfig,
            time_stride: int,
        ) -> tuple[Tensor, Tensor]:
        pde_processor = PDEProcessor(paras_range)
        pde_dict = self.solver_kwargs()
        para_normed = pde_processor.dict_norm_01(pde_dict)
        log_uni_normed = pde_processor.dict_log_uni_norm_01(pde_dict)

        unbatched = self.end_time.ndim == 0
        normalized = 2 * torch.stack(
            list(para_normed.values()) + list(log_uni_normed.values()),
            dim=-1
        ) - 1
        if unbatched:
            normalized = normalized.unsqueeze(0)
        assert normalized.min() > -1.1
        assert normalized.max() < 1.1
        
        end_time = self.end_time.reshape(-1, 1)
        end_con = self.end_con.reshape(-1, 1)
        sa = self.sa.reshape(-1, 1)
        sd = self.sd.reshape(-1, 1)
        ka = self.ka.reshape(-1, 1)
        kd = self.kd.reshape(-1, 1)

        dev = end_time.device
        dt = end_time.dtype
        tt = (
            torch.arange(n_time + 1, device=dev, dtype=dt) * time_stride
        ).view(1, -1)
        
        ext_pro = torch.where(
            tt < end_time,
            0.05 + (end_con - 0.05) / end_time * tt,
            end_con
        )
        
        adsorp = torch.exp(-sa * ext_pro) * ka
        desorp = torch.exp(-sd * ext_pro) * kd
        
        paras_t = 0.1 * torch.stack([
            adsorp[:, 1:], desorp[:, 1:]
        ], dim=-1)

        if unbatched:
            normalized = normalized.squeeze(0)
            paras_t = paras_t.squeeze(0)

        return normalized, paras_t
        

    def initializer_input(
            self, paras_range: DictConfig
        ) -> Tensor:
        pde_processor = PDEProcessor(paras_range)
        solver_kwargs = self.solver_kwargs()
        for k in pde_processor.oper_para_order:
            del solver_kwargs[k]
        para_normed = pde_processor.dict_norm_01(solver_kwargs)

        normalized = 2 * torch.stack(
            list(para_normed.values()), dim=-1
        ) - 1
        assert normalized.min() > -1.1
        assert normalized.max() < 1.1

        return normalized

    def solver_kwargs(self, idx: int = None) -> dict[str, float | int]:
        if idx is not None:
            kwargs = {
                key: float(getattr(self, key)[idx])
                for key in PARA_ORDER
            }
            kwargs['end_time'] = int(kwargs['end_time'])
            return kwargs
        else:
            return {
                key: getattr(self, key)
                for key in PARA_ORDER
            }
    
    
    def save(self, path: str, idx: int = None):
        assert path.endswith('.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self.solver_kwargs(idx=idx), f)


    @property
    def bsz(self) -> int:
        if self.v.ndim == 1:
            return 1
        return self.v.size(0)

    @property
    def values(self) -> Tensor:
        return self.v
    
    def update(self, v: Tensor):
        self.end_con, self.end_time, \
        self.sa, self.sd, self.sq, \
        self.ka, self.kd, self.qmax, \
        self.col_disp, self.fil_diff, self.par_diff \
            = v.T
        self._refresh_v()


    def set_exp_paras(
            self,
            end_con: Tensor, end_time: Tensor
        ) -> None:
        self.end_con = end_con
        self.end_time = end_time
        self._refresh_v()

    def new_exp_paras(
            self,
            end_con: Tensor, end_time: Tensor
        ) -> 'PDEBatch':
        if not torch.is_tensor(end_con):
            end_con = torch.as_tensor(end_con)
        if not torch.is_tensor(end_time):
            end_time = torch.as_tensor(end_time)
        if end_con.ndim == end_time.ndim == 0:
            bsz = self.sa.shape[0]
            end_con, end_time = map(
                lambda x: torch.full((bsz,), x, dtype=self.sa.dtype, device=self.sa.device),
                (end_con, end_time)
            )
        assert end_con.shape == end_time.shape == self.sa.shape
        return PDEBatch(
            end_con=end_con,
            end_time=end_time,
            sa=self.sa,
            sd=self.sd,
            sq=self.sq,
            ka=self.ka,
            kd=self.kd,
            qmax=self.qmax,
            col_disp=self.col_disp,
            fil_diff=self.fil_diff,
            par_diff=self.par_diff,
        )

    @staticmethod
    def from_list(batch_list: list) -> 'PDEBatch':
        end_con, end_time = [], []
        sa, sd, sq, ka, kd, qmax = [], [], [], [], [], []
        col_disp, fil_diff, par_diff = [], [], []
        for batch in batch_list:
            end_con.append(batch.end_con)
            end_time.append(batch.end_time)
            sa.append(batch.sa)
            sd.append(batch.sd)
            sq.append(batch.sq)
            ka.append(batch.ka)
            kd.append(batch.kd)
            qmax.append(batch.qmax)
            col_disp.append(batch.col_disp)
            fil_diff.append(batch.fil_diff)
            par_diff.append(batch.par_diff)
        return PDEBatch(
            end_con=torch.stack(end_con),
            end_time=torch.stack(end_time),
            sa=torch.stack(sa),
            sd=torch.stack(sd),
            sq=torch.stack(sq),
            ka=torch.stack(ka),
            kd=torch.stack(kd),
            qmax=torch.stack(qmax),
            col_disp=torch.stack(col_disp),
            fil_diff=torch.stack(fil_diff),
            par_diff=torch.stack(par_diff),
        )

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
        else:
            return np.log10(x)

    def clip01(self, x):
        if isinstance(x, Tensor):
            return torch.clip(x, min=0.0, max=1.0)
        else:
            return np.clip(x, a_min=0.0, a_max=1.0)

    def dict_norm_01(self, paras: dict) -> dict:
        normed = {}
        for para_k, para_v in paras.items():
            para_info = self.paras[para_k]
            if para_info.log_scale:
                v = self.log10(para_v)
            else:
                v = para_v
            normed[para_k] = self.clip01(self.norm_01(
                v,
                para_info['min'], para_info['max']
            ))
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
    
    def dict_log_uni_norm_01(self, paras: dict) -> dict:
        normed = {}
        for para_k, para_v in paras.items():
            para_info = self.paras[para_k]
            if not para_info.log_scale:
                continue
            normed[para_k] = self.clip01(self.norm_01(
                para_v,
                10 ** para_info['min'], 10 ** para_info['max']
            ))
        return normed

    @property
    def oper_para_order(self):
        return list(OPER_PARA_ORDER)

    @property
    def initializer_para_order(self):
        return [k for k in self.para_order if k not in self.oper_para_order]

    @property
    def para_order(self):
        return list(PARA_ORDER)
