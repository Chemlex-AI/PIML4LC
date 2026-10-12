from __future__ import annotations

import torch.nn as nn

from torch import Tensor
from omegaconf import DictConfig

__all__ = ['ChrPred']

class ChrPred(nn.Module):
    bul_only: bool
    n_time: int
    n_para: int
    n_z: int
    n_r: int
    n_c: int
    def __init__(
            self,
            bul_only: bool,
            var_dims: DictConfig[str, int],
            **kwargs
        ):
        super().__init__()
        self.bul_only = bul_only

        vd = var_dims
        self.n_time = vd.n_time
        self.n_para = vd.n_para
        self.n_exp_para = vd.n_exp_para
        self.n_oper_para = vd.n_oper_para

        self.n_z, self.n_r = vd.n_z, vd.n_r
        self.n_c = 1 if bul_only else 2 * self.n_r + 1 

    def forward(
            self, paras: Tensor, paras_t: Tensor
        ) -> tuple[Tensor, Tensor, Tensor] | Tensor:
        # paras.shape == (bsz, n_pde_para)
        # paras_t.shape == (bsz, n_time, 2)
        raise NotImplementedError()



