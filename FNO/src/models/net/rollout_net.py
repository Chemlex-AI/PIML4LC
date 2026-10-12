from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor
from functools import partial

from .base import ChrPred
from .utils import MPFourierEmbedding, MLP

__all__ = ['RolloutNet']

class RolloutNet(ChrPred):
    def __init__(
            self,
            net: partial[nn.Module],
            cond_embed_size: int, cond_n_layers: int,
            time_downsample: int,
            relu: bool = True,
            pred_residual: bool = False,
            time_emb: bool = True,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.time_downsample = time_downsample
        self.n_time_full = self.n_time
        self.n_time = (
            self.n_time_full + time_downsample - 1
        ) // time_downsample

        n_para = self.n_para + self.n_exp_para
        self.init_bul = nn.Linear(n_para, self.n_z)

        self.net = net(
            n_time=self.n_time,
            n_z=self.n_z,
            n_c=self.n_c,
            cond_embed_size=cond_embed_size,
            **kwargs
        )

        self.pred_residual = pred_residual

        self.t_emb = nn.Sequential(
            MPFourierEmbedding(cond_embed_size),
            nn.Linear(cond_embed_size, cond_embed_size)
        ) if time_emb else None

        self.para_emb  = MLP(
            [n_para] + [cond_embed_size] * cond_n_layers
        )
        self.para_emb_t  = MLP(
            [2] + [cond_embed_size] * cond_n_layers
        )

        self.relu_mid = nn.ReLU() if relu else nn.Identity()
        self.relu_out = nn.LeakyReLU(0.1) if relu else nn.Identity()

    def forward(
            self, paras: Tensor, paras_t: Tensor
        ) -> tuple[Tensor, Tensor, Tensor] | Tensor:
        # paras.shape == (bsz, n_pde_para)
        # paras_t.shape == (bsz, n_time, 2)
        bsz = paras.size(0)

        pde_v0 = torch.cat((
            self.init_bul(paras).unsqueeze(-1),
            torch.zeros(
                size=(bsz, self.n_z, self.n_c-1),
                device=paras.device
            ),
        ), -1)
        preds = [pde_v0]
        paras_emb: Tensor = self.para_emb(paras)
        n_t = min(self.n_time, paras_t.size(1))

        t_emb: Tensor = self.para_emb_t(paras_t)
        if self.t_emb:
            t_emb: Tensor = t_emb + self.t_emb(
                torch.arange(n_t, device=paras.device)
            ).unsqueeze(0).repeat(bsz, 1, 1)

        for i in range(n_t):
            # paras_emb.shape == (bsz, cond_embed_size)
            t_in_embed = paras_emb + t_emb[:, i]
            model_pred: Tensor = self.net(
                y=self.relu_mid(preds[-1]), paras=t_in_embed
            )
            if self.pred_residual:
                model_pred = model_pred + preds[-1]
            preds.append(model_pred)
        preds: Tensor = self.relu_out(torch.stack(preds, dim=1)[:, 1:])
        
        if self.bul_only:
            return preds
        else:
            par_dim = (preds.size(-1) - 1) // 2
            bul, par, sol = torch.split(
                preds, [1, par_dim, par_dim], dim=-1
            )
            bul = bul.squeeze(-1)
            par = par.flatten(2).view(*par.shape[:2], -1)
            sol = sol.flatten(2).view(*sol.shape[:2], -1)
            return bul, par, sol



