from __future__ import annotations

from torch import Tensor

import torch
import torch.nn.functional as F

from numpy import ndarray
from omegaconf import OmegaConf

from .base_module import PredModule

from src.utils import RankedLogger
from src.data.batch import PDEBatch
from ..net.rollout_net import RolloutNet

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ['RolloutPredModule']

class RolloutPredModule(PredModule):
    def __init__(
        self,
        rollout_net: RolloutNet,
        paras_cfg_path: str,
        **kwargs,
    ):
        cfg = OmegaConf.load(paras_cfg_path)
        loss_weights = OmegaConf.to_container(cfg.loss_weights, resolve=True)
        super().__init__(
            model=rollout_net,
            paras_cfg_path=paras_cfg_path,
            loss_weights=loss_weights,
            **kwargs,
        )
        self.loss_weights = self.hparams.loss_weights
        log.info(f"loss_weights={dict(self.loss_weights)}")

    def build_torchmetric(self):
        super().build_torchmetric(
            metrics=['loss', 'peak_pos'],
        )
        
    @torch.inference_mode()
    def forward(
            self,
            pde_batch: PDEBatch,
            paras_range,
        ) -> ndarray:
        device = next(self.model.parameters()).device
        paras, paras_t = pde_batch.solver_input(
            n_time=self.model.n_time,
            paras_range=paras_range,
            time_stride=self.model.time_downsample,
        )
        paras   = paras.float().to(device)
        paras_t = paras_t.float().to(device)

        pred = self.model(paras=paras, paras_t=paras_t)
        if isinstance(pred, tuple):
            pred = pred[0][:, :, -1]
        else:
            pred = pred[:, :, -1, 0]
        if self.model.time_downsample > 1:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=self.model.n_time_full,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        return torch.clip(pred, min=0.0).detach().cpu().numpy()

    def forward_tensor(
            self,
            pde_batch: PDEBatch,
            paras_range,
        ) -> Tensor:
        device = next(self.model.parameters()).device
        paras, paras_t = pde_batch.solver_input(
            n_time=self.model.n_time,
            paras_range=paras_range,
            time_stride=self.model.time_downsample,
        )
        paras = paras.float().to(device)
        paras_t = paras_t.float().to(device)
        pred = self.model(paras=paras, paras_t=paras_t)
        if isinstance(pred, tuple):
            pred = pred[0][:, :, -1]
        else:
            pred = pred[:, :, -1, 0]
        if self.model.time_downsample > 1:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=self.model.n_time_full,
                mode="linear",
                align_corners=True,
            ).squeeze(1)
        return torch.clip(pred, min=0.0)

    def step(
            self, batch: list[
                Tensor, Tensor | None, Tensor | None,
                Tensor, Tensor, PDEBatch
            ]
        ) -> tuple[Tensor, dict[str, float]]:
        bul, par, sol, paras, paras_t, _ = batch

        pred = self.model(paras=paras, paras_t=paras_t)
        log_out = {}
        if isinstance(pred, Tensor):
            pred_bul = pred.squeeze(-1)
            loss = self.mse(pred_bul, bul)
            bul_out_loss = loss

        else:
            pred_bul, pred_par, pred_sol = pred
            pred = torch.cat((pred), -1)

            gt = torch.cat((bul, par, sol), -1)

            assert pred.shape == gt.shape
            assert pred_bul.shape == bul.shape
            assert pred_par.shape == par.shape
            assert pred_sol.shape == sol.shape

            bul_loss = self.mse(pred_bul, bul)
            par_loss = self.mse(pred_par, par)
            sol_loss = self.mse(pred_sol, sol)

            w = self.loss_weights
            loss = (
                w['bul'] * bul_loss
                + w['par'] * par_loss
                + w['sol'] * sol_loss
            )
            bul_out_loss = self.mse(
                pred_bul[..., -1], bul[..., -1]
            )

            log_out['bul'] = bul_loss
            log_out['par'] = par_loss
            log_out['sol'] = sol_loss

        log_out['loss'] = loss
        log_out['bul_out'] = bul_out_loss
        log_out['peak_pos'] = (
            pred_bul[..., -1].argmax(1) - bul[..., -1].argmax(1)
        ).abs().float().mean()

        if torch.isnan(loss):
            log.info("Loss NAN on step ", self.global_step)
            loss = loss * 0

        return loss, log_out
