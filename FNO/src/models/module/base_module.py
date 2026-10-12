from __future__ import annotations

from torch import Tensor

from typing import Optional

import torch
import torch.nn as nn

from torchmetrics import MeanMetric

from lightning.pytorch.utilities import grad_norm
from lightning import LightningModule, seed_everything

from src.utils import RankedLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

log = RankedLogger(__name__, rank_zero_only=True)

__all__ = ['PredModule']

class PredModule(LightningModule):
    metrics: list[str]

    def __init__(
        self,
        model: nn.Module,
        optimizer, scheduler,
        load_pretrained_ckpt: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=['model', 'rollout_net'], logger=False
        )
    
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self._load_pretrained_ckpt = load_pretrained_ckpt

    @torch.inference_mode()
    def forward(
            self,
            paras: Tensor,
            bul_out_only: bool = True,
            **kwargs
        ):
        raise NotImplementedError()

    def _load_pretrained_weights(self):
        ckpt_path = self._load_pretrained_ckpt
        if ckpt_path is None:
            return
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt

        model_state = {}
        prefix = "model."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                model_state[k[len(prefix):]] = v

        missing, unexpected = self.model.load_state_dict(
            model_state, strict=False
        )
        log.info(
            f"Loaded pretrained weights from {ckpt_path}, "
            f"missing_keys={len(missing)}, unexpected_keys={len(unexpected)}"
        )

    def setup(self, stage=None) -> None:
        self.stage = stage
        self._load_pretrained_weights()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_torchmetric(
            self,
            metrics: list[str],
        ):
        self.mse = nn.MSELoss()
        self.metrics = metrics
        for metric in metrics:
            setattr(self, f'val_{metric}', MeanMetric())
            setattr(self, f'test_{metric}', MeanMetric())





    def step(
            self, batch: list[Tensor]
        ) -> tuple[Tensor, dict[str, float]]:
        raise NotImplementedError()
    




    def training_step(self, batch, batch_idx: int):
        loss, logging_output = self.step(batch)

        self.log('global_step', self.global_step, on_step=True, on_epoch=False, prog_bar=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, on_epoch=False, prog_bar=True)
        for k, v in logging_output.items():
            self.log(f'train/{k}', v, on_step=True, on_epoch=False, prog_bar=True)

        return logging_output

    def on_before_optimizer_step(self, optimizer):
        if self.global_rank == 0:
            gn = grad_norm(self.trainer.model, norm_type=2)
            gn = max([v.item() for v in gn.values()]) if len(gn.values()) > 0 else 0
            self.log('train/grad_norm', gn, on_step=True, on_epoch=False, prog_bar=True)
                
    
    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.trainer.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            log.info(f'warning: detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()

    def on_train_epoch_end(self) -> None:
        pass






    def on_validation_start(self):
        seed_everything(42 + self.global_rank, workers=True)

    def validation_step(self, batch, batch_idx: int):
        loss, logging_output = self.step(batch)
        for metric in self.metrics:
            getattr(self, f'val_{metric}').update(logging_output[metric])

    def on_validation_epoch_end(self):
        log_info = {}
        for metric in self.metrics:
            eval_val = getattr(self, f'val_{metric}').compute()
            getattr(self, f'val_{metric}').reset()
            log_info[f"val/{metric}"] = eval_val
        self.log_validation_info(log_info)


    def log_validation_info(
        self,
        logging_info: dict[str, Tensor | float]
    ) -> None:
        for k, v in logging_info.items():
            self.log(
                f"{k}", v, on_step=False, on_epoch=True, prog_bar=True
            )
        dict_info = ", ".join(
            f"{key}={val:.3f}" for key, val in logging_info.items()
        )
        log.info(
            f"Val Info @ (Epoch {self.current_epoch}, step {self.global_step}: {dict_info}"
            )

    
    def on_test_start(self):
        seed_everything(42 + self.global_rank, workers=True)

    def test_step(self, batch, batch_idx: int):
        loss, logging_output = self.step(batch)
        for test in self.metrics:
            getattr(self, f'test_{test}').update(logging_output[test])


    def on_test_epoch_end(self):
        log_info = {}
        for test in self.metrics:
            eval_test = getattr(self, f'test_{test}').compute()
            getattr(self, f'test_{test}').reset()
            log_info[f"test/{test}"] = eval_test

        self.log_dict(log_info, on_step=False, on_epoch=True)

        log.info(f"Test Info @ (Epoch {self.current_epoch}, global step {self.global_step}: {log_info}")


    
    def configure_optimizers(self):
        optimizer = self.optimizer(
            params=self.trainer.model.parameters()
        )
        sched = self.scheduler(optimizer)
        if isinstance(sched, ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": sched,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise NotImplementedError()


