from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from numpy import ndarray
from scipy.spatial import cKDTree
from torch import Tensor
from src.data.batch import PDEBatch
from src.models.pde_processor import PDEProcessor


from abc import ABC, abstractmethod

__all__ = ['InverseAlg']


class InverseAlg(ABC):
    def __init__(
            self, solver: nn.Module, paras_cfg
        ) -> None:
        super().__init__()
        self.solver = solver
        self.paras_cfg = paras_cfg
        self.pde_processor = PDEProcessor(paras_cfg)
        self.pred_chunk_size = 128

    def rand_init(self, bsz: int) -> PDEBatch:
        paras = {
            k: torch.rand((bsz, ))
            for k in self.pde_processor.para_order
        }
        unnormed = self.pde_processor.dict_unnorm_01(paras)
        return PDEBatch(**unnormed)

    def pin_oper_params(
            self,
            pde_batch: PDEBatch,
            end_con: ndarray,
            end_time: ndarray,
        ) -> PDEBatch:
        return pde_batch.new_exp_paras(
            end_con=float(np.asarray(end_con).reshape(-1)[0]),
            end_time=float(np.asarray(end_time).reshape(-1)[0]),
        )

    def physical_indices(self) -> list[int]:
        order = self.pde_processor.para_order
        return [
            order.index(k)
            for k in self.pde_processor.initializer_para_order
        ]

    def physical_values(self, paras: torch.Tensor) -> torch.Tensor:
        idx = torch.as_tensor(
            self.physical_indices(),
            dtype=torch.long,
            device=paras.device,
        )
        return paras.index_select(-1, idx)

    def compute_batch(
            self,
            pde_batch: PDEBatch,
            *,
            end_con: ndarray, end_time: ndarray
        ) -> list[PDEBatch]:
        n_exper = len(end_con)
        assert n_exper == len(end_time)
        return [
            pde_batch.new_exp_paras(
                end_con=end_con[i_exper], 
                end_time=end_time[i_exper],
            )
            for i_exper in range(n_exper)
        ]


    def model_pred(
            self,
            pde_batch: PDEBatch | list[PDEBatch],
            *,
            end_con: ndarray,
            end_time: ndarray
        ) -> ndarray:
        n_exper = len(end_con)
        assert n_exper == len(end_time)

        if isinstance(pde_batch, PDEBatch):
            exper_bathes = self.compute_batch(
                pde_batch=pde_batch, 
                end_con=end_con, end_time=end_time
            )
        elif isinstance(pde_batch, list):
            exper_bathes = pde_batch
        else:
            raise ValueError('pde_batch should be PDEBatch or list of PDEBatch.')

        return np.stack([
            self._solver_chunked(exp_batch)
            for exp_batch in exper_bathes
        ], axis=-1)

    def model_pred_torch(
            self,
            pde_batch: PDEBatch | list[PDEBatch],
            *,
            end_con: ndarray,
            end_time: ndarray,
        ) -> Tensor:
        n_exper = len(end_con)
        assert n_exper == len(end_time)
        if isinstance(pde_batch, PDEBatch):
            exper_bathes = self.compute_batch(
                pde_batch=pde_batch,
                end_con=end_con, end_time=end_time
            )
        elif isinstance(pde_batch, list):
            exper_bathes = pde_batch
        else:
            raise ValueError('pde_batch should be PDEBatch or list of PDEBatch.')
        preds = [
            self._solver_chunked_torch(exp_batch)
            for exp_batch in exper_bathes
        ]
        return torch.stack(preds, dim=-1)

    def _solver_chunked_torch(self, exp_batch: PDEBatch) -> Tensor:
        n = len(exp_batch)
        chunk = int(self.pred_chunk_size)
        if chunk <= 0 or n <= chunk:
            return self.solver.forward_tensor(exp_batch, paras_range=self.paras_cfg)
        outs = []
        for i in range(0, n, chunk):
            sub = self._slice_pde_batch(exp_batch, slice(i, min(i + chunk, n)))
            outs.append(self.solver.forward_tensor(sub, paras_range=self.paras_cfg))
        return torch.cat(outs, dim=0)

    def _slice_pde_batch(self, pde_batch: PDEBatch, sl) -> PDEBatch:
        kwargs = pde_batch.solver_kwargs()
        return PDEBatch(**{k: v[sl] for k, v in kwargs.items()})

    def _solver_chunked(self, exp_batch: PDEBatch) -> ndarray:
        n = len(exp_batch)
        chunk = int(self.pred_chunk_size)
        if chunk <= 0 or n <= chunk:
            return self.solver(exp_batch, paras_range=self.paras_cfg)
        outs = []
        for i in range(0, n, chunk):
            sub = self._slice_pde_batch(exp_batch, slice(i, min(i + chunk, n)))
            outs.append(self.solver(sub, paras_range=self.paras_cfg))
        return np.concatenate(outs, axis=0)
    



    def unique_indices(
            self, arr: ndarray, eps: float
        ) -> ndarray:
        if len(arr) == 0:
            return np.array([], dtype=int)
        arr = np.asarray(arr)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        tree = cKDTree(arr)
        visited = np.zeros(len(arr), dtype=bool)
        unique_idxs = []

        for i in range(len(arr)):
            if visited[i]:
                continue
            unique_idxs.append(i)
            idxs = tree.query_ball_point(arr[i], eps)
            visited[idxs] = True
        return np.array(unique_idxs, dtype=int)

    



    @abstractmethod
    def optimize(
            self,
            oper_para: dict[str, ndarray | float | int],
            gt_cl_t: ndarray
        ):
        pass
