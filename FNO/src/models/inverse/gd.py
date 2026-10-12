from __future__ import annotations

import torch
import torch.nn as nn

from numpy import ndarray
from torch import Tensor

from src.data.batch import PDEBatch
from .inverse_alg import InverseAlg

__all__ = ['AdamRefine', 'LBFGSInverse']

PHYSICAL_KEYS = [
    'sa', 'sd', 'sq', 'ka', 'kd', 'qmax', 'col_disp', 'fil_diff', 'par_diff'
]


class AdamRefine(InverseAlg):
    def __init__(
            self,
            solver: nn.Module,
            paras_cfg,
            *,
            loss_weights: dict,
            zero_first_n: int | None,
            num_iters: int,
            lr: float,
        ) -> None:
        super().__init__(solver=solver, paras_cfg=paras_cfg)
        required_keys = {'mse', 'penalty_on_0', 'peak_pos', 'peak_v', 'tv_penalty'}
        if set(loss_weights.keys()) != required_keys:
            raise ValueError(
                f'loss_weights must contain exactly keys: {sorted(required_keys)}'
            )
        self.loss_weights = {k: float(v) for k, v in loss_weights.items()}
        if zero_first_n is not None and (not isinstance(zero_first_n, int) or zero_first_n < 0):
            raise ValueError()
        self.zero_first_n = zero_first_n
        self.num_iters = int(num_iters)
        self.lr = float(lr)
        for p in self.solver.parameters():
            p.requires_grad_(False)
        self.solver.eval()

    def optimize(self, *args, **kwargs):
        raise NotImplementedError()

    def _logit(self, u: Tensor, eps: float = 1e-4) -> Tensor:
        u = u.clamp(eps, 1.0 - eps)
        return torch.log(u) - torch.log(1.0 - u)

    def _norm_physical(self, all_paras: Tensor) -> Tensor:
        order = self.pde_processor.para_order
        phys = {
            k: all_paras[:, i]
            for i, k in enumerate(order)
            if k in PHYSICAL_KEYS
        }
        normed = self.pde_processor.dict_norm_01(phys)
        return torch.stack([normed[k] for k in PHYSICAL_KEYS], dim=-1)

    def _unnorm_u(self, u: Tensor) -> dict[str, Tensor]:
        packed = {k: u[:, i] for i, k in enumerate(PHYSICAL_KEYS)}
        return self.pde_processor.dict_unnorm_01(packed)

    def _batch_from_phys(
            self,
            phys: dict[str, Tensor],
            end_con0: float,
            end_time0: float,
        ) -> PDEBatch:
        bsz = phys['sa'].shape[0]
        device = phys['sa'].device
        dtype = phys['sa'].dtype
        kwargs = {
            'end_con': torch.full((bsz,), float(end_con0), device=device, dtype=dtype),
            'end_time': torch.full((bsz,), float(end_time0), device=device, dtype=dtype),
        }
        kwargs.update(phys)
        return PDEBatch(**kwargs)

    def tv_loss(self, x: Tensor, dim: int) -> Tensor:
        return torch.abs(torch.diff(x, dim=dim)).sum(dim=dim)

    def fit_loss_diff(self, model_pred: Tensor, gt_value: Tensor) -> Tensor:
        if self.zero_first_n is not None:
            head = model_pred[:, :self.zero_first_n]
            model_pred = torch.cat(
                [head * 0.0, model_pred[:, self.zero_first_n:]],
                dim=1,
            )
        n_time = min(model_pred.shape[1], gt_value.shape[0])
        model_pred = model_pred[:, :n_time, :]
        gt_value = gt_value[:n_time, :]
        normalize_cl = gt_value.max(0).values + 1e-5
        mse = torch.square(
            (model_pred - gt_value.unsqueeze(0)) / normalize_cl.reshape(1, 1, -1)
        ).reshape(model_pred.shape[0], -1).mean(-1)
        t = torch.arange(n_time, device=model_pred.device, dtype=model_pred.dtype)
        w_pred = model_pred.clamp_min(0.0) + 1e-8
        t_pred = (w_pred * t.view(1, -1, 1)).sum(1) / w_pred.sum(1)
        w_gt = gt_value.clamp_min(0.0) + 1e-8
        t_gt = (w_gt * t.view(-1, 1)).sum(0) / w_gt.sum(0)
        peak_pos = torch.square(t_pred - t_gt.reshape(1, -1)).mean(-1) / n_time
        peak_v = torch.square(
            (torch.max(model_pred, 1).values - torch.max(gt_value, 0).values.reshape(1, -1))
            / normalize_cl.reshape(1, -1)
        ).mean(-1)
        gt_tv = self.tv_loss(gt_value, dim=0) + 1e-8
        model_tv = self.tv_loss(model_pred, dim=1)
        tv_threshold = 3.0 * gt_tv.unsqueeze(0)
        tv_penalty = 1e8 * torch.relu(model_tv - tv_threshold).mean(-1)
        return (
            self.loss_weights['mse'] * mse
            + self.loss_weights['peak_pos'] * peak_pos
            + self.loss_weights['peak_v'] * peak_v
            + self.loss_weights['tv_penalty'] * tv_penalty
        )

    def refine(
            self,
            all_paras: Tensor,
            end_con: ndarray,
            end_time: ndarray,
            gt_cl_t: ndarray,
        ) -> tuple[Tensor, Tensor, list[float]]:
        if self.num_iters <= 0:
            raise ValueError('num_iters must be positive')
        device = next(self.solver.parameters()).device
        dtype = torch.float32
        all_paras = all_paras.detach().to(device=device, dtype=dtype)
        u0 = self._norm_physical(all_paras).clamp(1e-4, 1.0 - 1e-4)
        raw = nn.Parameter(self._logit(u0))
        opt = torch.optim.Adam([raw], lr=self.lr)
        gt = torch.as_tensor(gt_cl_t, dtype=dtype, device=device)
        n = int(u0.shape[0])
        mb = max(1, int(self.pred_chunk_size))
        best_u = u0.detach().clone()
        best_loss = torch.full((n,), float('inf'), device=device, dtype=dtype)
        history = []
        for _ in range(self.num_iters):
            opt.zero_grad(set_to_none=True)
            loss_chunks = []
            for i0 in range(0, n, mb):
                u = torch.sigmoid(raw)
                phys = self._unnorm_u(u)
                sl = slice(i0, min(i0 + mb, n))
                phys_i = {k: v[sl] for k, v in phys.items()}
                batch = self._batch_from_phys(
                    phys_i,
                    float(end_con[0]),
                    float(end_time[0]),
                )
                pred = self.model_pred_torch(
                    pde_batch=batch,
                    end_con=end_con,
                    end_time=end_time,
                )
                loss_vec = self.fit_loss_diff(pred, gt)
                loss_vec.sum().backward()
                loss_chunks.append(loss_vec.detach())
            with torch.no_grad():
                loss_all = torch.cat(loss_chunks, dim=0)
                improved = loss_all < best_loss
                best_loss = torch.where(improved, loss_all, best_loss)
                u_now = torch.sigmoid(raw)
                best_u = torch.where(improved.unsqueeze(-1), u_now, best_u)
                history.append(float(loss_all.min().cpu()))
            opt.step()
            print(
                f'  gd iter {len(history)}/{self.num_iters} min_diff_loss={history[-1]:.4e}',
                flush=True,
            )
        phys_best = self._unnorm_u(best_u)
        out = all_paras.clone()
        order = self.pde_processor.para_order
        for i, k in enumerate(order):
            if k in phys_best:
                out[:, i] = phys_best[k]
        return out.detach().cpu(), best_loss.detach().cpu(), history


class LBFGSInverse(InverseAlg):
    def __init__(
            self,
            solver: nn.Module,
            paras_cfg,
            *,
            zero_first_n: int,
            max_iter: int,
            lr: float,
            history_size: int,
            tolerance_grad: float,
            tolerance_change: float,
            target_peak_weight: float,
            peak_position_weight: float,
            peak_height_weight: float,
        ) -> None:
        super().__init__(solver=solver, paras_cfg=paras_cfg)
        if zero_first_n < 0:
            raise ValueError('zero_first_n must be non-negative')
        if max_iter <= 0:
            raise ValueError('max_iter must be positive')
        if lr <= 0:
            raise ValueError('lr must be positive')
        if history_size <= 0:
            raise ValueError('history_size must be positive')
        if target_peak_weight < 0:
            raise ValueError('target_peak_weight must be non-negative')
        if peak_position_weight < 0:
            raise ValueError('peak_position_weight must be non-negative')
        if peak_height_weight < 0:
            raise ValueError('peak_height_weight must be non-negative')
        self.zero_first_n = int(zero_first_n)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.history_size = int(history_size)
        self.tolerance_grad = float(tolerance_grad)
        self.tolerance_change = float(tolerance_change)
        self.target_peak_weight = float(target_peak_weight)
        self.peak_position_weight = float(peak_position_weight)
        self.peak_height_weight = float(peak_height_weight)
        for parameter in self.solver.parameters():
            parameter.requires_grad_(False)
        self.solver.eval()

    def _logit(self, value: Tensor, eps: float = 1e-6) -> Tensor:
        value = value.clamp(eps, 1.0 - eps)
        return torch.log(value) - torch.log1p(-value)

    def _unnorm_u(self, value: Tensor) -> dict[str, Tensor]:
        packed = {key: value[:, i] for i, key in enumerate(PHYSICAL_KEYS)}
        return self.pde_processor.dict_unnorm_01(packed)

    def _batch_from_phys(
            self,
            phys: dict[str, Tensor],
            end_con: float,
            end_time: float,
        ) -> PDEBatch:
        batch_size = phys[PHYSICAL_KEYS[0]].shape[0]
        reference = phys[PHYSICAL_KEYS[0]]
        values = {
            'end_con': torch.full(
                (batch_size,), end_con, device=reference.device, dtype=reference.dtype
            ),
            'end_time': torch.full(
                (batch_size,), end_time, device=reference.device, dtype=reference.dtype
            ),
        }
        values.update(phys)
        return PDEBatch(**values)

    def _loss_vector(self, prediction: Tensor, target: Tensor) -> Tensor:
        if self.zero_first_n >= prediction.shape[1]:
            raise ValueError('zero_first_n must be smaller than the prediction length')
        n_time = min(prediction.shape[1], target.shape[0])
        prediction = prediction[:, self.zero_first_n:n_time]
        target = target[self.zero_first_n:n_time]
        scale = target.abs().amax(dim=0).clamp_min(1e-5)
        residual = (prediction - target.unsqueeze(0)) / scale.reshape(1, 1, -1)
        target_relative = target.abs() / scale.reshape(1, -1)
        weights = 1.0 + self.target_peak_weight * target_relative.square()
        weighted = residual.square() * weights.unsqueeze(0)
        normalizer = weights.sum()
        mse = weighted.sum(dim=(1, 2)) / normalizer
        time = torch.arange(
            self.zero_first_n,
            n_time,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        prediction_mass = prediction.clamp_min(0).square()
        target_mass = target.clamp_min(0).square()
        prediction_center = (
            prediction_mass * time.reshape(1, -1, 1)
        ).sum(dim=1) / prediction_mass.sum(dim=1).clamp_min(1e-12)
        target_center = (
            target_mass * time.reshape(-1, 1)
        ).sum(dim=0) / target_mass.sum(dim=0).clamp_min(1e-12)
        peak_position = (
            prediction_center - target_center.reshape(1, -1)
        ).square().mean(dim=-1) / n_time
        peak_height = (
            (
                prediction.amax(dim=1)
                - target.amax(dim=0).reshape(1, -1)
            )
            / scale.reshape(1, -1)
        ).square().mean(dim=-1)
        return (
            mse
            + self.peak_position_weight * peak_position
            + self.peak_height_weight * peak_height
        )

    def _zero_artifact(self, prediction: Tensor) -> Tensor:
        prediction = prediction.clone()
        prediction[:, :self.zero_first_n] = 0
        return prediction

    def _random_initial(self, n_starts: int, seed: int, device: torch.device) -> Tensor:
        if n_starts <= 0:
            raise ValueError('n_starts must be positive')
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return torch.rand(
            (n_starts, len(PHYSICAL_KEYS)),
            generator=generator,
            device=device,
            dtype=torch.float32,
        ).clamp(1e-4, 1.0 - 1e-4)

    def physical_to_u(self, physical: Tensor) -> Tensor:
        if physical.ndim != 2 or physical.shape[1] != len(PHYSICAL_KEYS):
            raise ValueError(
                f'physical must have shape (n, {len(PHYSICAL_KEYS)}), got {tuple(physical.shape)}'
            )
        packed = {
            key: physical[:, i]
            for i, key in enumerate(PHYSICAL_KEYS)
        }
        normed = self.pde_processor.dict_norm_01(packed)
        stacked = torch.stack([normed[key] for key in PHYSICAL_KEYS], dim=-1)
        return stacked.clamp(1e-4, 1.0 - 1e-4)

    def optimize(
            self,
            end_con: ndarray,
            end_time: ndarray,
            gt_cl_t: ndarray,
            *,
            n_starts: int | None = None,
            seed: int = 0,
            initial_u: Tensor | None = None,
            initial_physical: Tensor | None = None,
        ) -> dict:
        end_con = torch.as_tensor(end_con, dtype=torch.float64).cpu().numpy().reshape(-1)
        end_time = torch.as_tensor(end_time, dtype=torch.float64).cpu().numpy().reshape(-1)
        if end_con.shape != end_time.shape:
            raise ValueError('end_con and end_time must have the same shape')
        target_np = torch.as_tensor(gt_cl_t, dtype=torch.float32).cpu().numpy()
        if target_np.ndim != 2 or target_np.shape[1] != end_con.shape[0]:
            raise ValueError('gt_cl_t must have shape (n_time, n_experiments)')
        device = next(self.solver.parameters()).device
        target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
        if initial_physical is not None and initial_u is not None:
            raise ValueError('pass only one of initial_physical or initial_u')
        if initial_physical is not None:
            initial_u = self.physical_to_u(
                torch.as_tensor(initial_physical, dtype=torch.float32, device=device)
            )
        elif initial_u is not None:
            initial_u = torch.as_tensor(initial_u, dtype=torch.float32, device=device)
            if initial_u.ndim != 2 or initial_u.shape[1] != len(PHYSICAL_KEYS):
                raise ValueError(
                    f'initial_u must have shape (n, {len(PHYSICAL_KEYS)}), got {tuple(initial_u.shape)}'
                )
            initial_u = initial_u.clamp(1e-4, 1.0 - 1e-4)
        else:
            if n_starts is None:
                raise ValueError('n_starts is required without initial_u/initial_physical')
            initial_u = self._random_initial(n_starts, seed, device)
        n_starts = int(initial_u.shape[0])
        raw = nn.Parameter(self._logit(initial_u))
        optimizer = torch.optim.LBFGS(
            [raw],
            lr=self.lr,
            max_iter=self.max_iter,
            history_size=self.history_size,
            tolerance_grad=self.tolerance_grad,
            tolerance_change=self.tolerance_change,
            line_search_fn='strong_wolfe',
        )
        best_u = initial_u.detach().clone()
        best_loss = torch.full(
            (n_starts,), float('inf'), device=device, dtype=torch.float32
        )
        history = []
        evaluation = 0
        chunk_size = max(1, int(self.pred_chunk_size))

        def closure():
            nonlocal evaluation
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for start in range(0, n_starts, chunk_size):
                # Each chunk needs its own sigmoid graph before backward frees it.
                current_u = torch.sigmoid(raw)
                stop = min(start + chunk_size, n_starts)
                phys = self._unnorm_u(current_u[start:stop])
                batch = self._batch_from_phys(
                    phys,
                    float(end_con[0]),
                    float(end_time[0]),
                )
                prediction = self.model_pred_torch(
                    pde_batch=batch,
                    end_con=end_con,
                    end_time=end_time,
                )
                loss_chunk = self._loss_vector(prediction, target)
                (loss_chunk.sum() / n_starts).backward()
                losses.append(loss_chunk.detach())
                with torch.no_grad():
                    improved = loss_chunk < best_loss[start:stop]
                    best_loss[start:stop] = torch.where(
                        improved, loss_chunk, best_loss[start:stop]
                    )
                    best_u[start:stop] = torch.where(
                        improved.unsqueeze(-1),
                        current_u[start:stop].detach(),
                        best_u[start:stop],
                    )
            loss_vector = torch.cat(losses)
            row = {
                'evaluation': evaluation,
                'loss_min': float(loss_vector.min().cpu()),
                'loss_mean': float(loss_vector.mean().cpu()),
                'loss_max': float(loss_vector.max().cpu()),
                'best_loss_min': float(best_loss.min().cpu()),
            }
            history.append(row)
            print(
                f"lbfgs eval {evaluation} loss={row['loss_min']:.6e} "
                f"best={row['best_loss_min']:.6e}",
                flush=True,
            )
            evaluation += 1
            return loss_vector.mean()

        optimizer.step(closure)
        with torch.no_grad():
            current_u = torch.sigmoid(raw)
            phys = self._unnorm_u(current_u)
            batch = self._batch_from_phys(
                phys,
                float(end_con[0]),
                float(end_time[0]),
            )
            prediction = self.model_pred_torch(
                pde_batch=batch,
                end_con=end_con,
                end_time=end_time,
            )
            prediction = self._zero_artifact(prediction)
            final_loss = self._loss_vector(prediction, target)
            improved = final_loss < best_loss
            best_loss = torch.where(improved, final_loss, best_loss)
            best_u = torch.where(improved.unsqueeze(-1), current_u, best_u)
            phys = self._unnorm_u(best_u)
            batch = self._batch_from_phys(
                phys,
                float(end_con[0]),
                float(end_time[0]),
            )
            prediction = self.model_pred_torch(
                pde_batch=batch,
                end_con=end_con,
                end_time=end_time,
            )
            prediction = self._zero_artifact(prediction)
            final_loss = self._loss_vector(prediction, target)
            order = torch.argsort(final_loss)
            batch.update(batch.values[order])
            prediction = prediction[order]
            final_loss = final_loss[order]
            best_u = best_u[order]
            initial_u = initial_u[order]
        return {
            'all_paras': batch.values.detach().cpu(),
            'losses': final_loss.detach().cpu(),
            'model_pred': prediction.detach().cpu().numpy(),
            'initial_u': initial_u.detach().cpu(),
            'final_u': best_u.detach().cpu(),
            'history': history,
        }
