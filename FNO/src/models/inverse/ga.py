from __future__ import annotations

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from torch import Tensor
import numpy as np
from numpy import ndarray

from tqdm import tqdm
from .inverse_alg import InverseAlg

try:
    from IPython.display import clear_output, display
except ImportError:
    def clear_output(wait=False):
        return None

    def display(obj):
        return None

from src.utils.draw import draw_loss, draw_cl
from src.data.batch import PDEBatch

__all__ = ['SimpleGA']


class GA(InverseAlg):
    def __init__(
            self,
            solver: nn.Module,
            paras_cfg,
            *,
            loss_weights: dict,
            zero_first_n: int | None,
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
    
    def tv_loss(self, x: Tensor, dim: int) -> Tensor:
        return torch.abs(torch.diff(x, dim=dim)).sum(dim=dim)

    def fit_loss(self, model_pred: Tensor, gt_value: Tensor) -> Tensor:
        # model_pred.shape == (pop_size, n_time_pred, n_exper)
        # gt_value.shape == (n_time_gt, n_exper)
        if self.zero_first_n is not None:
            model_pred = model_pred.clone()
            model_pred[:, :self.zero_first_n] = 0

        n_time = min(model_pred.shape[1], gt_value.shape[0])

        model_pred = model_pred[:, :n_time, :]
        gt_value = gt_value[:n_time, :]

        normalize_cl = gt_value.max(0).values + 1e-5

        penalty_on_0 = model_pred.max(1).values
        penalty_on_0 = torch.where(
            penalty_on_0 < 5e-3 * normalize_cl,
            torch.tensor(1e8, dtype=model_pred.dtype, device=model_pred.device),
            torch.tensor(0.0, dtype=model_pred.dtype, device=model_pred.device)
        ).mean(-1)

        mse = torch.square(
            (model_pred - gt_value.unsqueeze(0)) / normalize_cl.reshape(1, 1, -1)
        ).reshape(model_pred.shape[0], -1).mean(-1)

        # peak_pos.shape == (pop_size, )
        peak_time = torch.argmax(model_pred, 1)
        peak_pos = torch.square(
            (peak_time - torch.argmax(gt_value, 0).reshape(1, -1)).float()
        ).mean(-1) / n_time

        peak_v = torch.square(
            (torch.max(model_pred, 1).values - torch.max(gt_value, 0).values.reshape(1, -1)) / normalize_cl.reshape(1, -1)
        ).mean(-1)

        # gt_tv.shape == (n_exper,), model_tv.shape == (pop_size, n_exper)
        gt_tv = self.tv_loss(gt_value, dim=0) + 1e-8  # (n_exper,)
        model_tv = self.tv_loss(model_pred, dim=1)     # (pop_size, n_exper)
        tv_threshold = 3.0 * gt_tv.unsqueeze(0)  # (1, n_exper)
        tv_penalty = 1e8 * torch.relu(model_tv - tv_threshold).mean(-1)  # (pop_size,)

        # print()
        # print('mse', mse.mean(), mse.median(), mse.max())
        # print('penalty_on_0', penalty_on_0.mean(), penalty_on_0.median(), penalty_on_0.max())
        # print('peak_pos', peak_pos.mean(), peak_pos.median(), peak_pos.max())
        # print('peak_v', peak_v.mean(), peak_v.median(), peak_v.max())
        # print('tv_penalty', tv_penalty.mean(), tv_penalty.median(), tv_penalty.max())
        # print()
        return (
            self.loss_weights['mse'] * mse
            + self.loss_weights['penalty_on_0'] * penalty_on_0
            + self.loss_weights['peak_pos'] * peak_pos
            + self.loss_weights['peak_v'] * peak_v
            + self.loss_weights['tv_penalty'] * tv_penalty
        )
        # return 5e-2 * mse + peak_pos + 2e-1 * peak_v


class SimpleGA(GA):
    def __init__(
            self,
            solver,
            paras_cfg,
            *,
            loss_weights: dict,
            zero_first_n: int | None,

            sol_per_pop: int = 500,
            num_generations: int = 50,
            mutation_probability: float = 0.1,
            max_copies: int = 20,
            elite_size: int = 2,
            tournament_size: int = 3,
            **kwargs
        ) -> None:
        super().__init__(
            solver,
            paras_cfg,
            loss_weights=loss_weights,
            zero_first_n=zero_first_n,
        )

        self.pop_size = sol_per_pop
        self.generations = num_generations
        self.mutation_rate = mutation_probability
        self.max_copies = max_copies
        self.elite_size = min(max(1, elite_size), self.pop_size)
        self.tournament_size = min(max(2, tournament_size), self.pop_size)

        self.n_top_k = sol_per_pop // 10
        self.max_try = 1000

    def _para_cache_key(self, p_np: ndarray):
        phys_idx = self.physical_indices()
        return np.asarray(p_np, dtype=np.float32)[phys_idx].tobytes()

    def _norm_physical_dict(self, para_row: Tensor) -> dict[str, Tensor]:
        order = self.pde_processor.para_order
        paras = {
            k: para_row[i]
            for i, k in enumerate(order)
            if k in self.pde_processor.initializer_para_order
        }
        return self.pde_processor.dict_norm_01(paras)

    def _write_physical(self, para_row: Tensor, physical: dict) -> Tensor:
        child = para_row.clone()
        order = self.pde_processor.para_order
        for i, k in enumerate(order):
            if k not in physical:
                continue
            child[i] = torch.as_tensor(
                physical[k],
                device=child.device,
                dtype=child.dtype,
            )
        return child

    def selection(
            self,
            populations: Tensor,
            fitnesses: Tensor,
        ) -> tuple[Tensor, Tensor]:
        order = torch.argsort(fitnesses)
        top_indices = order[:self.n_top_k]

        selected_idx = top_indices.clone()
        selection_count = torch.zeros(self.pop_size, dtype=torch.int64, device=fitnesses.device)
        selection_count[top_indices] += 1

        remaining_to_select = self.pop_size - self.n_top_k
        pop_indices = torch.arange(self.pop_size, dtype=torch.long, device=fitnesses.device)
        selected_remainder = []

        for _ in range(remaining_to_select):
            chosen_idx = None
            for _ in range(self.max_try):
                candidate_ids = torch.randint(
                    0, self.pop_size, (self.tournament_size,),
                    device=fitnesses.device
                )
                candidates = pop_indices[candidate_ids]
                winner = candidates[torch.argmin(fitnesses[candidates])].item()
                if selection_count[winner].item() < self.max_copies:
                    chosen_idx = winner
                    break
            if chosen_idx is None:
                available = torch.where(selection_count < self.max_copies)[0]
                if len(available) == 0:
                    selection_count.zero_()
                    selection_count[top_indices] += 1
                    available = torch.where(selection_count < self.max_copies)[0]
                chosen_idx = available[torch.randint(0, len(available), (1,), device=fitnesses.device)].item()
            selected_remainder.append(chosen_idx)
            selection_count[chosen_idx] += 1

        selected_idx = torch.cat(
            [
                selected_idx,
                torch.as_tensor(selected_remainder, dtype=torch.long, device=fitnesses.device)
            ]
        )
        return populations[selected_idx], fitnesses[selected_idx]

    def crossover(
            self,
            parent1: Tensor, parent2: Tensor
        ) -> Tensor:
        n1 = self._norm_physical_dict(parent1)
        n2 = self._norm_physical_dict(parent2)
        keys = self.pde_processor.initializer_para_order
        mask = torch.randint(
            0, 2, (len(keys),),
            device=parent1.device,
        ).to(dtype=parent1.dtype)
        mixed = {}
        for i, k in enumerate(keys):
            v1 = torch.as_tensor(n1[k], device=parent1.device, dtype=parent1.dtype)
            v2 = torch.as_tensor(n2[k], device=parent1.device, dtype=parent1.dtype)
            mixed[k] = mask[i] * v1 + (1 - mask[i]) * v2
        unnormed = self.pde_processor.dict_unnorm_01(mixed)
        return self._write_physical(parent1, unnormed)

    def mutate(
            self,
            child_para: Tensor,
            mutation_rate: float
        ) -> Tensor:
        if torch.rand(1).item() < mutation_rate:
            new_para = child_para.clone()
            phys_idx = self.physical_indices()
            mutate_point = phys_idx[torch.randint(0, len(phys_idx), (1,)).item()]
            para_key = self.pde_processor.para_order[mutate_point]
            u = torch.rand((), device=child_para.device, dtype=child_para.dtype)
            new_para[mutate_point] = self.pde_processor.dict_unnorm_01(
                {para_key: u}
            )[para_key]
            return new_para
        return child_para

    def next_gen(
            self,
            fitnesses: Tensor,
            all_paras: Tensor,
        ) -> Tensor:
        order = torch.argsort(fitnesses)
        elites = all_paras[order[:self.elite_size]]

        n_offspring = self.pop_size - self.elite_size
        n_pool = len(all_paras)
        offspring = []

        for _ in range(n_offspring):
            parent_ids = torch.randint(
                0, n_pool, (2,),
                device=all_paras.device
            )
            child = self.crossover(
                all_paras[parent_ids[0]],
                all_paras[parent_ids[1]],
            )
            child = self.mutate(child, self.mutation_rate)
            offspring.append(child)

        offspring = torch.stack(offspring)
        return torch.cat([elites, offspring], dim=0)

    def optimize(
            self,
            end_con: ndarray, end_time: ndarray,
            gt_cl_t: ndarray,
            draw_fig: bool = False,
            top_n: float = 0.3,
            return_global_best: bool = True,
            first_population: PDEBatch | None = None
        ) -> tuple[Tensor, Tensor, plt.Figure]:
        """optimize
        Args:
            end_con: ndarray, end_time: ndarray: shape == (n_exper, )
            gt_cl_t (ndarray): shape == (n_time, n_exper)
            return_global_best (bool): if True, return the best solutions from all generations;
                if False, return the best solutions only from the last population
        """

        loss_track = []
        fig = None
        if draw_fig:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

        gt_cl_t_tensor = torch.as_tensor(gt_cl_t, dtype=torch.float32)
        if first_population is None:
            pde_batch = self.rand_init(self.pop_size)
        else:
            if first_population.values.ndim != 2:
                raise ValueError()
            if first_population.values.shape[0] != self.pop_size:
                raise ValueError()
            if first_population.values.shape[1] != len(self.pde_processor.para_order):
                raise ValueError()
            pde_batch = first_population

        pde_batch = self.pin_oper_params(pde_batch, end_con, end_time)

        para_memo = {}
        pbar = tqdm(range(self.generations))
        for epoch in pbar:
            current_paras = pde_batch.values
            current_paras_cpu = current_paras.detach().cpu().numpy()
            
            needed_indices = []
            needed_paras = []
            keys = []
            pending_new_keys = set()

            for i, p_np in enumerate(current_paras_cpu):
                k = self._para_cache_key(p_np)
                keys.append(k)
                if k not in para_memo and k not in pending_new_keys:
                    needed_indices.append(i)
                    needed_paras.append(current_paras[i])
                    pending_new_keys.add(k)
            
            if needed_paras:
                batch_to_run = torch.stack(needed_paras)
                
                temp_pde_batch = self.rand_init(len(batch_to_run))
                temp_pde_batch.update(batch_to_run)

                new_preds = self.model_pred(
                    pde_batch=temp_pde_batch,
                    end_con=end_con, end_time=end_time
                )
                if isinstance(new_preds, ndarray):
                    new_preds = torch.from_numpy(new_preds).float()

                new_fitnesses = self.fit_loss(new_preds, gt_cl_t_tensor)

                for idx, fit, pred in zip(needed_indices, new_fitnesses, new_preds):
                    k = keys[idx]
                    para_memo[k] = {
                        'para': current_paras[idx].detach().clone(),
                        'pred': pred.detach().clone(),
                        'fitness': fit.detach().clone(),
                    }
            
            pop_preds_list = []
            pop_fitness_list = []
            
            for k in keys:
                pop_preds_list.append(para_memo[k]['pred'])
                pop_fitness_list.append(para_memo[k]['fitness'])
            
            model_pred = torch.stack(pop_preds_list)
            fitnesses = torch.stack(pop_fitness_list)

            best_child = torch.argmin(fitnesses).item()
            loss_track.append(fitnesses.detach().cpu().numpy().copy())

            if draw_fig and (epoch == self.generations - 1 or epoch % 5 == 0):
                for ax in axes:
                    ax.clear()
                draw_loss(loss_track, axes[0])
                draw_cl(
                    -model_pred[best_child].detach().cpu().numpy(), gt_cl_t, axes[1],
                    gt_bul=None
                )
                fig.canvas.draw()

            if epoch != self.generations - 1:
                selected_paras, selected_fitnesses = self.selection(
                    pde_batch.values, fitnesses
                )
                pde_batch.update(
                    self.next_gen(
                        selected_fitnesses,
                        selected_paras
                    )
                )
                pde_batch = self.pin_oper_params(pde_batch, end_con, end_time)

            pbar.set_description(
                f'Best Fitness: {torch.min(fitnesses).item():.4e} | Evaluated Params: {len(para_memo)}'
            )

        if draw_fig:
            plt.close(fig)

            fig = plt.figure(figsize=(20, 10))

            ax0 = fig.add_subplot(1, 2, 1)
            draw_loss(loss_track, ax0)

            ax1 = fig.add_subplot(1, 2, 2)
            draw_cl(
                -model_pred[best_child].detach().cpu().numpy(), gt_cl_t, ax1,
                gt_bul=None
            )

            fig.tight_layout()

        if return_global_best:
            all_paras_list = []
            all_fitness_list = []
            for k in para_memo:
                all_paras_list.append(para_memo[k]['para'])
                all_fitness_list.append(para_memo[k]['fitness'])
            all_paras = torch.stack(all_paras_list)
            fitnesses = torch.stack(all_fitness_list)
        else:
            last_paras = pde_batch.values
            last_paras_cpu = last_paras.detach().cpu().numpy()
            all_paras_list = []
            all_fitness_list = []
            for p_np in last_paras_cpu:
                k = self._para_cache_key(p_np)
                if k in para_memo:
                    all_paras_list.append(para_memo[k]['para'])
                    all_fitness_list.append(para_memo[k]['fitness'])
            all_paras = torch.stack(all_paras_list)
            fitnesses = torch.stack(all_fitness_list)

        if isinstance(top_n, int):
            n_best = min(max(1, top_n), len(fitnesses))
        elif isinstance(top_n, float):
            if not (0 < top_n <= 1):
                raise ValueError('top_n as float should be in (0, 1].')
            n_best = min(max(1, int(np.ceil(len(fitnesses) * top_n))), len(fitnesses))
        else:
            raise TypeError(f'top_n should be int or float, got {type(top_n)}')
        order_best = torch.argsort(fitnesses)[:n_best]
        all_paras, fitnesses = (
            all_paras[order_best],
            fitnesses[order_best]
        )

        return all_paras, fitnesses, fig, loss_track