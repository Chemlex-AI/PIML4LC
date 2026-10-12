from __future__ import annotations

import torch
from torch.utils.data import Dataset

import h5py

import numpy as np

from typing import Literal
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.utils import RankedLogger
from src.data.batch import PDEBatch
from .datamodule import DataModule

__all__ = ['SolverDataModule']

log = RankedLogger(__name__, rank_zero_only=True)

class SolverDataModule(DataModule):
    def __init__(
        self,
        h5_file_path: str,
        var_dims: DictConfig[str, int],
        loader_cfg: DictConfig,
        paras_cfg_path: str,
        apply_abs: bool,
        bul_only: bool,
        time_downsample: int,
        sample_filter: DictConfig | None = None,
        shuffle_seed: int = 42,
        debug: bool = False,
        **kwargs
    ):
        super().__init__(loader_cfg, collator=self.collate_fn)
        self.save_hyperparameters(logger=False)
        
        self.h5_file_path = h5_file_path
        self.var_dims = var_dims
        self.paras_cfg_path = paras_cfg_path

        self.apply_abs = apply_abs
        self.bul_only = bul_only
        self.sample_filter = sample_filter
        self.shuffle_seed = shuffle_seed
        self.debug = debug
        self.time_downsample = time_downsample
        self._prepare_indices()

    def _prepare_indices(self):
        log.info(f'Reading dataset from {self.h5_file_path}')
        paras_cfg = OmegaConf.load(self.paras_cfg_path).paras
        with h5py.File(self.h5_file_path, 'r') as h5f:
            total_samples = h5f['bul'].shape[0]
            log.info(f'Total samples in dataset: {total_samples}')
            all_indices = np.arange(total_samples)
            if self.sample_filter is not None:
                filter_cfg = OmegaConf.to_container(self.sample_filter, resolve=True)
                keep_mask = np.ones(total_samples, dtype=bool)
                for key, values in filter_cfg.items():
                    data = np.asarray(h5f[key][:])
                    value_array = np.asarray(values)
                    candidate_values = [value_array]
                    if (
                        np.issubdtype(data.dtype, np.floating)
                        and key in paras_cfg
                    ):
                        data_min = float(np.min(data))
                        data_max = float(np.max(data))
                        if -1e-8 <= data_min and data_max <= 1 + 1e-8:
                            para_min = float(paras_cfg[key].min)
                            para_max = float(paras_cfg[key].max)
                            normalized_values = (value_array - para_min) / (para_max - para_min)
                            candidate_values.append(normalized_values)
                    if np.issubdtype(data.dtype, np.floating):
                        current_mask = np.zeros(total_samples, dtype=bool)
                        for candidates in candidate_values:
                            for value in candidates:
                                current_mask |= np.isclose(data, value)
                    else:
                        current_mask = np.isin(data, value_array)
                    keep_mask &= current_mask
                all_indices = all_indices[keep_mask]
                log.info(
                    f'Filtered samples by {filter_cfg}: '
                    f'{len(all_indices)}/{total_samples} kept'
                )
        
        if self.shuffle_seed is not None:
            rng = np.random.RandomState(self.shuffle_seed)
            rng.shuffle(all_indices)
            log.info(f'Shuffled indices with seed {self.shuffle_seed}')

        filtered_total = len(all_indices)
        if filtered_total < 10:
            raise ValueError('At least 10 samples are required for nonempty train/validation/test splits.')
        test_n = int(min(1e4, 0.1 * filtered_total))
        train_n = filtered_total - 2 * test_n
        self.train_indices = all_indices[:train_n].tolist()
        self.valid_indices = all_indices[train_n:train_n+test_n].tolist()
        self.test_indices = all_indices[train_n+test_n:].tolist()

        if self.debug:
            max_samples = 512
            self.train_indices = self.train_indices[:max_samples]
            self.valid_indices = self.valid_indices[:max_samples]
            self.test_indices = self.test_indices[:max_samples]
            log.info(
                f'Debug mode: limiting to at most {max_samples} samples '
                f'train={len(self.train_indices)}, '
                f'valid={len(self.valid_indices)}, '
                f'test={len(self.test_indices)}'
            )
            
        log.info(
            f'Data split: train={len(self.train_indices)}, '
            f'valid={len(self.valid_indices)}, '
            f'test={len(self.test_indices)}'
        )

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        self.stage = stage
        
        dataset_kargs = {
            "h5_file_path": self.h5_file_path,
            "var_dims": self.var_dims,
            "paras_cfg_path": self.paras_cfg_path,
            "apply_abs": self.apply_abs,
            "bul_only": self.bul_only,
            "time_downsample": self.time_downsample,
        }
        match stage:
            case "fit":
                log.info(f'Setting up training dataset with {len(self.train_indices)} samples')
                self.train_dataset = SolverDataset(
                    sample_index=self.train_indices,
                    **dataset_kargs,
                )
                log.info(f'Setting up validation dataset with {len(self.valid_indices)} samples')
                self.valid_dataset = SolverDataset(
                    sample_index=self.valid_indices,
                    **dataset_kargs,
                )
                
            case "validate":
                log.info(f'Setting up validation dataset with {len(self.valid_indices)} samples')
                self.valid_dataset = SolverDataset(
                    sample_index=self.valid_indices,
                    **dataset_kargs,
                )
                
            case "test" | "predict":
                log.info(f'Setting up test dataset with {len(self.test_indices)} samples')
                self.test_dataset = SolverDataset(
                    sample_index=self.test_indices,
                    **dataset_kargs,
                )
                
            case _:
                raise ValueError(f"Invalid stage: {stage}")

    def collate_fn(self, batch_list: list):
        if self.bul_only:
            bul_batch = torch.stack([item[0] for item in batch_list], dim=0)
            pde_paras = torch.stack([item[3] for item in batch_list], dim=0)
            paras_t = torch.stack([item[4] for item in batch_list], dim=0)
            pde_batch = PDEBatch.from_list([item[-1] for item in batch_list])
            return [bul_batch, None, None, pde_paras, paras_t, pde_batch]
        else:
            data = [
                torch.stack([item[i] for item in batch_list], dim=0)
                for i in range(5)
            ]
            return data + [PDEBatch.from_list([item[-1] for item in batch_list])]



class SolverDataset(Dataset):
    def __init__(
        self,
        h5_file_path: str,
        sample_index: list[int],
        var_dims: DictConfig[str, int],
        paras_cfg_path: str,
        apply_abs: bool,
        bul_only: bool,
        time_downsample: int,
    ):
        self.h5_file_path = h5_file_path
        self.indices = sample_index
        self.var_dims = var_dims

        self.apply_abs = apply_abs
        self.bul_only = bul_only
        self.time_downsample = time_downsample
        self.n_time_full = var_dims.n_time
        assert self.n_time_full <= var_dims.total_time
        self.n_time = (
            self.n_time_full + time_downsample - 1
        ) // time_downsample

        self.paras_range = OmegaConf.load(paras_cfg_path)['paras']
        field_norm = OmegaConf.load(paras_cfg_path).field_norm
        self.field_norm = {
            'bul': float(field_norm.bul),
            'par': float(field_norm.par),
            'sol': float(field_norm.sol),
        }

        with h5py.File(self.h5_file_path, 'r') as h5f:
            vd = self.var_dims
            nz_nr = vd.n_z * vd.n_r
            assert h5f['bul'][0].shape == (vd.total_time, vd.n_z)
            assert h5f['par'][0].shape == (vd.total_time, nz_nr)
            assert h5f['sol'][0].shape == (vd.total_time, nz_nr)

            self.pde_paras: list[torch.Tensor] = []
            self.paras_t: list[torch.Tensor] = []
            self.pde_batches: list[PDEBatch] = []

            for data_idx in tqdm(self.indices, desc="Precomputing PDE parameters"):
                pdes = {
                    p: torch.tensor(
                        h5f[p][data_idx].astype(np.float64)
                    )
                    for p in self.para_names
                }
                pde_batch = PDEBatch(**pdes)
                pde_paras, paras_t = pde_batch.solver_input(
                    n_time=self.n_time,
                    paras_range=self.paras_range,
                    time_stride=self.time_downsample,
                )
                self.pde_paras.append(pde_paras.float())
                self.paras_t.append(paras_t.float())
                self.pde_batches.append(pde_batch)

    @property
    def para_names(self):
        return [
            'end_con', 'end_time',
            'sa', 'sd', 'sq', 'ka', 'kd', 'qmax',
            'col_disp', 'fil_diff', 'par_diff',
        ]
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]

        with h5py.File(self.h5_file_path, 'r') as h5f:
            sl = slice(None, self.n_time_full, self.time_downsample)
            bul = torch.from_numpy(
                h5f['bul'][data_idx, sl]
            ).float()
            if self.bul_only:
                par, sol = None, None
            else:
                par = torch.from_numpy(
                    h5f['par'][data_idx, sl]
                ).float()
                sol = torch.from_numpy(
                    h5f['sol'][data_idx, sl]
                ).float()
        pde_batch = self.pde_batches[idx]

        if self.apply_abs:
            bul = torch.abs(bul)
            if not self.bul_only:
                par, sol = torch.abs(par), torch.abs(sol)

        if not self.bul_only:
            par = par * self.field_norm['par']
            sol = sol * self.field_norm['sol']

        return bul, par, sol, \
            self.pde_paras[idx], self.paras_t[idx], pde_batch
    



