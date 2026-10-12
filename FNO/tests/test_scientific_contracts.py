"""Numerical contracts that must survive release cleanup."""
from functools import partial
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.data.batch import PARA_ORDER, PDEBatch
from src.models.pde_processor import PDEProcessor
from src.models.net import FNO, RolloutNet
from src.models.inverse.gd import LBFGSInverse

CONFIG = OmegaConf.load(Path(__file__).resolve().parents[1] / 'data/para_info.yaml')


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_physical_round_trip_and_conditioning_order():
    processor = PDEProcessor(CONFIG.paras)
    unit = {key: torch.tensor([.25, .75]) for key in PARA_ORDER}
    physical = processor.dict_unnorm_01(unit)
    normalized = processor.dict_norm_01(physical)
    for key in PARA_ORDER:
        torch.testing.assert_close(normalized[key], unit[key])
    batch = PDEBatch(**physical)
    static, rates = batch.solver_input(12, CONFIG.paras, 1)
    assert static.shape == (2, 17)
    assert rates.shape == (2, 12, 2)
    torch.testing.assert_close(static[:, :11], 2 * torch.stack(list(unit.values()), -1) - 1)
    psi = .05 + (batch.end_con - .05) / batch.end_time
    torch.testing.assert_close(rates[:, 0, 0], .1 * batch.ka * torch.exp(-batch.sa * psi))
    torch.testing.assert_close(rates[:, 0, 1], .1 * batch.kd * torch.exp(-batch.sd * psi))


@pytest.mark.parametrize('bul_only', [False, True])
def test_rollout_field_layout_and_parameter_gradient(bul_only):
    torch.manual_seed(4)
    dims = OmegaConf.create(dict(n_time=3, n_z=9, n_r=2, n_para=11, n_exp_para=6, n_oper_para=2))
    model = RolloutNet(net=partial(FNO, n_modes_height=6, hidden_channels=4),
                       cond_embed_size=4, cond_n_layers=3, time_downsample=1,
                       bul_only=bul_only, var_dims=dims, time_emb=False, relu=False)
    static = torch.rand(2, 17, requires_grad=True)
    rates = torch.rand(2, 3, 2, requires_grad=True)
    output = model(static, rates)
    if bul_only:
        assert output.shape == (2, 3, 9, 1)
        loss = output.square().mean()
    else:
        assert [value.shape for value in output] == [(2, 3, 9), (2, 3, 18), (2, 3, 18)]
        loss = sum(value.square().mean() for value in output)
    loss.backward()
    assert torch.isfinite(static.grad).all() and static.grad.abs().sum() > 0
    assert torch.isfinite(rates.grad).all() and rates.grad.abs().sum() > 0


class AnalyticSolver(torch.nn.Module):
    """A positive parameter-sensitive curve with an independently known optimum."""
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(1.))

    def forward_tensor(self, batch, paras_range):
        time = torch.linspace(.2, 1, 12, device=batch.sa.device)
        return (.1 + batch.sa[:, None] / 10) * time[None, :] * self.anchor


def test_multistart_lbfgs_multiple_chunks_reduce_known_loss():
    algorithm = LBFGSInverse(AnalyticSolver(), CONFIG.paras, zero_first_n=0,
        max_iter=15, lr=1., history_size=10, tolerance_grad=1e-8, tolerance_change=1e-10,
        target_peak_weight=0., peak_position_weight=0., peak_height_weight=0.)
    algorithm.pred_chunk_size = 1
    target = (.1 + .7) * torch.linspace(.2, 1, 12)[:, None]
    initial = torch.full((3, 9), .3)
    result = algorithm.optimize(np.array([.95]), np.array([120.]), target.numpy(), initial_u=initial)
    assert torch.isfinite(result['losses']).all()
    assert result['losses'].max() < 1e-5
    torch.testing.assert_close(result['all_paras'][:, PARA_ORDER.index('sa')], torch.full((3,), 7.), atol=.03, rtol=.01)
    assert not algorithm.solver.anchor.requires_grad
