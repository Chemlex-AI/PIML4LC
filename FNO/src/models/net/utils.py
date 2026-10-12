from __future__ import annotations

import math

import torch
import torch.nn as nn

from torch import Tensor
from einops import rearrange


__all__ = [
    'MPFourierEmbedding',
    'SinusoidalPosEmb',
    'MLP'
]

class MPFourierEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = False)

    def forward(self, x: Tensor):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * torch.pi
        return torch.cat((freqs.sin(), freqs.cos()), dim = -1)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self, hidden_dims: list):
        super(MLP, self).__init__()

        self.layers = [nn.Linear(hidden_dims[0], hidden_dims[1])]
        for i in range(1, len(hidden_dims) - 1):
            self.layers.append(nn.GELU())
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
    
