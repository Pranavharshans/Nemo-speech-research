#!/usr/bin/env python
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn

__all__ = [
    "BiasNorm",
    "SwooshR",
    "ZipformerDownsampleBlock",
    "ZipformerUpsampleFusionBlock",
]


class BiasNorm(nn.Module):
    """
    A lightweight BiasNorm module based on the Zipformer paper.

    This normalizes by RMS(x - b) across the channel dimension and applies a
    learnable scalar output scale.
    """

    def __init__(self, num_channels: int, channel_dim: int = -1, log_scale: float = 0.0, eps: float = 1.0e-5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale)))
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channel_dim = self.channel_dim if self.channel_dim >= 0 else x.ndim + self.channel_dim
        bias = self.bias
        shape = [1] * x.ndim
        shape[channel_dim] = bias.numel()
        bias = bias.view(*shape)
        centered = x - bias
        rms = torch.sqrt(torch.mean(centered * centered, dim=channel_dim, keepdim=True) + self.eps)
        return x * (self.log_scale.exp() / rms)


class SwooshR(nn.Module):
    """
    SwooshR activation from the Zipformer paper:
      log(1 + exp(x - 1)) - 0.08x - 0.313261687
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.exp(x - 1.0)) - 0.08 * x - 0.313261687


class ZipformerDownsampleBlock(nn.Module):
    """
    A lightweight learned time reduction block for the lower-frame-rate middle stage.
    """

    def __init__(
        self,
        d_model: int,
        downsample_factor: int = 2,
        kernel_size: int = 3,
        expansion_factor: int = 2,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()
        if downsample_factor < 1:
            raise ValueError(f"downsample_factor must be >= 1, got {downsample_factor}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for same-padding behavior, got {kernel_size}")

        inner_dim = d_model * expansion_factor
        self.downsample_factor = downsample_factor
        self.norm = BiasNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim, bias=use_bias)
        self.depthwise = nn.Conv1d(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=kernel_size,
            stride=downsample_factor,
            padding=kernel_size // 2,
            groups=inner_dim,
            bias=use_bias,
        )
        self.out_proj = nn.Linear(inner_dim, d_model, bias=use_bias)
        self.activation = SwooshR()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.norm(x)
        x = self.in_proj(x)
        x = x.transpose(1, 2)
        x = self.depthwise(x)
        x = x.transpose(1, 2)
        x = self.activation(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        lengths = torch.div(lengths + self.downsample_factor - 1, self.downsample_factor, rounding_mode="floor")
        return x, lengths


class ZipformerUpsampleFusionBlock(nn.Module):
    """
    Upsamples the low-rate path back to the skip path length and fuses the two streams.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, use_bias: bool = True):
        super().__init__()
        self.high_norm = BiasNorm(d_model)
        self.low_norm = BiasNorm(d_model)
        self.low_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.fusion_proj = nn.Linear(2 * d_model, d_model, bias=use_bias)
        self.gate_proj = nn.Linear(2 * d_model, d_model, bias=use_bias)
        self.activation = SwooshR()
        self.dropout = nn.Dropout(dropout)

    def forward(self, high_res: torch.Tensor, low_res: torch.Tensor) -> torch.Tensor:
        target_len = high_res.size(1)
        high_res_residual = high_res
        low_res = self.low_norm(low_res)
        low_res = self.low_proj(low_res)
        low_res = torch.nn.functional.interpolate(
            low_res.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        high_res = self.high_norm(high_res)
        fused = torch.cat([high_res, low_res], dim=-1)
        gate = torch.sigmoid(self.gate_proj(fused))
        update = self.fusion_proj(self.activation(fused))
        return high_res_residual + self.dropout(update * gate)
