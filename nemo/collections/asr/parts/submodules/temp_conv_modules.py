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

from typing import Optional

import torch
from torch import nn

__all__ = ['TempConvBlock']


class TempConvBlock(nn.Module):
    """
    Cheap local temporal modeling block with pre-norm residual structure.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 2,
        kernel_size: int = 15,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()

        if expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {expansion_factor}")
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd for same-padding depthwise conv, got {kernel_size}")

        inner_dim = d_model * expansion_factor
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim, bias=use_bias)
        self.depthwise_conv = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=inner_dim,
            bias=use_bias,
        )
        self.out_proj = nn.Linear(inner_dim, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        cache_last_channel: Optional[torch.Tensor] = None,
        cache_last_time: Optional[torch.Tensor] = None,
    ):
        del att_mask, pos_emb, pad_mask

        residual = x
        x = self.norm(x)
        x = self.in_proj(x)
        x = self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)
        x = torch.nn.functional.silu(x)
        x = self.out_proj(x)
        x = residual + self.dropout(x)

        if cache_last_channel is None:
            return x
        return x, cache_last_channel, cache_last_time
