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

__all__ = ['GatedFCBlock']


class GatedFCBlock(nn.Module):
    """
    Drop-in encoder block for FastConformer experiments that uses a gated MLP instead
    of the stock attention+conv stack while preserving pre-norm residual structure.

    Args:
        d_model: Hidden dimension.
        expansion_factor: Width multiplier for the gated hidden state.
        dropout: Dropout rate applied before the residual add.
        use_bias: Whether the linear layers use bias.
    """

    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1, use_bias: bool = True):
        super().__init__()

        if expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {expansion_factor}")

        inner_dim = d_model * expansion_factor
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2, bias=use_bias)
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
        a, b = torch.chunk(self.in_proj(x), chunks=2, dim=-1)
        x = torch.nn.functional.silu(a) * torch.sigmoid(b)
        x = self.out_proj(x)
        x = residual + self.dropout(x)

        if cache_last_channel is None:
            return x
        return x, cache_last_channel, cache_last_time
