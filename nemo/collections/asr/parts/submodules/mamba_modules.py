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

from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D

__all__ = ['MambaBlock']


class MambaBlock(nn.Module):
    """
    A lightweight streaming-friendly Mamba-style block used for experimental ASR encoder variants.

    This is intentionally self-contained so V1 can live beside the stock FastConformer encoder without
    introducing third-party Mamba dependencies into the baseline codepath.

    Args:
        d_model: Hidden dimension.
        expand: Expansion factor for the selective projection branch.
        local_conv_kernel_size: Kernel size for the optional local depthwise convolution.
        local_conv_padding: Optional `[left, right]` padding override for the local convolution.
        dropout: Dropout rate applied before the residual add.
        use_bias: Whether linear/conv layers use bias.
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        local_conv_kernel_size: int = 4,
        local_conv_padding: Optional[list[int]] = None,
        dropout: float = 0.1,
        use_bias: bool = True,
    ):
        super().__init__()

        if expand < 1:
            raise ValueError(f"expand must be >= 1, got {expand}")

        inner_dim = d_model * expand
        self.d_model = d_model
        self.inner_dim = inner_dim

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 3, bias=use_bias)
        self.local_conv = CausalConv1D(
            in_channels=inner_dim,
            out_channels=inner_dim,
            kernel_size=local_conv_kernel_size,
            padding=local_conv_padding,
            groups=inner_dim,
            bias=use_bias,
        )
        self.delta_proj = nn.Linear(d_model, inner_dim, bias=use_bias)
        self.state_mix = nn.Parameter(torch.zeros(inner_dim))
        self.skip_scale = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def _get_initial_state(self, x: torch.Tensor, cache_last_channel: Optional[torch.Tensor]) -> torch.Tensor:
        if cache_last_channel is None or cache_last_channel.size(1) == 0:
            return x.new_zeros(x.size(0), self.inner_dim)
        return cache_last_channel[:, -1, : self.inner_dim]

    def _build_next_channel_cache(
        self,
        cache_last_channel: Optional[torch.Tensor],
        final_state: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if cache_last_channel is None:
            return None

        next_cache = torch.zeros_like(cache_last_channel)
        if next_cache.size(1) > 0:
            next_cache[:, -1, : self.inner_dim] = final_state
        return next_cache

    def forward(
        self,
        x: torch.Tensor,
        att_mask: Optional[torch.Tensor] = None,
        pos_emb: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        cache_last_channel: Optional[torch.Tensor] = None,
        cache_last_time: Optional[torch.Tensor] = None,
    ):
        del att_mask, pos_emb

        residual = x
        x = self.norm(x)

        proj, gate, drive = torch.chunk(self.in_proj(x), chunks=3, dim=-1)
        proj = torch.nn.functional.silu(proj)
        gate = torch.sigmoid(gate)

        conv_input = proj.transpose(1, 2)
        conv_out = self.local_conv(conv_input, cache=cache_last_time)
        if cache_last_time is not None:
            conv_out, cache_last_time = conv_out
        conv_out = conv_out.transpose(1, 2)

        delta = torch.sigmoid(self.delta_proj(x)).clamp_min(1e-4)
        drive = torch.tanh(drive)

        state = self._get_initial_state(x, cache_last_channel)
        outputs = []
        active_mask = None if pad_mask is None else (~pad_mask).to(x.dtype).unsqueeze(-1)

        for step in range(x.size(1)):
            step_delta = delta[:, step, :]
            recurrent_mix = torch.sigmoid(self.state_mix).unsqueeze(0)
            proposed = recurrent_mix * state + step_delta * drive[:, step, :]
            if active_mask is not None:
                token_mask = active_mask[:, step, :]
                state = proposed * token_mask + state * (1.0 - token_mask)
            else:
                state = proposed

            step_out = (state + self.skip_scale * conv_out[:, step, :]) * gate[:, step, :]
            if active_mask is not None:
                step_out = step_out * active_mask[:, step, :]
            outputs.append(step_out)

        x = torch.stack(outputs, dim=1)
        x = self.out_proj(x)
        x = residual + self.dropout(x)

        if cache_last_channel is None:
            return x

        cache_last_channel = self._build_next_channel_cache(cache_last_channel, final_state=state)
        return x, cache_last_channel, cache_last_time
