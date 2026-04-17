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
from torch.nn import LayerNorm

from nemo.collections.asr.parts.submodules.causal_convs import CausalConv1D
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerFeedForward
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
    RelPositionMultiHeadAttentionLongformer,
)
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.common.parts.utils import activation_registry

__all__ = ['BranchformerLayer', 'ConvolutionalGatingMLP']


class ConvolutionalGatingMLP(nn.Module):
    """
    Convolutional gating MLP used in Branchformer / E-Branchformer style local branches.

    The implementation follows the standard cgMLP structure:
    input -> linear -> activation -> split -> depthwise conv on gate branch -> elementwise gate -> linear.
    """

    def __init__(
        self,
        d_model: int,
        linear_units: int,
        kernel_size: int,
        dropout: float,
        conv_context_size=None,
        use_linear_after_conv: bool = False,
        gate_activation: str = 'identity',
        use_bias: bool = True,
    ):
        super().__init__()

        if linear_units % 2 != 0:
            raise ValueError(f"linear_units must be even for cgMLP gating, got {linear_units}")

        if conv_context_size is None:
            conv_context_size = (kernel_size - 1) // 2

        self.linear_units = linear_units
        self.hidden_units = linear_units // 2
        self.use_linear_after_conv = use_linear_after_conv

        self.channel_proj_in = nn.Linear(d_model, linear_units, bias=use_bias)
        self.channel_proj_out = nn.Linear(self.hidden_units, d_model, bias=use_bias)
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(self.hidden_units)
        self.depthwise_conv = CausalConv1D(
            in_channels=self.hidden_units,
            out_channels=self.hidden_units,
            kernel_size=kernel_size,
            stride=1,
            padding=conv_context_size,
            groups=self.hidden_units,
            bias=use_bias,
        )

        if use_linear_after_conv:
            self.linear_after_conv = nn.Linear(self.hidden_units, self.hidden_units, bias=use_bias)
        else:
            self.linear_after_conv = None

        if gate_activation not in activation_registry:
            raise ValueError(
                f"gate_activation='{gate_activation}' is not valid. Supported values: {sorted(activation_registry)}"
            )
        self.gate_activation = activation_registry[gate_activation]()

    def forward(self, x, pad_mask=None, cache=None):
        x = self.channel_proj_in(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_residual, x_gate = x.chunk(2, dim=-1)
        x_gate = self.norm(x_gate)

        if pad_mask is not None:
            x_gate = x_gate.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x_gate = x_gate.transpose(1, 2)
        x_gate = self.depthwise_conv(x_gate, cache=cache)
        if cache is not None:
            x_gate, cache = x_gate
        x_gate = x_gate.transpose(1, 2)

        if self.linear_after_conv is not None:
            x_gate = self.linear_after_conv(x_gate)
        x_gate = self.gate_activation(x_gate)

        x = x_residual * x_gate
        x = self.channel_proj_out(x)
        x = self.dropout(x)

        if cache is None:
            return x
        return x, cache


class BranchformerLayer(nn.Module):
    """
    E-Branchformer style encoder block with parallel attention and cgMLP branches.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        cgmlp_linear_units: int,
        cgmlp_conv_kernel: int,
        self_attention_model: str = 'rel_pos',
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        n_heads: int = 4,
        merge_conv_kernel: int = 3,
        dropout: float = 0.1,
        dropout_att: float = 0.1,
        pos_bias_u=None,
        pos_bias_v=None,
        att_context_size=[-1, -1],
        conv_context_size=None,
        use_linear_after_conv: bool = False,
        gate_activation: str = 'identity',
        use_ffn: bool = True,
        macaron_ffn: bool = True,
        use_bias: bool = True,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
    ):
        super().__init__()

        self.self_attention_model = self_attention_model
        self.use_ffn = use_ffn
        self.ff_scale = 1.0

        if use_ffn:
            self.norm_ff = LayerNorm(d_model)
            self.feed_forward = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias)
        else:
            self.norm_ff = None
            self.feed_forward = None

        if use_ffn and macaron_ffn:
            self.ff_scale = 0.5
            self.norm_ff_macaron = LayerNorm(d_model)
            self.feed_forward_macaron = ConformerFeedForward(
                d_model=d_model, d_ff=d_ff, dropout=dropout, use_bias=use_bias
            )
        else:
            self.norm_ff_macaron = None
            self.feed_forward_macaron = None

        self.norm_self_att = LayerNorm(d_model)
        self.norm_cgmlp = LayerNorm(d_model)
        self.norm_out = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if use_pytorch_sdpa_backends is None:
            use_pytorch_sdpa_backends = []

        mha_max_cache_len = att_context_size[0]
        if self_attention_model == 'rel_pos':
            self.self_attn = RelPositionMultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=mha_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa,
                use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            )
        elif self_attention_model == 'rel_pos_local_attn':
            self.self_attn = RelPositionMultiHeadAttentionLongformer(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
                max_cache_len=mha_max_cache_len,
                att_context_size=att_context_size,
                global_tokens=global_tokens,
                global_tokens_spacing=global_tokens_spacing,
                global_attn_separate=global_attn_separate,
                use_bias=use_bias,
            )
        elif self_attention_model == 'abs_pos':
            self.self_attn = MultiHeadAttention(
                n_head=n_heads,
                n_feat=d_model,
                dropout_rate=dropout_att,
                max_cache_len=mha_max_cache_len,
                use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa,
                use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            )
        else:
            raise ValueError(
                f"'{self_attention_model}' is not a valid self_attention_model for BranchformerLayer"
            )

        self.cgmlp = ConvolutionalGatingMLP(
            d_model=d_model,
            linear_units=cgmlp_linear_units,
            kernel_size=cgmlp_conv_kernel,
            dropout=dropout,
            conv_context_size=conv_context_size,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            use_bias=use_bias,
        )

        self.depthwise_conv_fusion = nn.Conv1d(
            in_channels=d_model * 2,
            out_channels=d_model * 2,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=d_model * 2,
            bias=use_bias,
        )
        self.merge_proj = nn.Linear(d_model * 2, d_model, bias=use_bias)

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None):
        if self.feed_forward_macaron is not None:
            residual = x
            x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        residual = x
        x_att = self.norm_self_att(x)
        if self.self_attention_model == 'rel_pos':
            x_att = self.self_attn(
                query=x_att, key=x_att, value=x_att, mask=att_mask, pos_emb=pos_emb, cache=cache_last_channel
            )
        elif self.self_attention_model == 'rel_pos_local_attn':
            x_att = self.self_attn(
                query=x_att, key=x_att, value=x_att, pad_mask=pad_mask, pos_emb=pos_emb, cache=cache_last_channel
            )
        else:
            x_att = self.self_attn(query=x_att, key=x_att, value=x_att, mask=att_mask, cache=cache_last_channel)

        if cache_last_channel is not None:
            x_att, cache_last_channel = x_att
        x_att = self.dropout(x_att)

        x_mlp = self.norm_cgmlp(x)
        x_mlp = self.cgmlp(x_mlp, pad_mask=pad_mask, cache=cache_last_time)
        if cache_last_time is not None:
            x_mlp, cache_last_time = x_mlp
        x_mlp = self.dropout(x_mlp)

        x_concat = torch.cat([x_att, x_mlp], dim=-1)
        x_merge = self.depthwise_conv_fusion(x_concat.transpose(1, 2)).transpose(1, 2)
        x = residual + self.dropout(self.merge_proj(x_concat + x_merge))

        if self.feed_forward is not None:
            residual = x
            x = self.norm_ff(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        x = self.norm_out(x)

        if cache_last_channel is None:
            return x
        return x, cache_last_channel, cache_last_time
