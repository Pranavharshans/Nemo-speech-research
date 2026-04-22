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

__all__ = [
    'BranchformerLayer',
    'ConvolutionalGatingMLP',
    'MultiScaleConvolutionalGatingMLP',
    'MultiScaleBranchformerLayer',
    'DeltaBranchformerLayer',
    'TemporalDeltaBranch',
]


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


class MultiScaleConvolutionalGatingMLP(nn.Module):
    """
    cgMLP branch with multiple parallel depthwise convolutions merged through
    learned softmax-normalized per-scale weights.
    """

    def __init__(
        self,
        d_model: int,
        linear_units: int,
        kernel_sizes: list[int],
        dropout: float,
        use_linear_after_conv: bool = False,
        gate_activation: str = 'identity',
        use_bias: bool = True,
    ):
        super().__init__()

        if linear_units % 2 != 0:
            raise ValueError(f"linear_units must be even for cgMLP gating, got {linear_units}")
        if len(kernel_sizes) == 0:
            raise ValueError("kernel_sizes must not be empty for MultiScaleConvolutionalGatingMLP")
        if any(kernel_size <= 0 or kernel_size % 2 == 0 for kernel_size in kernel_sizes):
            raise ValueError(f"All kernel_sizes must be positive odd integers, got {kernel_sizes}")

        self.linear_units = linear_units
        self.hidden_units = linear_units // 2
        self.use_linear_after_conv = use_linear_after_conv
        self.kernel_sizes = list(kernel_sizes)

        self.channel_proj_in = nn.Linear(d_model, linear_units, bias=use_bias)
        self.channel_proj_out = nn.Linear(self.hidden_units, d_model, bias=use_bias)
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(self.hidden_units)
        self.depthwise_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.hidden_units,
                    out_channels=self.hidden_units,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=self.hidden_units,
                    bias=use_bias,
                )
                for kernel_size in self.kernel_sizes
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(self.kernel_sizes), dtype=torch.float32))

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
        if cache is not None:
            raise NotImplementedError("MultiScaleConvolutionalGatingMLP currently supports offline execution only.")

        x = self.channel_proj_in(x)
        x = self.activation(x)
        x = self.dropout(x)

        x_residual, x_gate = x.chunk(2, dim=-1)
        x_gate = self.norm(x_gate)

        if pad_mask is not None:
            x_gate = x_gate.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        x_gate = x_gate.transpose(1, 2)
        conv_outputs = [depthwise_conv(x_gate) for depthwise_conv in self.depthwise_convs]
        scale_weights = torch.softmax(self.scale_logits, dim=0)
        x_gate = sum(weight * conv_output for weight, conv_output in zip(scale_weights, conv_outputs))
        x_gate = x_gate.transpose(1, 2)

        if self.linear_after_conv is not None:
            x_gate = self.linear_after_conv(x_gate)
        x_gate = self.gate_activation(x_gate)

        x = x_residual * x_gate
        x = self.channel_proj_out(x)
        x = self.dropout(x)
        return x


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


class TemporalDeltaBranch(nn.Module):
    """
    Lightweight temporal-difference branch for Delta-Branchformer.

    It computes first- and optional second-order frame differences, projects
    them back to `d_model`, and gates the result using the current hidden state.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float,
        use_second_order: bool = True,
        alpha_init: float = 1.0,
        use_bias: bool = True,
    ):
        super().__init__()

        self.use_second_order = use_second_order
        delta_input_dim = d_model * 2 if use_second_order else d_model

        self.delta_norm = LayerNorm(delta_input_dim)
        self.delta_proj = nn.Linear(delta_input_dim, d_model, bias=use_bias)
        self.gate_proj = nn.Linear(d_model, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.branch_scale = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    @staticmethod
    def _temporal_difference(x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        return delta

    def forward(self, x, pad_mask=None):
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        delta1 = self._temporal_difference(x)
        if pad_mask is not None:
            delta1 = delta1.masked_fill(pad_mask.unsqueeze(-1), 0.0)

        if self.use_second_order:
            delta2 = self._temporal_difference(delta1)
            if pad_mask is not None:
                delta2 = delta2.masked_fill(pad_mask.unsqueeze(-1), 0.0)
            delta_features = torch.cat([delta1, delta2], dim=-1)
        else:
            delta_features = delta1

        delta_features = self.delta_norm(delta_features)
        delta_features = self.delta_proj(delta_features)
        gate = torch.sigmoid(self.gate_proj(x))
        out = self.branch_scale * delta_features * gate
        return self.dropout(out)


class DeltaBranchformerLayer(BranchformerLayer):
    """
    Branchformer layer with an additional gated temporal-difference branch.
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
        delta_use_second_order: bool = True,
        delta_alpha_init: float = 1.0,
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            cgmlp_linear_units=cgmlp_linear_units,
            cgmlp_conv_kernel=cgmlp_conv_kernel,
            self_attention_model=self_attention_model,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            n_heads=n_heads,
            merge_conv_kernel=merge_conv_kernel,
            dropout=dropout,
            dropout_att=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            att_context_size=att_context_size,
            conv_context_size=conv_context_size,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            use_ffn=use_ffn,
            macaron_ffn=macaron_ffn,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )

        self.norm_delta = LayerNorm(d_model)
        self.delta_branch = TemporalDeltaBranch(
            d_model=d_model,
            dropout=dropout,
            use_second_order=delta_use_second_order,
            alpha_init=delta_alpha_init,
            use_bias=use_bias,
        )
        self.depthwise_conv_fusion = nn.Conv1d(
            in_channels=d_model * 3,
            out_channels=d_model * 3,
            kernel_size=merge_conv_kernel,
            stride=1,
            padding=(merge_conv_kernel - 1) // 2,
            groups=d_model * 3,
            bias=use_bias,
        )
        self.merge_proj = nn.Linear(d_model * 3, d_model, bias=use_bias)

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

        x_delta = self.norm_delta(x)
        x_delta = self.delta_branch(x_delta, pad_mask=pad_mask)

        x_concat = torch.cat([x_att, x_mlp, x_delta], dim=-1)
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


class MultiScaleBranchformerLayer(BranchformerLayer):
    """
    Branchformer layer with a multi-scale cgMLP local branch.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        cgmlp_linear_units: int,
        cgmlp_scale_kernels: list[int],
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
        primary_kernel = cgmlp_scale_kernels[len(cgmlp_scale_kernels) // 2]
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            cgmlp_linear_units=cgmlp_linear_units,
            cgmlp_conv_kernel=primary_kernel,
            self_attention_model=self_attention_model,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            n_heads=n_heads,
            merge_conv_kernel=merge_conv_kernel,
            dropout=dropout,
            dropout_att=dropout_att,
            pos_bias_u=pos_bias_u,
            pos_bias_v=pos_bias_v,
            att_context_size=att_context_size,
            conv_context_size=conv_context_size,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            use_ffn=use_ffn,
            macaron_ffn=macaron_ffn,
            use_bias=use_bias,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
        )
        self.cgmlp = MultiScaleConvolutionalGatingMLP(
            d_model=d_model,
            linear_units=cgmlp_linear_units,
            kernel_sizes=cgmlp_scale_kernels,
            dropout=dropout,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            use_bias=use_bias,
        )
