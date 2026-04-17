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

from nemo.collections.asr.modules.branchformer_encoder import BranchformerEncoder
from nemo.collections.asr.parts.submodules.branchformer_modules import DeltaBranchformerLayer

__all__ = ['DeltaBranchformerEncoder']


class DeltaBranchformerEncoder(BranchformerEncoder):
    """
    Experimental Delta-Branchformer encoder that augments each Branchformer
    layer with a gated temporal-difference branch.
    """

    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        causal_downsampling=False,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_chunking_factor=1,
        subsampling_conv_channels=-1,
        reduction=None,
        reduction_position=None,
        reduction_factor=1,
        ff_expansion_factor=4,
        self_attention_model='rel_pos',
        n_heads=4,
        att_context_size=None,
        att_context_probs=None,
        att_context_style='regular',
        xscaling=True,
        untie_biases=True,
        pos_emb_max_len=5000,
        conv_kernel_size=31,
        conv_norm_type='layer_norm',
        conv_context_size=None,
        use_bias=True,
        dropout=0.1,
        dropout_pre_encoder=0.1,
        dropout_emb=0.1,
        dropout_att=0.0,
        stochastic_depth_drop_prob: float = 0.0,
        stochastic_depth_mode: str = "linear",
        stochastic_depth_start_layer: int = 1,
        global_tokens: int = 0,
        global_tokens_spacing: int = 1,
        global_attn_separate: bool = False,
        use_pytorch_sdpa: bool = False,
        use_pytorch_sdpa_backends=None,
        sync_max_audio_length: bool = True,
        cgmlp_conv_kernel: int | None = None,
        cgmlp_expansion_factor: int = 4,
        merge_conv_kernel: int = 3,
        use_linear_after_conv: bool = False,
        gate_activation: str = 'identity',
        use_ffn: bool = True,
        macaron_ffn: bool = False,
        delta_use_second_order: bool = True,
        delta_alpha_init: float = 1.0,
    ):
        super().__init__(
            feat_in=feat_in,
            n_layers=n_layers,
            d_model=d_model,
            feat_out=feat_out,
            causal_downsampling=causal_downsampling,
            subsampling=subsampling,
            subsampling_factor=subsampling_factor,
            subsampling_conv_chunking_factor=subsampling_conv_chunking_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            reduction=reduction,
            reduction_position=reduction_position,
            reduction_factor=reduction_factor,
            ff_expansion_factor=ff_expansion_factor,
            self_attention_model=self_attention_model,
            n_heads=n_heads,
            att_context_size=att_context_size,
            att_context_probs=att_context_probs,
            att_context_style=att_context_style,
            xscaling=xscaling,
            untie_biases=untie_biases,
            pos_emb_max_len=pos_emb_max_len,
            conv_kernel_size=conv_kernel_size,
            conv_norm_type=conv_norm_type,
            conv_context_size=conv_context_size,
            use_bias=use_bias,
            dropout=dropout,
            dropout_pre_encoder=dropout_pre_encoder,
            dropout_emb=dropout_emb,
            dropout_att=dropout_att,
            stochastic_depth_drop_prob=stochastic_depth_drop_prob,
            stochastic_depth_mode=stochastic_depth_mode,
            stochastic_depth_start_layer=stochastic_depth_start_layer,
            global_tokens=global_tokens,
            global_tokens_spacing=global_tokens_spacing,
            global_attn_separate=global_attn_separate,
            use_pytorch_sdpa=use_pytorch_sdpa,
            use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
            sync_max_audio_length=sync_max_audio_length,
            cgmlp_conv_kernel=cgmlp_conv_kernel,
            cgmlp_expansion_factor=cgmlp_expansion_factor,
            merge_conv_kernel=merge_conv_kernel,
            use_linear_after_conv=use_linear_after_conv,
            gate_activation=gate_activation,
            use_ffn=use_ffn,
            macaron_ffn=macaron_ffn,
        )

        self.delta_use_second_order = delta_use_second_order
        self.delta_alpha_init = delta_alpha_init

        resolved_cgmlp_conv_kernel = cgmlp_conv_kernel if cgmlp_conv_kernel is not None else conv_kernel_size
        d_ff = d_model * ff_expansion_factor
        cgmlp_linear_units = d_model * cgmlp_expansion_factor * 2

        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None

        for layer_idx in range(n_layers):
            self.layers[layer_idx] = DeltaBranchformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                cgmlp_linear_units=cgmlp_linear_units,
                cgmlp_conv_kernel=resolved_cgmlp_conv_kernel,
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
                att_context_size=self.att_context_size,
                conv_context_size=self.conv_context_size,
                use_linear_after_conv=use_linear_after_conv,
                gate_activation=gate_activation,
                use_ffn=use_ffn,
                macaron_ffn=macaron_ffn,
                use_bias=use_bias,
                use_pytorch_sdpa=use_pytorch_sdpa,
                use_pytorch_sdpa_backends=use_pytorch_sdpa_backends,
                delta_use_second_order=delta_use_second_order,
                delta_alpha_init=delta_alpha_init,
            )

        self.setup_streaming_params()

    def get_layer_type(self, layer_idx: int) -> str:
        if isinstance(self.layers[layer_idx], DeltaBranchformerLayer):
            return "delta_branchformer"
        return "branchformer"
