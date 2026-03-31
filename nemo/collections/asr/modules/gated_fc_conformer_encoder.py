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

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.gated_fc_modules import GatedFCBlock

__all__ = ['GatedFCConformerEncoder']


class GatedFCConformerEncoder(ConformerEncoder):
    """
    Experimental mixed encoder that swaps selected FastConformer layers with GatedFC blocks.
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
        conv_norm_type='batch_norm',
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
        gated_fc_layer_indices=None,
        gated_fc_expansion_factor: int = 4,
    ):
        if gated_fc_layer_indices is None:
            gated_fc_layer_indices = [5, 6]

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
        )

        self.gated_fc_layer_indices = self._validate_layer_indices(gated_fc_layer_indices, n_layers=n_layers)
        self.gated_fc_expansion_factor = gated_fc_expansion_factor

        for layer_idx in self.gated_fc_layer_indices:
            self.layers[layer_idx] = GatedFCBlock(
                d_model=d_model,
                expansion_factor=gated_fc_expansion_factor,
                dropout=dropout,
                use_bias=use_bias,
            )

        self.setup_streaming_params()

    @staticmethod
    def _validate_layer_indices(layer_indices, n_layers: int) -> list[int]:
        seen = set()
        indices = []
        for layer_idx in layer_indices:
            if layer_idx < 0 or layer_idx >= n_layers:
                raise ValueError(f"gated_fc_layer_indices must be within [0, {n_layers - 1}], got {layer_idx}")
            if layer_idx in seen:
                raise ValueError(f"gated_fc_layer_indices contains duplicate layer index {layer_idx}")
            seen.add(layer_idx)
            indices.append(layer_idx)
        return indices

    def get_layer_type(self, layer_idx: int) -> str:
        if isinstance(self.layers[layer_idx], GatedFCBlock):
            return "gated_fc"
        return "fastconformer"
