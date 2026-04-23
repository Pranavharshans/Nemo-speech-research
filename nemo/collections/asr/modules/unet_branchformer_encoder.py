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

import random

import torch
from torch import nn

from nemo.collections.asr.modules.branchformer_encoder import BranchformerEncoder
from nemo.collections.asr.parts.submodules.branchformer_modules import TemporalDownsampleBlock, TemporalUpsampleFusionBlock

__all__ = ['UNetBranchformerEncoder']


class UNetBranchformerEncoder(BranchformerEncoder):
    """
    U-Net-style Branchformer encoder with a full-rate -> low-rate -> full-rate schedule.

    This variant currently supports offline training/inference only.
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
        unet_stage_layers=None,
        unet_downsample_kernel_size: int = 3,
        unet_upsample_kernel_size: int = 3,
        capture_stage1_output: bool = False,
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

        if unet_stage_layers is None:
            unet_stage_layers = [4, 8, 4]
        if sum(unet_stage_layers) != n_layers:
            raise ValueError(f"unet_stage_layers must sum to n_layers. Got {unet_stage_layers} for n_layers={n_layers}")

        self.unet_stage_layers = list(unet_stage_layers)
        self.stage_boundaries = (self.unet_stage_layers[0], self.unet_stage_layers[0] + self.unet_stage_layers[1])
        self.capture_stage1_output = capture_stage1_output
        self._stage1_output = None
        self._stage1_length = None
        self.downsample = TemporalDownsampleBlock(
            d_model=d_model,
            kernel_size=unet_downsample_kernel_size,
            stride=2,
        )
        self.upsample_fusion = TemporalUpsampleFusionBlock(d_model=d_model, kernel_size=unet_upsample_kernel_size)

    def get_layer_type(self, layer_idx: int) -> str:
        stage1_end, stage2_end = self.stage_boundaries
        if layer_idx < stage1_end:
            return "unet_branchformer_fullres"
        if layer_idx < stage2_end:
            return "unet_branchformer_lowres"
        return "unet_branchformer_fullres"

    def _capture_interctc(self, layer_idx: int, audio_signal: torch.Tensor, length: torch.Tensor):
        if self.is_access_enabled(getattr(self, "model_guid", None)):
            if self.interctc_capture_at_layers is None:
                self.interctc_capture_at_layers = self.access_cfg.get('interctc', {}).get('capture_layers', [])
            if layer_idx in self.interctc_capture_at_layers:
                lth_audio_signal = audio_signal
                if self.out_proj is not None:
                    lth_audio_signal = self.out_proj(audio_signal)
                self.register_accessible_tensor(
                    name=f'interctc/layer_output_{layer_idx}', tensor=torch.transpose(lth_audio_signal, 1, 2)
                )
                self.register_accessible_tensor(name=f'interctc/layer_length_{layer_idx}', tensor=length)

    def _run_layer_range(self, x, length, start_layer: int, end_layer: int, att_context_size):
        max_audio_length = x.size(1)
        x, pos_emb = self.pos_enc(x=x, cache_len=0)
        pad_mask, att_mask = self._create_masks(
            att_context_size=att_context_size,
            padding_length=length,
            max_audio_length=max_audio_length,
            offset=None,
            device=x.device,
        )

        for layer_idx in range(start_layer, end_layer):
            original_signal = x
            x = self.layers[layer_idx](x=x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
            drop_prob = self.layer_drop_probs[layer_idx]
            if self.training and drop_prob > 0.0:
                should_drop = torch.rand(1, device=x.device) < drop_prob
                if should_drop:
                    x = x * 0.0 + original_signal
                else:
                    x = (x - original_signal) / (1.0 - drop_prob) + original_signal

            self._capture_interctc(layer_idx=layer_idx, audio_signal=x, length=length)

        return x, length

    def get_stage1_output(self):
        return self._stage1_output, self._stage1_length

    def clear_stage1_output(self):
        self._stage1_output = None
        self._stage1_length = None

    def forward_internal(
        self,
        audio_signal,
        length,
        cache_last_channel=None,
        cache_last_time=None,
        cache_last_channel_len=None,
        bypass_pre_encode=False,
    ):
        if cache_last_channel is not None or cache_last_time is not None or cache_last_channel_len is not None:
            raise NotImplementedError("UNetBranchformerEncoder currently supports offline execution only.")
        if self.reduction_position is not None:
            raise NotImplementedError("UNetBranchformerEncoder does not support reduction_position.")

        if length is None:
            length = audio_signal.new_full(
                (audio_signal.size(0),), audio_signal.size(-1), dtype=torch.int64, device=audio_signal.device
            )

        if self.training and len(self.att_context_size_all) > 1:
            cur_att_context_size = random.choices(self.att_context_size_all, weights=self.att_context_probs)[0]
        else:
            cur_att_context_size = self.att_context_size

        if not bypass_pre_encode:
            audio_signal = torch.transpose(audio_signal, 1, 2)
            if isinstance(self.pre_encode, nn.Linear):
                audio_signal = self.pre_encode(audio_signal)
            else:
                audio_signal, length = self.pre_encode(x=audio_signal, lengths=length)
                length = length.to(torch.int64)

        stage1_end, stage2_end = self.stage_boundaries
        high_res, length = self._run_layer_range(audio_signal, length, 0, stage1_end, cur_att_context_size)
        if self.capture_stage1_output:
            stage1_output = high_res
            if self.out_proj is not None:
                stage1_output = self.out_proj(stage1_output)
            self._stage1_output = torch.transpose(stage1_output, 1, 2)
            self._stage1_length = length.to(dtype=torch.int64)
        else:
            self.clear_stage1_output()
        skip = high_res

        low_res, low_length = self.downsample(high_res, length)
        low_res, low_length = self._run_layer_range(low_res, low_length, stage1_end, stage2_end, cur_att_context_size)

        fused = self.upsample_fusion(skip, low_res)
        fused, length = self._run_layer_range(fused, length, stage2_end, self.n_layers, cur_att_context_size)

        if self.out_proj is not None:
            fused = self.out_proj(fused)
        fused = torch.transpose(fused, 1, 2)
        length = length.to(dtype=torch.int64)
        return fused, length
