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

import pytest
import torch

from nemo.collections.asr.modules.tempcov_gatedfc_conformer_encoder import TempConvGatedFCConformerEncoder
from nemo.collections.asr.parts.submodules.gated_fc_modules import GatedFCBlock
from nemo.collections.asr.parts.submodules.temp_conv_modules import TempConvBlock


class TestTempConvGatedFCConformerEncoder:
    @pytest.mark.unit
    def test_replaces_requested_layers(self):
        model = TempConvGatedFCConformerEncoder(
            feat_in=10,
            n_layers=8,
            d_model=8,
            feat_out=8,
            subsampling_factor=2,
            conv_kernel_size=5,
            conv_context_size=[4, 0],
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            temp_conv_layer_indices=[2],
            gated_fc_layer_indices=[3],
            temp_conv_kernel_size=5,
        )

        assert isinstance(model.layers[2], TempConvBlock)
        assert isinstance(model.layers[3], GatedFCBlock)
        assert model.get_layer_type(1) == "fastconformer"
        assert model.get_layer_type(2) == "temp_conv"
        assert model.get_layer_type(3) == "gated_fc"

    @pytest.mark.unit
    def test_forward(self):
        batch_size = 2
        feat_in = 10
        n_frames = 16
        feat_out = 8

        model = TempConvGatedFCConformerEncoder(
            feat_in=feat_in,
            n_layers=4,
            d_model=8,
            feat_out=feat_out,
            subsampling_factor=2,
            conv_kernel_size=5,
            conv_context_size=[4, 0],
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            temp_conv_layer_indices=[1],
            gated_fc_layer_indices=[2],
            temp_conv_kernel_size=5,
        )

        audio = torch.randn(batch_size, feat_in, n_frames)
        length = torch.tensor([n_frames, n_frames - 3], dtype=torch.int64)
        encoded, encoded_len = model(audio_signal=audio, length=length)

        assert encoded.shape[0] == batch_size
        assert encoded.shape[1] == feat_out
        assert encoded.shape[2] == torch.max(encoded_len)

    @pytest.mark.unit
    def test_invalid_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            TempConvGatedFCConformerEncoder(
                feat_in=10,
                n_layers=4,
                d_model=8,
                temp_conv_layer_indices=[1],
                gated_fc_layer_indices=[1],
            )
