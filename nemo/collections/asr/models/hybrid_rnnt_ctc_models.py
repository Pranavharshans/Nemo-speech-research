# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import copy
from typing import Any, List, Optional, Tuple, Union

import editdistance
import torch
import torch.nn.functional as F
from torch import nn
from lightning.pytorch import Trainer
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.ctc import CTCLoss
from nemo.collections.asr.metrics.wer import WER
from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
from nemo.collections.asr.parts.mixins import ASRBPEMixin, ASRTranscriptionMixin, InterCTCMixin, TranscribeConfig
from nemo.collections.asr.parts.mixins.transcription import TranscriptionReturnType
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from nemo.collections.asr.parts.submodules.ctc_decoding import CTCDecoding, CTCDecodingConfig
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import process_timestamp_outputs
from nemo.collections.common.losses import SmoothedCrossEntropyLoss
from nemo.core.classes.common import PretrainedModelInfo
from nemo.core.classes.mixins import AccessMixin
from nemo.utils import logging, model_utils


class FastSlowTransducerFusion(nn.Module):
    """Blend shallow full-rate and deep refined encoder views for the transducer branch only."""

    def __init__(self, hidden_size: int, fast_weight_init: float = -1.38629436112):
        super().__init__()
        self.fast_projection = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fast_path_logit = nn.Parameter(torch.tensor(float(fast_weight_init), dtype=torch.float32))
        nn.init.normal_(self.fast_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.fast_projection.bias)

    def fast_path_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.fast_path_logit)

    def forward(self, slow_features: torch.Tensor, fast_features: torch.Tensor) -> torch.Tensor:
        if slow_features.shape != fast_features.shape:
            raise ValueError(
                "Fast/slow transducer fusion requires matching tensor shapes. "
                f"Got slow={tuple(slow_features.shape)} and fast={tuple(fast_features.shape)}."
            )

        fast_features = fast_features.transpose(1, 2)
        fast_features = self.fast_projection(fast_features)
        fast_features = fast_features.transpose(1, 2)

        fast_weight = self.fast_path_weight().to(device=slow_features.device, dtype=slow_features.dtype)
        return fast_weight * fast_features + (1.0 - fast_weight) * slow_features


class TextOnlyScanDecoder(nn.Module):
    """Lightweight text-only recurrent scanner over token-rate acoustic embeddings."""

    def __init__(
        self,
        acoustic_dim: int,
        vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
        embedding_dim: int = 256,
        hidden_size: int = 320,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.input_projection = nn.Linear(acoustic_dim + embedding_dim, hidden_size)
        self.rnn_cell = nn.LSTMCell(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, vocab_size)

    def _step(
        self,
        acoustic_step: torch.Tensor,
        prev_tokens: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.embedding(prev_tokens)
        step_input = torch.cat([acoustic_step, embedded], dim=-1)
        step_input = torch.tanh(self.input_projection(step_input))

        if state is None:
            h = acoustic_step.new_zeros(acoustic_step.size(0), self.rnn_cell.hidden_size)
            c = acoustic_step.new_zeros(acoustic_step.size(0), self.rnn_cell.hidden_size)
        else:
            h, c = state

        h, c = self.rnn_cell(step_input, (h, c))
        logits = self.output_projection(self.dropout(h))
        return logits, (h, c)

    def forward_teacher_forced(self, acoustic_tokens: torch.Tensor, decoder_inputs: torch.Tensor) -> torch.Tensor:
        logits = []
        state = None
        for step_idx in range(acoustic_tokens.size(1)):
            step_logits, state = self._step(acoustic_tokens[:, step_idx, :], decoder_inputs[:, step_idx], state)
            logits.append(step_logits)
        return torch.log_softmax(torch.stack(logits, dim=1), dim=-1)

    @torch.no_grad()
    def greedy_decode(
        self, acoustic_tokens: torch.Tensor, acoustic_token_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, max_steps, _ = acoustic_tokens.shape
        max_decode_steps = int(acoustic_token_lengths.max().item()) if acoustic_token_lengths.numel() > 0 else 0
        max_decode_steps = max(max_decode_steps, 1)
        predictions = torch.full(
            (batch_size, max_decode_steps),
            fill_value=self.pad_id,
            dtype=torch.long,
            device=acoustic_tokens.device,
        )
        prediction_lengths = acoustic_token_lengths.new_zeros(batch_size)

        prev_tokens = acoustic_tokens.new_full((batch_size,), fill_value=self.bos_id, dtype=torch.long)
        state = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=acoustic_tokens.device)

        for step_idx in range(max_decode_steps):
            acoustic_step = acoustic_tokens[:, min(step_idx, max_steps - 1), :]
            step_logits, state = self._step(acoustic_step, prev_tokens, state)
            next_tokens = step_logits.argmax(dim=-1)

            inactive = step_idx >= acoustic_token_lengths
            next_tokens = torch.where(inactive | finished, next_tokens.new_full((), self.eos_id), next_tokens)

            predictions[:, step_idx] = next_tokens
            first_finish = (~finished) & ((next_tokens == self.eos_id) | inactive)
            prediction_lengths = torch.where(
                first_finish, prediction_lengths.new_full((), step_idx), prediction_lengths
            )
            finished = finished | first_finish
            prev_tokens = next_tokens

        prediction_lengths = torch.where(
            prediction_lengths == 0,
            acoustic_token_lengths.clamp(min=1, max=max_decode_steps),
            prediction_lengths.clamp(min=1, max=max_decode_steps),
        )
        return predictions, prediction_lengths


class SoftTokenAlignerBridge(nn.Module):
    """Soft monotonic token pooling over the final encoder stream."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        temperature: float = 0.35,
        length_loss_weight: float = 0.25,
        max_decode_tokens: int = 192,
    ):
        super().__init__()
        self.temperature = temperature
        self.length_loss_weight = length_loss_weight
        self.max_decode_tokens = max_decode_tokens
        self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.weight_projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_token_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, hidden_size, max_time = encoder_outputs.shape
        features = encoder_outputs.transpose(1, 2)
        conv_features = self.temporal_conv(encoder_outputs).transpose(1, 2)
        raw_weights = F.softplus(self.weight_projection(torch.silu(conv_features)).squeeze(-1)) + 1e-4

        time_mask = torch.arange(max_time, device=encoder_outputs.device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        raw_weights = raw_weights * time_mask
        raw_mass = raw_weights.sum(dim=1)

        if target_token_lengths is None:
            token_lengths = raw_mass.round().clamp(min=1, max=self.max_decode_tokens).long()
            alignment_loss = raw_mass.new_zeros(())
        else:
            token_lengths = target_token_lengths.long().clamp(min=1, max=self.max_decode_tokens)
            alignment_loss = F.l1_loss(raw_mass, token_lengths.float()) * self.length_loss_weight

        max_tokens = int(token_lengths.max().item())
        acoustic_tokens = encoder_outputs.new_zeros((batch_size, max_tokens, hidden_size))

        for batch_idx in range(batch_size):
            frame_count = int(encoder_lengths[batch_idx].item())
            token_count = int(token_lengths[batch_idx].item())
            if frame_count <= 0 or token_count <= 0:
                continue

            frame_features = features[batch_idx, :frame_count]
            frame_weights = raw_weights[batch_idx, :frame_count]
            scale = token_count / frame_weights.sum().clamp_min(1e-6)
            scaled_weights = frame_weights * scale
            centers = torch.cumsum(scaled_weights, dim=0) - 0.5 * scaled_weights
            token_centers = torch.arange(token_count, device=encoder_outputs.device, dtype=centers.dtype) + 0.5
            distance = centers.unsqueeze(1) - token_centers.unsqueeze(0)
            assignments = torch.exp(-(distance**2) / (2.0 * self.temperature * self.temperature))
            assignments = assignments * scaled_weights.unsqueeze(1)
            assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-6)
            acoustic_tokens[batch_idx, :token_count] = assignments.transpose(0, 1) @ frame_features

        return acoustic_tokens, token_lengths, alignment_loss


class CIFTokenBridge(nn.Module):
    """Differentiable CIF-style token firing on a monotonic cumulative weight path."""

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 3,
        quantity_loss_weight: float = 1.0,
        max_decode_tokens: int = 192,
    ):
        super().__init__()
        self.quantity_loss_weight = quantity_loss_weight
        self.max_decode_tokens = max_decode_tokens
        self.temporal_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.weight_projection = nn.Linear(hidden_size, 1)

    def forward(
        self,
        encoder_outputs: torch.Tensor,
        encoder_lengths: torch.Tensor,
        target_token_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, hidden_size, max_time = encoder_outputs.shape
        features = encoder_outputs.transpose(1, 2)
        conv_features = self.temporal_conv(encoder_outputs).transpose(1, 2)
        alpha = torch.sigmoid(self.weight_projection(torch.silu(conv_features)).squeeze(-1))

        time_mask = torch.arange(max_time, device=encoder_outputs.device).unsqueeze(0) < encoder_lengths.unsqueeze(1)
        alpha = alpha * time_mask
        raw_mass = alpha.sum(dim=1)

        if target_token_lengths is None:
            token_lengths = raw_mass.round().clamp(min=1, max=self.max_decode_tokens).long()
            quantity_loss = raw_mass.new_zeros(())
        else:
            token_lengths = target_token_lengths.long().clamp(min=1, max=self.max_decode_tokens)
            quantity_loss = F.l1_loss(raw_mass, token_lengths.float()) * self.quantity_loss_weight

        max_tokens = int(token_lengths.max().item())
        acoustic_tokens = encoder_outputs.new_zeros((batch_size, max_tokens, hidden_size))

        for batch_idx in range(batch_size):
            frame_count = int(encoder_lengths[batch_idx].item())
            token_count = int(token_lengths[batch_idx].item())
            if frame_count <= 0 or token_count <= 0:
                continue

            frame_features = features[batch_idx, :frame_count]
            frame_alpha = alpha[batch_idx, :frame_count]
            scale = token_count / frame_alpha.sum().clamp_min(1e-6)
            scaled_alpha = frame_alpha * scale
            centers = torch.cumsum(scaled_alpha, dim=0) - 0.5 * scaled_alpha
            token_centers = torch.arange(token_count, device=encoder_outputs.device, dtype=centers.dtype) + 0.5
            distance = torch.abs(centers.unsqueeze(1) - token_centers.unsqueeze(0))
            assignments = F.relu(1.0 - distance)
            assignments = assignments * scaled_alpha.unsqueeze(1)
            assignments = assignments / assignments.sum(dim=0, keepdim=True).clamp_min(1e-6)
            acoustic_tokens[batch_idx, :token_count] = assignments.transpose(0, 1) @ frame_features

        return acoustic_tokens, token_lengths, quantity_loss


class EncDecHybridRNNTCTCModel(EncDecRNNTModel, ASRBPEMixin, InterCTCMixin, ASRTranscriptionMixin):
    """Base class for hybrid RNNT/CTC models."""

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        fast_slow_cfg = cfg.get("transducer_fast_slow_fusion", None)
        scan_cfg = cfg.get("experimental_scan_decoder", None)
        if fast_slow_cfg is not None and fast_slow_cfg.get("enabled", False):
            with open_dict(cfg.encoder):
                cfg.encoder.capture_stage1_output = True
        if (
            scan_cfg is not None
            and scan_cfg.get("enabled", False)
            and scan_cfg.get("mode", "").lower() == "cif"
            and scan_cfg.get("cif", {}).get("input_source", "stage1") == "stage1"
        ):
            with open_dict(cfg.encoder):
                cfg.encoder.capture_stage1_output = True
        super().__init__(cfg=cfg, trainer=trainer)

        self.transducer_fast_slow_fusion = None
        fast_slow_cfg = self.cfg.get("transducer_fast_slow_fusion", None)
        if fast_slow_cfg is not None and fast_slow_cfg.get("enabled", False):
            if not hasattr(self.encoder, "get_stage1_output"):
                raise ValueError(
                    "transducer_fast_slow_fusion requires an encoder that exposes stage-1 outputs via "
                    "`get_stage1_output()`."
                )
            self.transducer_fast_slow_fusion = FastSlowTransducerFusion(
                hidden_size=self.encoder._feat_out,
                fast_weight_init=fast_slow_cfg.get("fast_weight_init", -1.38629436112),
            )

        if 'aux_ctc' not in self.cfg:
            raise ValueError(
                "The config need to have a section for the CTC decoder named as aux_ctc for Hybrid models."
            )
        with open_dict(self.cfg.aux_ctc):
            if "feat_in" not in self.cfg.aux_ctc.decoder or (
                not self.cfg.aux_ctc.decoder.feat_in and hasattr(self.encoder, '_feat_out')
            ):
                self.cfg.aux_ctc.decoder.feat_in = self.encoder._feat_out
            if "feat_in" not in self.cfg.aux_ctc.decoder or not self.cfg.aux_ctc.decoder.feat_in:
                raise ValueError("param feat_in of the decoder's config is not set!")

            if self.cfg.aux_ctc.decoder.num_classes < 1 and self.cfg.aux_ctc.decoder.vocabulary is not None:
                logging.info(
                    "\nReplacing placeholder number of classes ({}) with actual number of classes - {}".format(
                        self.cfg.aux_ctc.decoder.num_classes, len(self.cfg.aux_ctc.decoder.vocabulary)
                    )
                )
                self.cfg.aux_ctc.decoder["num_classes"] = len(self.cfg.aux_ctc.decoder.vocabulary)

        self.ctc_decoder = EncDecRNNTModel.from_config_dict(self.cfg.aux_ctc.decoder)
        self.ctc_loss_weight = self.cfg.aux_ctc.get("ctc_loss_weight", 0.5)

        self.ctc_loss = CTCLoss(
            num_classes=self.ctc_decoder.num_classes_with_blank - 1,
            zero_infinity=True,
            reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
        )

        ctc_decoding_cfg = self.cfg.aux_ctc.get('decoding', None)
        if ctc_decoding_cfg is None:
            ctc_decoding_cfg = OmegaConf.structured(CTCDecodingConfig)
            with open_dict(self.cfg.aux_ctc):
                self.cfg.aux_ctc.decoding = ctc_decoding_cfg

        self.ctc_decoding = CTCDecoding(self.cfg.aux_ctc.decoding, vocabulary=self.ctc_decoder.vocabulary)
        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.cfg.aux_ctc.get('use_cer', False),
            dist_sync_on_step=True,
            log_prediction=self.cfg.get("log_prediction", False),
        )

        # setting the RNNT decoder as the default one
        self.cur_decoder = "rnnt"

        self.experimental_scan_cfg = self.cfg.get("experimental_scan_decoder", None)
        self.experimental_scan_decoder = None
        self.experimental_scan_bridge = None
        self.experimental_scan_loss = None
        if self.experimental_scan_cfg is not None and self.experimental_scan_cfg.get("enabled", False):
            if not hasattr(self, "tokenizer"):
                raise ValueError("experimental_scan_decoder requires a tokenizer-enabled hybrid model.")

            vocab_size = len(self.joint.vocabulary)
            self.experimental_scan_decoder = TextOnlyScanDecoder(
                acoustic_dim=self.encoder._feat_out,
                vocab_size=vocab_size,
                bos_id=self.tokenizer.bos_id,
                eos_id=self.tokenizer.eos_id,
                pad_id=self.tokenizer.pad_id,
                embedding_dim=self.experimental_scan_cfg.get("embedding_dim", self.encoder._feat_out),
                hidden_size=self.experimental_scan_cfg.get("hidden_size", 320),
                dropout=self.experimental_scan_cfg.get("dropout", 0.1),
            )
            self.experimental_scan_loss = SmoothedCrossEntropyLoss(
                pad_id=self.tokenizer.pad_id,
                label_smoothing=self.experimental_scan_cfg.get("label_smoothing", 0.0),
            )

            mode = self.experimental_scan_cfg.get("mode", "").lower()
            if mode == "aligner":
                aligner_cfg = self.experimental_scan_cfg.get("aligner", {})
                self.experimental_scan_bridge = SoftTokenAlignerBridge(
                    hidden_size=self.encoder._feat_out,
                    kernel_size=aligner_cfg.get("weight_conv_kernel", 3),
                    temperature=aligner_cfg.get("temperature", 0.35),
                    length_loss_weight=aligner_cfg.get("length_loss_weight", 0.25),
                    max_decode_tokens=self.experimental_scan_cfg.get("max_decode_tokens", 192),
                )
            elif mode == "cif":
                cif_cfg = self.experimental_scan_cfg.get("cif", {})
                self.experimental_scan_bridge = CIFTokenBridge(
                    hidden_size=self.encoder._feat_out,
                    kernel_size=cif_cfg.get("weight_conv_kernel", 3),
                    quantity_loss_weight=cif_cfg.get("quantity_loss_weight", 1.0),
                    max_decode_tokens=self.experimental_scan_cfg.get("max_decode_tokens", 192),
                )
            else:
                raise ValueError(
                    f"Unsupported experimental_scan_decoder mode: {self.experimental_scan_cfg.get('mode')}"
                )

        # setting up interCTC loss (from InterCTCMixin)
        self.setup_interctc(decoder_name='ctc_decoder', loss_name='ctc_loss', wer_name='ctc_wer')

    def _get_transducer_encoder_output(self, slow_encoded: torch.Tensor, encoded_len: Optional[torch.Tensor] = None):
        if self.transducer_fast_slow_fusion is None:
            return slow_encoded

        fast_encoded, fast_len = self.encoder.get_stage1_output()
        if fast_encoded is None:
            raise RuntimeError(
                "transducer_fast_slow_fusion is enabled but no stage-1 encoder features were captured in the "
                "current forward pass."
            )
        if fast_len is not None and encoded_len is not None and not torch.equal(fast_len, encoded_len):
            raise RuntimeError(
                "Stage-1 and final encoder lengths diverged during fast-slow fusion: "
                f"fast={fast_len.tolist()} slow={encoded_len.tolist()}"
            )

        transducer_encoded = self.transducer_fast_slow_fusion(slow_features=slow_encoded, fast_features=fast_encoded)
        self.encoder.clear_stage1_output()
        return transducer_encoded

    def _experimental_scan_enabled(self) -> bool:
        return self.experimental_scan_decoder is not None and self.experimental_scan_bridge is not None

    def _uses_stage1_scan_input(self) -> bool:
        if not self._experimental_scan_enabled():
            return False
        return (
            self.experimental_scan_cfg.get("mode", "").lower() == "cif"
            and self.experimental_scan_cfg.get("cif", {}).get("input_source", "stage1") == "stage1"
        )

    def _build_scan_decoder_targets(
        self, transcript: torch.Tensor, transcript_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = transcript.size(0)
        device = transcript.device
        target_lengths = transcript_len + 1
        max_target_length = int(target_lengths.max().item())

        decoder_targets = torch.full(
            (batch_size, max_target_length),
            fill_value=self.tokenizer.pad_id,
            dtype=transcript.dtype,
            device=device,
        )
        decoder_inputs = torch.full_like(decoder_targets, fill_value=self.tokenizer.pad_id)

        for batch_idx in range(batch_size):
            cur_len = int(transcript_len[batch_idx].item())
            if cur_len > 0:
                decoder_targets[batch_idx, :cur_len] = transcript[batch_idx, :cur_len]
                decoder_inputs[batch_idx, 1 : cur_len + 1] = transcript[batch_idx, :cur_len]
            decoder_targets[batch_idx, cur_len] = self.tokenizer.eos_id
            decoder_inputs[batch_idx, 0] = self.tokenizer.bos_id

        return decoder_inputs, decoder_targets, target_lengths

    def _get_scan_bridge_inputs(
        self, encoded: torch.Tensor, encoded_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self._uses_stage1_scan_input():
            return encoded, encoded_len

        if not hasattr(self.encoder, "get_stage1_output"):
            raise RuntimeError("CIF experimental scan decoder requires stage-1 encoder output support.")

        stage1_encoded, stage1_len = self.encoder.get_stage1_output()
        if stage1_encoded is None:
            raise RuntimeError("CIF experimental scan decoder expected captured stage-1 outputs but found none.")
        self.encoder.clear_stage1_output()
        return stage1_encoded, stage1_len if stage1_len is not None else encoded_len

    def _strip_special_tokens(self, token_ids: List[int]) -> List[int]:
        cleaned = []
        for token_id in token_ids:
            if token_id == self.tokenizer.eos_id:
                break
            if token_id in (self.tokenizer.pad_id, self.tokenizer.bos_id):
                continue
            cleaned.append(token_id)
        return cleaned

    def _compute_text_error_counts(
        self, predicted_tokens: torch.Tensor, predicted_lengths: torch.Tensor, transcript: torch.Tensor, transcript_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = 0
        words = 0

        for prediction, pred_len, target, target_len in zip(
            predicted_tokens.detach().cpu().tolist(),
            predicted_lengths.detach().cpu().tolist(),
            transcript.detach().cpu().tolist(),
            transcript_len.detach().cpu().tolist(),
        ):
            prediction = self._strip_special_tokens(prediction[:pred_len])
            target = target[:target_len]

            hypothesis = self.tokenizer.ids_to_text(prediction).strip()
            reference = self.tokenizer.ids_to_text(target).strip()

            ref_words = reference.split()
            hyp_words = hypothesis.split()
            words += len(ref_words)
            scores += editdistance.eval(hyp_words, ref_words)

        return (
            transcript_len.new_tensor(float(scores), dtype=torch.float32),
            transcript_len.new_tensor(float(words), dtype=torch.float32),
        )

    def _run_experimental_scan_decoder(
        self,
        encoded: torch.Tensor,
        encoded_len: torch.Tensor,
        transcript: torch.Tensor,
        transcript_len: torch.Tensor,
        compute_loss: bool = True,
        compute_predictions: bool = False,
    ):
        decoder_inputs, decoder_targets, target_lengths = self._build_scan_decoder_targets(transcript, transcript_len)
        bridge_inputs, bridge_lengths = self._get_scan_bridge_inputs(encoded, encoded_len)
        acoustic_tokens, acoustic_token_lengths, bridge_aux_loss = self.experimental_scan_bridge(
            bridge_inputs,
            bridge_lengths,
            target_token_lengths=target_lengths if compute_loss else None,
        )

        scan_loss = None
        if compute_loss:
            log_probs = self.experimental_scan_decoder.forward_teacher_forced(acoustic_tokens, decoder_inputs)
            scan_loss = self.experimental_scan_loss(log_probs=log_probs, labels=decoder_targets) + bridge_aux_loss

        predicted_tokens, predicted_lengths = None, None
        if compute_predictions:
            predicted_tokens, predicted_lengths = self.experimental_scan_decoder.greedy_decode(
                acoustic_tokens, acoustic_token_lengths
            )

        return {
            "scan_loss": scan_loss,
            "bridge_aux_loss": bridge_aux_loss,
            "predicted_tokens": predicted_tokens,
            "predicted_lengths": predicted_lengths,
            "token_lengths": acoustic_token_lengths,
        }

    @torch.no_grad()
    def transcribe(
        self,
        audio: List[str],
        batch_size: int = 4,
        return_hypotheses: bool = False,
        partial_hypothesis: Optional[List['Hypothesis']] = None,
        num_workers: int = 0,
        channel_selector: Optional[ChannelSelectorType] = None,
        augmentor: DictConfig = None,
        verbose: bool = True,
        timestamps: bool = None,
        override_config: Optional[TranscribeConfig] = None,
    ) -> TranscriptionReturnType:
        """
        Uses greedy decoding to transcribe audio files. Use this method for debugging and prototyping.

        Args:

            audio: (a single or list) of paths to audio files or a np.ndarray audio array.
                Can also be a dataloader object that provides values that can be consumed by the model.
                Recommended length per file is between 5 and 25 seconds. \
                But it is possible to pass a few hours long file if enough GPU memory is available.
            batch_size: (int) batch size to use during inference. \
                Bigger will result in better throughput performance but would use more memory.
            return_hypotheses: (bool) Either return hypotheses or text
                With hypotheses can do some postprocessing like getting timestamp or rescoring
            num_workers: (int) number of workers for DataLoader
            channel_selector (int | Iterable[int] | str): select a single channel or a subset of 
                channels from multi-channel audio. If set to `'average'`, it performs averaging across channels. 
                Disabled if set to `None`. Defaults to `None`. Uses zero-based indexing.
            augmentor: (DictConfig): Augment audio samples during transcription if augmentor is applied.
            timestamps: Optional(Bool): timestamps will be returned if set to True as part of hypothesis object 
                (output.timestep['segment']/output.timestep['word']). Refer to `Hypothesis` class for more details.
                Default is None and would retain the previous state set by using self.change_decoding_strategy().
            verbose: (bool) whether to display tqdm progress bar
            logprobs: (bool) whether to return ctc logits insted of hypotheses

        Returns:
            Returns a tuple of 2 items -
            * A list of greedy transcript texts / Hypothesis
            * An optional list of beam search transcript texts / Hypothesis / NBestHypothesis.
        """

        if timestamps is not None:
            if self.cur_decoder not in ["ctc", "rnnt"]:
                raise ValueError(
                    f"{self.cur_decoder} is not supported for cur_decoder. Supported values are ['ctc', 'rnnt']"
                )
            decoding_cfg = self.cfg.aux_ctc.decoding if self.cur_decoder == "ctc" else self.cfg.decoding
            need_change_decoding = False
            if timestamps or (override_config is not None and override_config.timestamps):
                logging.info(
                    "Timestamps requested, setting decoding timestamps to True. Capture them in Hypothesis object, \
                        with output[idx].timestep['word'/'segment'/'char']"
                )
                return_hypotheses = True
                if decoding_cfg.get("compute_timestamps", None) is not True:
                    # compute_timestamps None, False or non-existent -> change to True
                    need_change_decoding = True
                    with open_dict(decoding_cfg):
                        decoding_cfg.compute_timestamps = True
            else:
                if decoding_cfg.get("compute_timestamps", None) is not False:
                    # compute_timestamps None, True or non-existent -> change to False
                    need_change_decoding = True
                    with open_dict(decoding_cfg):
                        decoding_cfg.compute_timestamps = False
            if need_change_decoding:
                self.change_decoding_strategy(decoding_cfg, decoder_type=self.cur_decoder, verbose=False)

        return ASRTranscriptionMixin.transcribe(
            self,
            audio=audio,
            batch_size=batch_size,
            return_hypotheses=return_hypotheses,
            partial_hypothesis=partial_hypothesis,
            num_workers=num_workers,
            channel_selector=channel_selector,
            augmentor=augmentor,
            verbose=verbose,
            timestamps=timestamps,
            override_config=override_config,
        )

    def _transcribe_on_begin(self, audio, trcfg: TranscribeConfig):
        super()._transcribe_on_begin(audio, trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.freeze()

    def _transcribe_on_end(self, trcfg: TranscribeConfig):
        super()._transcribe_on_end(trcfg)

        if hasattr(self, 'ctc_decoder'):
            self.ctc_decoder.unfreeze(partial=True)

    def _transcribe_forward(self, batch: Any, trcfg: TranscribeConfig):
        if self.cur_decoder == "rnnt":
            encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])
            encoded = self._get_transducer_encoder_output(encoded, encoded_len=encoded_len)
            return dict(encoded=encoded, encoded_len=encoded_len)

        # CTC Path
        encoded, encoded_len = self.forward(input_signal=batch[0], input_signal_length=batch[1])

        logits = self.ctc_decoder(encoder_output=encoded)
        if hasattr(self.encoder, "clear_stage1_output"):
            self.encoder.clear_stage1_output()
        output = dict(logits=logits, encoded_len=encoded_len)

        del encoded
        return output

    def _transcribe_output_processing(
        self, outputs, trcfg: TranscribeConfig
    ) -> Union[List['Hypothesis'], List[List['Hypothesis']]]:
        if self.cur_decoder == "rnnt":
            return super()._transcribe_output_processing(outputs, trcfg)

        # CTC Path
        logits = outputs.pop('logits')
        encoded_len = outputs.pop('encoded_len')

        hypotheses = self.ctc_decoding.ctc_decoder_predictions_tensor(
            logits,
            encoded_len,
            return_hypotheses=trcfg.return_hypotheses,
        )
        logits = logits.cpu()

        if trcfg.return_hypotheses:
            # dump log probs per file
            for idx in range(logits.shape[0]):
                hypotheses[idx].y_sequence = logits[idx][: encoded_len[idx]]
                if hypotheses[idx].alignments is None:
                    hypotheses[idx].alignments = hypotheses[idx].y_sequence

        # DEPRECATED?
        # if logprobs:
        #     for logit, elen in zip(logits, encoded_len):
        #         logits_list.append(logit[:elen])

        if trcfg.timestamps:
            hypotheses = process_timestamp_outputs(
                hypotheses, self.encoder.subsampling_factor, self.cfg['preprocessor']['window_stride']
            )

        del logits, encoded_len

        return hypotheses

    def change_vocabulary(
        self,
        new_vocabulary: List[str],
        decoding_cfg: Optional[DictConfig] = None,
        ctc_decoding_cfg: Optional[DictConfig] = None,
    ):
        """
        Changes vocabulary used during RNNT decoding process. Use this method when fine-tuning a 
        pre-trained model. This method changes only decoder and leaves encoder and pre-processing 
        modules unchanged. For example, you would use it if you want to use pretrained encoder 
        when fine-tuning on data in another language, or when you'd need model to learn capitalization,
        punctuation and/or special characters.

        Args:
            new_vocabulary: list with new vocabulary. Must contain at least 2 elements. Typically, \
                this is target alphabet.
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            ctc_decoding_cfg: A config for CTC decoding, which is optional and can be used to change decoding type.

        Returns: None

        """
        super().change_vocabulary(new_vocabulary=new_vocabulary, decoding_cfg=decoding_cfg)

        # set up the new tokenizer for the CTC decoder
        if hasattr(self, 'ctc_decoder'):
            if self.ctc_decoder.vocabulary == new_vocabulary:
                logging.warning(
                    f"Old {self.ctc_decoder.vocabulary} and new {new_vocabulary} match. Not changing anything."
                )
            else:
                if new_vocabulary is None or len(new_vocabulary) == 0:
                    raise ValueError(f'New vocabulary must be non-empty list of chars. But I got: {new_vocabulary}')
                decoder_config = self.ctc_decoder.to_config_dict()
                new_decoder_config = copy.deepcopy(decoder_config)
                new_decoder_config['vocabulary'] = new_vocabulary
                new_decoder_config['num_classes'] = len(new_vocabulary)

                del self.ctc_decoder
                self.ctc_decoder = EncDecHybridRNNTCTCModel.from_config_dict(new_decoder_config)
                del self.ctc_loss
                self.ctc_loss = CTCLoss(
                    num_classes=self.ctc_decoder.num_classes_with_blank - 1,
                    zero_infinity=True,
                    reduction=self.cfg.aux_ctc.get("ctc_reduction", "mean_batch"),
                )

                if ctc_decoding_cfg is None:
                    # Assume same decoding config as before
                    logging.info("No `ctc_decoding_cfg` passed when changing decoding strategy, using internal config")
                    ctc_decoding_cfg = self.cfg.aux_ctc.decoding

                # Assert the decoding config with all hyper parameters
                ctc_decoding_cls = OmegaConf.structured(CTCDecodingConfig)
                ctc_decoding_cls = OmegaConf.create(OmegaConf.to_container(ctc_decoding_cls))
                ctc_decoding_cfg = OmegaConf.merge(ctc_decoding_cls, ctc_decoding_cfg)

                self.ctc_decoding = CTCDecoding(decoding_cfg=ctc_decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

                self.ctc_wer = WER(
                    decoding=self.ctc_decoding,
                    use_cer=self.ctc_wer.use_cer,
                    log_prediction=self.ctc_wer.log_prediction,
                    dist_sync_on_step=True,
                )

                # Update config
                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoding = ctc_decoding_cfg

                with open_dict(self.cfg.aux_ctc):
                    self.cfg.aux_ctc.decoder = new_decoder_config

                ds_keys = ['train_ds', 'validation_ds', 'test_ds']
                for key in ds_keys:
                    if key in self.cfg:
                        with open_dict(self.cfg[key]):
                            self.cfg[key]['labels'] = OmegaConf.create(new_vocabulary)

                logging.info(f"Changed the tokenizer of the CTC decoder to {self.ctc_decoder.vocabulary} vocabulary.")

    def change_decoding_strategy(
        self, decoding_cfg: DictConfig = None, decoder_type: str = None, verbose: bool = True
    ):
        """
        Changes decoding strategy used during RNNT decoding process.

        Args:
            decoding_cfg: A config for the decoder, which is optional. If the decoding type
                needs to be changed (from say Greedy to Beam decoding etc), the config can be passed here.
            decoder_type: (str) Can be set to 'rnnt' or 'ctc' to switch between appropriate decoder in a
                model having RNN-T and CTC decoders. Defaults to None, in which case RNN-T decoder is
                used. If set to 'ctc', it raises error if 'ctc_decoder' is not an attribute of the model.
            verbose: (bool) whether to display logging information
        """
        if decoder_type is None or decoder_type == 'rnnt':
            self.cur_decoder = "rnnt"
            return super().change_decoding_strategy(decoding_cfg=decoding_cfg, verbose=verbose)

        assert decoder_type == 'ctc' and hasattr(self, 'ctc_decoder')
        if decoding_cfg is None:
            # Assume same decoding config as before
            logging.info("No `decoding_cfg` passed when changing decoding strategy, using internal config")
            decoding_cfg = self.cfg.aux_ctc.decoding

        # Assert the decoding config with all hyper parameters
        decoding_cls = OmegaConf.structured(CTCDecodingConfig)
        decoding_cls = OmegaConf.create(OmegaConf.to_container(decoding_cls))
        decoding_cfg = OmegaConf.merge(decoding_cls, decoding_cfg)

        self.ctc_decoding = CTCDecoding(decoding_cfg=decoding_cfg, vocabulary=self.ctc_decoder.vocabulary)

        self.ctc_wer = WER(
            decoding=self.ctc_decoding,
            use_cer=self.ctc_wer.use_cer,
            log_prediction=self.ctc_wer.log_prediction,
            dist_sync_on_step=True,
        )

        self.ctc_decoder.temperature = decoding_cfg.get('temperature', 1.0)

        # Update config
        with open_dict(self.cfg.aux_ctc):
            self.cfg.aux_ctc.decoding = decoding_cfg

        self.cur_decoder = "ctc"
        if verbose:
            logging.info(f"Changed decoding strategy to \n{OmegaConf.to_yaml(self.cfg.aux_ctc.decoding)}")

        return None

    def _training_step_experimental_scan(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb
        compute_wer = (sample_id + 1) % log_every_n_steps == 0

        scan_outputs = self._run_experimental_scan_decoder(
            encoded=encoded,
            encoded_len=encoded_len,
            transcript=transcript,
            transcript_len=transcript_len,
            compute_loss=True,
            compute_predictions=compute_wer,
        )
        branch_loss = self.add_auxiliary_losses(scan_outputs["scan_loss"])

        tensorboard_logs = {
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            'train_scan_loss': branch_loss,
            'train_alignment_aux_loss': scan_outputs["bridge_aux_loss"],
        }

        if compute_wer:
            wer_num, wer_denom = self._compute_text_error_counts(
                scan_outputs["predicted_tokens"],
                scan_outputs["predicted_lengths"],
                transcript,
                transcript_len,
            )
            tensorboard_logs['training_batch_wer'] = wer_num / wer_denom.clamp_min(1.0)

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * branch_loss + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs['training_batch_wer_ctc'] = ctc_wer
        else:
            loss_value = branch_loss

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        self.log_dict(tensorboard_logs)
        return {'loss': loss_value}

    def _validation_pass_experimental_scan(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        scan_outputs = self._run_experimental_scan_decoder(
            encoded=encoded,
            encoded_len=encoded_len,
            transcript=transcript,
            transcript_len=transcript_len,
            compute_loss=self.compute_eval_loss,
            compute_predictions=True,
        )
        tensorboard_logs = {}
        loss_value = scan_outputs["scan_loss"]
        if loss_value is not None:
            tensorboard_logs['val_scan_loss'] = loss_value

        wer_num, wer_denom = self._compute_text_error_counts(
            scan_outputs["predicted_tokens"],
            scan_outputs["predicted_lengths"],
            transcript,
            transcript_len,
        )
        tensorboard_logs['val_wer_num'] = wer_num
        tensorboard_logs['val_wer_denom'] = wer_denom
        tensorboard_logs['val_wer'] = wer_num / wer_denom.clamp_min(1.0)

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_scan_aux_loss'] = scan_outputs["bridge_aux_loss"]
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value

        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)

        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def _predict_step_experimental_scan(self, batch, batch_idx, dataloader_idx=0):
        signal, signal_len, transcript, transcript_len, sample_id = batch

        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        scan_outputs = self._run_experimental_scan_decoder(
            encoded=encoded,
            encoded_len=encoded_len,
            transcript=transcript,
            transcript_len=transcript_len,
            compute_loss=False,
            compute_predictions=True,
        )

        hypotheses = []
        for prediction, pred_len in zip(
            scan_outputs["predicted_tokens"].detach().cpu().tolist(),
            scan_outputs["predicted_lengths"].detach().cpu().tolist(),
        ):
            token_ids = self._strip_special_tokens(prediction[:pred_len])
            hypotheses.append(self.tokenizer.ids_to_text(token_ids))

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, hypotheses))

    # PTL-specific methods
    def training_step(self, batch, batch_nb):
        if self._experimental_scan_enabled():
            return self._training_step_experimental_scan(batch, batch_nb)

        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal
        transducer_encoded = self._get_transducer_encoder_output(encoded, encoded_len=encoded_len)

        # During training, loss must be computed, so decoder forward is necessary
        decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)

        if hasattr(self, '_trainer') and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb

        if (sample_id + 1) % log_every_n_steps == 0:
            compute_wer = True
        else:
            compute_wer = False

        # If fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            # Compute full joint and loss
            joint = self.joint(encoder_outputs=transducer_encoded, decoder_outputs=decoder)
            loss_value = self.loss(
                log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                self.wer.update(
                    predictions=transducer_encoded,
                    predictions_lengths=encoded_len,
                    targets=transcript,
                    targets_lengths=transcript_len,
                )
                _, scores, words = self.wer.compute()
                self.wer.reset()
                tensorboard_logs.update({'training_batch_wer': scores.float() / words})

        else:  # If fused Joint-Loss-WER is used
            # Fused joint step
            loss_value, wer, _, _ = self.joint(
                encoder_outputs=transducer_encoded,
                decoder_outputs=decoder,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=transcript_len,
                compute_wer=compute_wer,
            )

            # Add auxiliary losses, if registered
            loss_value = self.add_auxiliary_losses(loss_value)

            tensorboard_logs = {
                'learning_rate': self._optimizer.param_groups[0]['lr'],
                'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
            }

            if compute_wer:
                tensorboard_logs.update({'training_batch_wer': wer})

        if self.transducer_fast_slow_fusion is not None:
            tensorboard_logs['fast_slow_fast_weight'] = self.transducer_fast_slow_fusion.fast_path_weight().detach()

        if self.ctc_loss_weight > 0:
            log_probs = self.ctc_decoder(encoder_output=encoded)
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['train_rnnt_loss'] = loss_value
            tensorboard_logs['train_ctc_loss'] = ctc_loss
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            if compute_wer:
                self.ctc_wer.update(
                    predictions=log_probs,
                    targets=transcript,
                    targets_lengths=transcript_len,
                    predictions_lengths=encoded_len,
                )
                ctc_wer, _, _ = self.ctc_wer.compute()
                self.ctc_wer.reset()
                tensorboard_logs.update({'training_batch_wer_ctc': ctc_wer})

        # note that we want to apply interctc independent of whether main ctc
        # loss is used or not (to allow rnnt + interctc training).
        # assuming ``ctc_loss_weight=0.3`` and interctc is applied to a single
        # layer with weight of ``0.1``, the total loss will be
        # ``loss = 0.9 * (0.3 * ctc_loss + 0.7 * rnnt_loss) + 0.1 * interctc_loss``
        loss_value, additional_logs = self.add_interctc_losses(
            loss_value, transcript, transcript_len, compute_wer=compute_wer
        )
        tensorboard_logs.update(additional_logs)
        tensorboard_logs['train_loss'] = loss_value
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        # Log items
        self.log_dict(tensorboard_logs)

        # Preserve batch acoustic model T and language model U parameters if normalizing
        if self._optim_normalize_joint_txu:
            self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]

        return {'loss': loss_value}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        if self._experimental_scan_enabled():
            return self._predict_step_experimental_scan(batch, batch_idx, dataloader_idx=dataloader_idx)

        signal, signal_len, transcript, transcript_len, sample_id = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        if self.cur_decoder == 'rnnt':
            transducer_encoded = self._get_transducer_encoder_output(encoded, encoded_len=encoded_len)
            best_hyp = self.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=transducer_encoded, encoded_lengths=encoded_len, return_hypotheses=True
            )
        else:
            logits = self.ctc_decoder(encoder_output=encoded)
            if hasattr(self.encoder, "clear_stage1_output"):
                self.encoder.clear_stage1_output()
            best_hyp = self.ctc_decoding.ctc_decoder_predictions_tensor(
                decoder_outputs=logits,
                decoder_lengths=encoded_len,
                return_hypotheses=True,
            )

        if isinstance(sample_id, torch.Tensor):
            sample_id = sample_id.cpu().detach().numpy()
        return list(zip(sample_id, best_hyp))

    def validation_pass(self, batch, batch_idx, dataloader_idx):
        if self._experimental_scan_enabled():
            return self._validation_pass_experimental_scan(batch, batch_idx, dataloader_idx)

        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch

        # forward() only performs encoder forward
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal
        transducer_encoded = self._get_transducer_encoder_output(encoded, encoded_len=encoded_len)

        tensorboard_logs = {}
        loss_value = None

        # If experimental fused Joint-Loss-WER is not used
        if not self.joint.fuse_loss_wer:
            if self.compute_eval_loss:
                decoder, target_length, states = self.decoder(targets=transcript, target_length=transcript_len)
                joint = self.joint(encoder_outputs=transducer_encoded, decoder_outputs=decoder)

                loss_value = self.loss(
                    log_probs=joint, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
                )
                tensorboard_logs['val_loss'] = loss_value

            self.wer.update(
                predictions=transducer_encoded,
                predictions_lengths=encoded_len,
                targets=transcript,
                targets_lengths=transcript_len,
            )
            wer, wer_num, wer_denom = self.wer.compute()
            self.wer.reset()

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        else:
            # If experimental fused Joint-Loss-WER is used
            compute_wer = True

            if self.compute_eval_loss:
                decoded, target_len, states = self.decoder(targets=transcript, target_length=transcript_len)
            else:
                decoded = None
                target_len = transcript_len

            # Fused joint step
            loss_value, wer, wer_num, wer_denom = self.joint(
                encoder_outputs=transducer_encoded,
                decoder_outputs=decoded,
                encoder_lengths=encoded_len,
                transcripts=transcript,
                transcript_lengths=target_len,
                compute_wer=compute_wer,
            )
            if loss_value is not None:
                tensorboard_logs['val_loss'] = loss_value

            tensorboard_logs['val_wer_num'] = wer_num
            tensorboard_logs['val_wer_denom'] = wer_denom
            tensorboard_logs['val_wer'] = wer

        log_probs = self.ctc_decoder(encoder_output=encoded)
        if self.compute_eval_loss:
            ctc_loss = self.ctc_loss(
                log_probs=log_probs, targets=transcript, input_lengths=encoded_len, target_lengths=transcript_len
            )
            tensorboard_logs['val_ctc_loss'] = ctc_loss
            tensorboard_logs['val_rnnt_loss'] = loss_value
            loss_value = (1 - self.ctc_loss_weight) * loss_value + self.ctc_loss_weight * ctc_loss
            tensorboard_logs['val_loss'] = loss_value
        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()
        tensorboard_logs['val_wer_num_ctc'] = ctc_wer_num
        tensorboard_logs['val_wer_denom_ctc'] = ctc_wer_denom
        tensorboard_logs['val_wer_ctc'] = ctc_wer

        self.log('global_step', torch.tensor(self.trainer.global_step, dtype=torch.float32))

        loss_value, additional_logs = self.add_interctc_losses(
            loss_value,
            transcript,
            transcript_len,
            compute_wer=True,
            compute_loss=self.compute_eval_loss,
            log_wer_num_denom=True,
            log_prefix="val_",
        )
        if self.compute_eval_loss:
            # overriding total loss value. Note that the previous
            # rnnt + ctc loss is available in metrics as "val_final_loss" now
            tensorboard_logs['val_loss'] = loss_value
        tensorboard_logs.update(additional_logs)
        # Reset access registry
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)

        return tensorboard_logs

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        tensorboard_logs = self.validation_pass(batch, batch_idx, dataloader_idx)
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(tensorboard_logs)
        else:
            self.validation_step_outputs.append(tensorboard_logs)

        return tensorboard_logs

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        logs = self.validation_pass(batch, batch_idx, dataloader_idx=dataloader_idx)
        test_logs = {name.replace("val_", "test_"): value for name, value in logs.items()}
        if type(self.trainer.test_dataloaders) == list and len(self.trainer.test_dataloaders) > 1:
            self.test_step_outputs[dataloader_idx].append(test_logs)
        else:
            self.test_step_outputs.append(test_logs)
        return test_logs

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        if not outputs or not all([isinstance(x, dict) for x in outputs]):
            logging.warning("No outputs to process for validation_epoch_end")
            return {}
        if self.compute_eval_loss:
            val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
            val_loss_log = {'val_loss': val_loss_mean}
        else:
            val_loss_log = {}
        wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**val_loss_log, 'val_wer': wer_num.float() / wer_denom}
        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['val_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['val_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['val_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom
        metrics = {**val_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="val_")
        return metrics

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        if self.compute_eval_loss:
            test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
            test_loss_log = {'test_loss': test_loss_mean}
        else:
            test_loss_log = {}
        wer_num = torch.stack([x['test_wer_num'] for x in outputs]).sum()
        wer_denom = torch.stack([x['test_wer_denom'] for x in outputs]).sum()
        tensorboard_logs = {**test_loss_log, 'test_wer': wer_num.float() / wer_denom}

        if self.ctc_loss_weight > 0:
            ctc_wer_num = torch.stack([x['test_wer_num_ctc'] for x in outputs]).sum()
            ctc_wer_denom = torch.stack([x['test_wer_denom_ctc'] for x in outputs]).sum()
            tensorboard_logs['test_wer_ctc'] = ctc_wer_num.float() / ctc_wer_denom

        metrics = {**test_loss_log, 'log': tensorboard_logs}
        self.finalize_interctc_metrics(metrics, outputs, prefix="test_")
        return metrics

    # EncDecRNNTModel is exported in 2 parts
    def list_export_subnets(self):
        if self.cur_decoder == 'rnnt':
            return ['encoder', 'decoder_joint']
        else:
            return ['self']

    @property
    def output_module(self):
        if self.cur_decoder == 'rnnt':
            return self.decoder
        else:
            return self.ctc_decoder

    @classmethod
    def list_available_models(cls) -> Optional[PretrainedModelInfo]:
        """
        This method returns a list of pre-trained model which can be instantiated directly from NVIDIA's NGC cloud.

        Returns:
            List of available pre-trained models.
        """
        results = []
        model = PretrainedModelInfo(
            pretrained_model_name="parakeet-tdt_ctc-110m",
            description="For details on this model, please refer to https://ngc.nvidia.com/catalog/models/nvidia:nemo:parakeet-tdt_ctc-110m",
            location="https://api.ngc.nvidia.com/v2/models/nvidia/nemo/parakeet-tdt_ctc-110m/versions/v1/files/parakeet-tdt_ctc-110m.nemo",
        )
        results.append(model)
        return results
