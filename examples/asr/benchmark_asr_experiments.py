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

"""
Compare ASR experiments using TensorBoard event logs and optional offline manifest evaluation.

The script reads one JSON spec file describing a set of experiments. For each experiment it can:
1. Parse TensorBoard scalars from a NeMo experiment directory.
2. Load a `.nemo` model and benchmark offline transcription on a manifest.
3. Report bucketed WER by utterance duration, inference speed, and peak GPU memory.

Example experiment spec:

[
  {
    "label": "baseline",
    "run_dir": "/workspace/experiments/clean460_fc_hybrid_tdt_ctc_32m_v256_bs64_20k/2026-04-01_10-00-00",
    "model_path": "/workspace/experiments/clean460_fc_hybrid_tdt_ctc_32m_v256_bs64_20k/2026-04-01_10-00-00/checkpoints/clean460_fc_hybrid_tdt_ctc_32m_v256_bs64_20k.nemo",
    "decoder_type": "rnnt"
  },
  {
    "label": "gatedfc_v2",
    "run_dir": "/workspace/experiments/clean460_gatedfc_hybrid_tdt_ctc_32m_v2_v256_bs64_20k/2026-04-01_12-00-00",
    "decoder_type": "rnnt"
  }
]

Usage:

python examples/asr/benchmark_asr_experiments.py \
  --experiments-json /workspace/benchmarks/hybrid_compare.json \
  --manifest /workspace/data/librispeech_hf_clean/manifests/test.jsonl \
  --batch-size 32 \
  --decoder-type rnnt \
  --output-json /workspace/benchmarks/hybrid_compare_results.json \
  --output-csv /workspace/benchmarks/hybrid_compare_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from nemo.collections.asr.metrics.wer import word_error_rate
from nemo.collections.asr.models import ASRModel

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError as exc:  # pragma: no cover - depends on runtime env
    EventAccumulator = None
    _TB_IMPORT_ERROR = exc
else:
    _TB_IMPORT_ERROR = None


DECREASING_TAG_MARKERS = (
    "wer",
    "loss",
    "timing",
)

SUMMARY_TAGS = (
    "val_wer",
    "val_wer_ctc",
    "val_loss",
    "val_ctc_loss",
    "val_rnnt_loss",
    "test_wer",
    "test_wer_ctc",
    "train_loss",
    "training_batch_wer",
    "training_batch_wer_ctc",
    "train_step_timing in s",
    "validation_step_timing in s",
    "test_step_timing in s",
    "train_backward_timing in s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare NeMo ASR experiment runs.")
    parser.add_argument("--experiments-json", required=True, help="JSON file with experiment definitions.")
    parser.add_argument("--manifest", help="ASR manifest to benchmark with offline transcription.")
    parser.add_argument("--output-json", help="Optional JSON output path.")
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size for manifest benchmarking.")
    parser.add_argument("--last-n", type=int, default=10, help="Last N scalar values to average for timing metrics.")
    parser.add_argument(
        "--decoder-type",
        choices=["rnnt", "ctc"],
        default=None,
        help="Default decoder for hybrid models when per-experiment decoder_type is omitted.",
    )
    parser.add_argument(
        "--duration-buckets",
        type=float,
        nargs="*",
        default=[5.0, 10.0],
        help="Duration bucket boundaries in seconds. Default: 5 10 -> <=5, 5-10, >10.",
    )
    parser.add_argument("--cuda", type=int, default=None, help="CUDA device id. Use -1 for CPU.")
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="none",
        help="Optional autocast dtype during manifest benchmarking.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional cap on number of manifest samples to benchmark per experiment.",
    )
    return parser.parse_args()


def sanitize_tag(tag: str) -> str:
    cleaned = []
    for ch in tag.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    while "__" in "".join(cleaned):
        cleaned = list("".join(cleaned).replace("__", "_"))
    return "".join(cleaned).strip("_")


def load_experiment_specs(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Experiment spec JSON must be a list of experiment objects.")
    for idx, exp in enumerate(data):
        if "label" not in exp:
            raise ValueError(f"Experiment index {idx} is missing required key 'label'.")
    return data


def find_event_files(run_dir: str | None) -> list[Path]:
    if not run_dir:
        return []
    root = Path(run_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("events.out.tfevents*"))


def dedupe_scalar_events(events: list[Any]) -> list[Any]:
    deduped = {}
    for event in events:
        deduped[event.step] = event
    return [deduped[step] for step in sorted(deduped)]


def summarize_tensorboard(run_dir: str | None, last_n: int) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    event_files = find_event_files(run_dir)
    if not event_files:
        summary["tb_status"] = "missing"
        return summary
    if EventAccumulator is None:
        raise RuntimeError(
            "tensorboard is required to parse event logs. "
            f"Import failed with: {_TB_IMPORT_ERROR}"
        )

    tags_to_events: dict[str, list[Any]] = defaultdict(list)
    min_wall_time = None
    max_wall_time = None
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            events = accumulator.Scalars(tag)
            if not events:
                continue
            tags_to_events[tag].extend(events)
            min_wall_time = events[0].wall_time if min_wall_time is None else min(min_wall_time, events[0].wall_time)
            max_wall_time = events[-1].wall_time if max_wall_time is None else max(max_wall_time, events[-1].wall_time)

    summary["tb_status"] = "ok"
    summary["tb_event_files"] = len(event_files)
    if min_wall_time is not None and max_wall_time is not None:
        summary["tb_wall_hours"] = (max_wall_time - min_wall_time) / 3600.0

    for tag, raw_events in tags_to_events.items():
        events = dedupe_scalar_events(raw_events)
        values = [event.value for event in events]
        if not values:
            continue

        key = sanitize_tag(tag)
        summary[f"tb_{key}_final"] = float(values[-1])
        summary[f"tb_{key}_step"] = int(events[-1].step)

        if tag in SUMMARY_TAGS or any(marker in tag.lower() for marker in DECREASING_TAG_MARKERS):
            summary[f"tb_{key}_best"] = float(min(values))

        if "timing" in tag.lower():
            tail = values[-last_n:] if len(values) >= last_n else values
            summary[f"tb_{key}_avg_last_{last_n}"] = float(sum(tail) / len(tail))

    return summary


def resolve_model_path(exp: dict[str, Any]) -> str | None:
    model_path = exp.get("model_path")
    if model_path:
        return model_path

    run_dir = exp.get("run_dir")
    if not run_dir:
        return None

    checkpoints = sorted(Path(run_dir).rglob("*.nemo"))
    if checkpoints:
        return str(checkpoints[-1])

    return None


def get_device(cuda_idx: int | None) -> torch.device:
    if cuda_idx is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    if cuda_idx < 0:
        return torch.device("cpu")
    return torch.device(f"cuda:{cuda_idx}")


def load_manifest(manifest_path: str, limit_samples: int | None = None) -> list[dict[str, Any]]:
    entries = []
    with open(manifest_path, "r") as f:
        for idx, line in enumerate(f):
            if limit_samples is not None and idx >= limit_samples:
                break
            item = json.loads(line)
            if "audio_filepath" not in item or "text" not in item:
                raise ValueError("Manifest entries must contain 'audio_filepath' and 'text'.")
            entries.append(item)
    return entries


def bucket_name(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        return "all"
    if lower is None:
        return f"le_{str(upper).replace('.', '_')}s"
    if upper is None:
        return f"gt_{str(lower).replace('.', '_')}s"
    return f"{str(lower).replace('.', '_')}s_to_{str(upper).replace('.', '_')}s"


def compute_bucket_indices(durations: list[float], boundaries: list[float]) -> dict[str, list[int]]:
    sorted_bounds = sorted(boundaries)
    buckets: dict[str, list[int]] = {}
    previous = None
    for upper in sorted_bounds:
        name = bucket_name(previous, upper)
        buckets[name] = [idx for idx, dur in enumerate(durations) if (previous is None or dur > previous) and dur <= upper]
        previous = upper
    buckets[bucket_name(previous, None)] = [idx for idx, dur in enumerate(durations) if dur > previous]
    return buckets


def normalize_hypotheses(hypotheses: list[Any]) -> list[str]:
    normalized = []
    for hyp in hypotheses:
        if isinstance(hyp, str):
            normalized.append(hyp)
        elif hasattr(hyp, "text"):
            normalized.append(hyp.text)
        else:
            normalized.append(str(hyp))
    return normalized


def maybe_switch_decoder(model: ASRModel, decoder_type: str | None) -> None:
    if decoder_type is None:
        return
    if hasattr(model, "change_decoding_strategy"):
        model.change_decoding_strategy(decoding_cfg=None, decoder_type=decoder_type, verbose=False)


def get_autocast_dtype(name: str) -> torch.dtype | None:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return None


def benchmark_model(
    exp: dict[str, Any],
    manifest_path: str,
    batch_size: int,
    boundaries: list[float],
    device: torch.device,
    amp_dtype_name: str,
    limit_samples: int | None,
    default_decoder_type: str | None,
) -> dict[str, Any]:
    model_path = resolve_model_path(exp)
    if not model_path:
        return {"eval_status": "skipped_no_model"}

    entries = load_manifest(manifest_path, limit_samples=limit_samples)
    paths = [entry["audio_filepath"] for entry in entries]
    refs = [entry["text"] for entry in entries]
    durs = [float(entry.get("duration", 0.0)) for entry in entries]
    total_audio_sec = float(sum(durs))

    result: dict[str, Any] = {
        "eval_status": "ok",
        "eval_model_path": model_path,
        "eval_samples": len(entries),
        "eval_total_audio_sec": total_audio_sec,
    }

    model = ASRModel.restore_from(model_path, map_location=device)
    model.eval()
    model.to(device)

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    result["model_params_total"] = int(total_params)
    result["model_params_trainable"] = int(trainable_params)

    decoder_type = exp.get("decoder_type", default_decoder_type)
    if decoder_type is not None:
        maybe_switch_decoder(model, decoder_type)
        result["eval_decoder_type"] = decoder_type

    autocast_dtype = get_autocast_dtype(amp_dtype_name)
    use_autocast = autocast_dtype is not None and device.type == "cuda"

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    predictions: list[str] = []
    start = time.perf_counter()
    with torch.inference_mode():
        context = torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True) if use_autocast else nullcontext()
        with context:
            for start_idx in range(0, len(paths), batch_size):
                batch_paths = paths[start_idx : start_idx + batch_size]
                batch_hypotheses = model.transcribe(audio=batch_paths, batch_size=min(batch_size, len(batch_paths)))
                predictions.extend(normalize_hypotheses(batch_hypotheses))
    elapsed = time.perf_counter() - start

    result["eval_wall_sec"] = elapsed
    result["eval_rtfx"] = total_audio_sec / elapsed if elapsed > 0 else None
    result["eval_samples_per_sec"] = len(entries) / elapsed if elapsed > 0 else None
    if device.type == "cuda":
        result["eval_peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated(device) / (1024**2)

    result["eval_wer"] = word_error_rate(predictions, refs, use_cer=False)
    result["eval_cer"] = word_error_rate(predictions, refs, use_cer=True)

    for bucket, indices in compute_bucket_indices(durs, boundaries).items():
        if not indices:
            result[f"eval_wer_{bucket}"] = None
            continue
        bucket_hyps = [predictions[idx] for idx in indices]
        bucket_refs = [refs[idx] for idx in indices]
        result[f"eval_wer_{bucket}"] = word_error_rate(bucket_hyps, bucket_refs, use_cer=False)
        result[f"eval_samples_{bucket}"] = len(indices)

    return result


def write_json(path: str, rows: list[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, sort_keys=True)


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(rows: list[dict[str, Any]], last_n: int) -> None:
    columns = [
        "label",
        "tb_val_wer_best",
        "tb_val_wer_ctc_best",
        "tb_test_wer_final",
        "tb_test_wer_ctc_final",
        f"tb_train_step_timing_in_s_avg_last_{last_n}",
        f"tb_validation_step_timing_in_s_avg_last_{last_n}",
        "eval_decoder_type",
        "eval_wer",
        "eval_rtfx",
        "eval_peak_gpu_mem_mb",
    ]

    available = [column for column in columns if any(column in row for row in rows)]
    if not available:
        return

    print("\t".join(available))
    for row in rows:
        values = []
        for column in available:
            value = row.get(column)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append("" if value is None else str(value))
        print("\t".join(values))


def main() -> None:
    args = parse_args()
    experiments = load_experiment_specs(args.experiments_json)
    device = get_device(args.cuda)

    rows: list[dict[str, Any]] = []
    for exp in experiments:
        row: dict[str, Any] = {"label": exp["label"]}
        row.update(summarize_tensorboard(exp.get("run_dir"), last_n=args.last_n))
        if args.manifest:
            row.update(
                benchmark_model(
                    exp=exp,
                    manifest_path=args.manifest,
                    batch_size=exp.get("batch_size", args.batch_size),
                    boundaries=args.duration_buckets,
                    device=device,
                    amp_dtype_name=args.amp_dtype,
                    limit_samples=args.limit_samples,
                    default_decoder_type=args.decoder_type,
                )
            )
        rows.append(row)

    if args.output_json:
        write_json(args.output_json, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)

    print_summary(rows, last_n=args.last_n)


if __name__ == "__main__":
    main()
