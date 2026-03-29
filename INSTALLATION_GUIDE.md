# Installation Guide (Vast.ai + NeMo ASR)

This guide captures the exact setup and fixes used to run:

- `examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py`
- with Hugging Face `openslr/librispeech_asr` (`clean` subset)
- on a single RTX 3090 (24 GB)

It is based on the errors encountered during setup and the working final command flow.

## 1) Recommended VM / Template

- Provider: Vast.ai
- GPU: RTX 3090 24 GB (or better)
- OS: Ubuntu 22.04
- CUDA: 12.1/12.2 compatible stack
- Disk: 150+ GB (HF cache + logs + checkpoints)

## 2) Environment Setup

From repo root (`/workspace/NeMo`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

Install matching PyTorch binaries (avoid `torchvision::nms` mismatch):

```bash
pip uninstall -y torchvision torchaudio torch
pip install --no-cache-dir \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install NeMo from source:

```bash
cd /workspace/NeMo
pip install -e ".[all,cu12]"
```

Install tokenizer dependencies explicitly:

```bash
pip install tokenizers sentencepiece
```

Install audio libs:

```bash
sudo apt update
sudo apt install -y ffmpeg libsndfile1 sox
pip install -U soundfile librosa
```

## 3) Export Text for Tokenizer

Use non-decoding mode for HF audio column during text export:

```bash
python - <<'PY'
from datasets import load_dataset, Audio

splits = ["train.100", "train.360"]
out = "/workspace/librispeech_clean_train_text.txt"

with open(out, "w", encoding="utf-8") as f:
    for sp in splits:
        ds = load_dataset("openslr/librispeech_asr", "clean", split=sp)
        ds = ds.cast_column("audio", Audio(decode=False))
        for ex in ds:
            f.write(ex["text"].strip() + "\n")

print("Wrote", out)
PY
```

## 4) Build Tokenizer

```bash
cd /workspace/NeMo
python scripts/tokenizers/process_asr_text_tokenizer.py \
  --data_file="/workspace/librispeech_clean_train_text.txt" \
  --data_root="/workspace/tokenizers/libri_clean_bpe_1024" \
  --vocab_size=1024 \
  --tokenizer="spe" \
  --spe_type="unigram" \
  --spe_character_coverage=1.0 \
  --no_lower_case
```

Tokenizer path to use later:

- `/workspace/tokenizers/libri_clean_bpe_1024/tokenizer_spe_unigram_v1024`

## 5) HF Dataset Compatibility Fix

If training crashes with:

- `ImportError: To support decoding audio data, please install 'torchcodec'`

Pin `datasets` to a compatible version and avoid torchcodec dependency path:

```bash
pip uninstall -y torchcodec datasets
pip install "datasets==2.21.0"
```

## 6) Working Training Command (Hydra-safe overrides)

Use `++` for keys that do not exist in base YAML.
Do not pass JSON strings like `'[{"path":...}]'` directly to Hydra.

```bash
python examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py \
  --config-path="../conf/fastconformer/hybrid_transducer_ctc" \
  --config-name="fastconformer_hybrid_tdt_ctc_bpe.yaml" \
  model.tokenizer.dir="/workspace/tokenizers/libri_clean_bpe_1024/tokenizer_spe_unigram_v1024" \
  model.tokenizer.type="bpe" \
  model.train_ds.manifest_filepath="huggingface" \
  model.validation_ds.manifest_filepath="huggingface" \
  ++model.train_ds.hf_data_cfg.path="openslr/librispeech_asr" \
  ++model.train_ds.hf_data_cfg.name="clean" \
  ++model.train_ds.hf_data_cfg.split="train.100" \
  ++model.validation_ds.hf_data_cfg.path="openslr/librispeech_asr" \
  ++model.validation_ds.hf_data_cfg.name="clean" \
  ++model.validation_ds.hf_data_cfg.split="validation" \
  ++model.train_ds.audio_key="audio.array" \
  ++model.train_ds.sample_rate_key="audio.sampling_rate" \
  ++model.train_ds.text_key="text" \
  ++model.validation_ds.audio_key="audio.array" \
  ++model.validation_ds.sample_rate_key="audio.sampling_rate" \
  ++model.validation_ds.text_key="text" \
  trainer.devices=1 \
  trainer.accelerator="gpu" \
  trainer.precision="16-mixed" \
  model.train_ds.batch_size=12 \
  model.validation_ds.batch_size=12 \
  trainer.accumulate_grad_batches=2 \
  model.train_ds.max_duration=20 \
  trainer.max_steps=2000 \
  trainer.max_epochs=null \
  trainer.log_every_n_steps=10 \
  exp_manager.name="speed_probe_fc_tdt_ctc_3090_b12_acc2_d20"
```

## 7) Common Errors and Exact Fixes

### Error: `No module named tokenizers`
```bash
pip install tokenizers sentencepiece
```

### Error: `No module named nemo`
```bash
cd /workspace/NeMo
pip install -e ".[all,cu12]"
```

### Error: `operator torchvision::nms does not exist`
Torch / torchvision mismatch. Reinstall matched versions:
```bash
pip uninstall -y torchvision torchaudio torch
pip install --no-cache-dir \
  torch==2.5.1+cu121 \
  torchvision==0.20.1+cu121 \
  torchaudio==2.5.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Error: Hydra parse failure `no viable alternative at input '[{"path"'`
Do not pass JSON list/object strings. Use Hydra field overrides with `++`.

### Error: `Could not delete from config ... hf_data_cfg does not exist`
Do not use `~model.train_ds.hf_data_cfg` when the key is absent.
Use `++model.train_ds.hf_data_cfg.*=...` instead.

### Warning: `trust_remote_code is not supported anymore`
Remove `trust_remote_code` keys from HF config overrides.

## 8) Terminating Stuck Runs

```bash
pkill -f "speech_to_text_hybrid_rnnt_ctc_bpe.py"
```

If still running:

```bash
nvidia-smi
kill -9 <PID>
```

## 9) Notes on Metrics During Early Scratch Training

- Very early `training_batch_wer` can stay near `~1.0` and still be normal.
- For meaningful comparison use:
  - `val_wer`
  - final `test_wer`
- Short probe runs are mainly for speed/memory profiling.
