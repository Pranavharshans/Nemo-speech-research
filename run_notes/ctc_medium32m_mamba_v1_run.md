# Medium 32M CTC Mamba V1 Run

This file captures the recommended launch command for the Medium (32M)
FastConformer-CTC Mamba V1 experiment that mirrors the working baseline recipe.

## Architecture

- Base config: `examples/asr/conf/fastconformer/fast-conformer_ctc_bpe.yaml`
- Experimental config: `examples/asr/conf/fastconformer/fast-conformer_ctc_bpe_mamba_v1_medium32m.yaml`
- Encoder target: `nemo.collections.asr.modules.MambaConformerEncoder`
- Medium dimensions: `d_model=256`, `n_heads=4`, `n_layers=16`
- Mamba swap: layers `[6, 7, 8, 9]`

## Data and Tokenizer

- Train manifest: `train_460.jsonl`
- Validation manifest: `validation.jsonl`
- Test manifest: `test.jsonl`
- Tokenizer: `tokenizer_spe_unigram_v256`

## Recommended Run

```bash
cd /workspace/nemo-mamba-v1/examples/asr/asr_ctc

export CUDA_HOME=/workspace/venv312/lib/python3.12/site-packages/nvidia/cuda_nvcc
export LD_LIBRARY_PATH=/workspace/venv312/lib/python3.12/site-packages/nvidia/cuda_nvcc/nvvm/lib64:$LD_LIBRARY_PATH
export NUMBA_FORCE_CUDA_CC=8.9

python speech_to_text_ctc_bpe.py \
  --config-path=../conf/fastconformer \
  --config-name=fast-conformer_ctc_bpe_mamba_v1_medium32m \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.strategy=auto \
  trainer.sync_batchnorm=False \
  trainer.max_steps=20000 \
  trainer.max_epochs=1000 \
  trainer.log_every_n_steps=10 \
  trainer.val_check_interval=250 \
  model.train_ds.manifest_filepath=/workspace/data/librispeech_hf_clean/manifests/train_460.jsonl \
  model.validation_ds.manifest_filepath=/workspace/data/librispeech_hf_clean/manifests/validation.jsonl \
  model.test_ds.manifest_filepath=/workspace/data/librispeech_hf_clean/manifests/test.jsonl \
  model.tokenizer.dir=/workspace/data/librispeech_hf_clean/tokenizer/tokenizer_spe_unigram_v256 \
  model.tokenizer.type=bpe \
  model.train_ds.batch_size=64 \
  model.validation_ds.batch_size=64 \
  model.test_ds.batch_size=64 \
  model.train_ds.max_duration=20 \
  trainer.precision=bf16-mixed \
  model.optim.sched.warmup_steps=4000 \
  exp_manager.create_wandb_logger=False \
  exp_manager.name=librispeech_clean460_fastconformer_ctc_mamba_v1_medium32m_v256_bs64_20k \
  exp_manager.exp_dir=/workspace/experiments
```

## Comparison Rule

Keep the following identical between baseline and Mamba V1:

- manifests
- tokenizer
- batch size
- max duration
- precision
- warmup
- max steps
- validation interval

Only the encoder implementation should differ.
