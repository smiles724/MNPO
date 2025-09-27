# MNPO Training Pipeline

This repository packages the full iterative **Multiplayer Nash Preference Optimization (MNPO)** workflow that we used to fine-tune instruction-following language models with on-policy preference data. It bundles scripts for dataset preparation, preference data generation, reward model annotation, and multi-GPU MNPO training so you can reproduce or adapt our alignment pipeline end-to-end.

## Key Features
- **End-to-end alignment loop** – Automates dataset splitting, precomputation, MNPO training, and optional on-policy data refreshes across multiple iterations.
- **Configurable infrastructure** – Includes ready-to-use Accelerate/DeepSpeed launch configs and per-iteration YAML training recipes targeting Gemma-2 instruction-tuned checkpoints.
- **On-policy preference generation** – Provides decoding, post-processing, and reward-model scoring utilities for creating MNPO-ready binary preference datasets.
- **Modular alignment utilities** – Reuses the shared `alignment` package for argument parsing, tokenizer handling, and adapter-aware checkpoint loading.

## Repository Layout

| Path | Description |
| --- | --- |
| `mnpo_scripts/` | MNPO orchestration code: configuration dataclasses, precomputation loop, MNPO trainer, and CLI entrypoints such as `run_mnpo.py` and `split_dataset.py`. |
| `on_policy_data_gen/` | Tools for generating and annotating on-policy preference pairs (decoding, post-processing, reward model annotation). |
| `alignment/` | Shared alignment helpers for data loading, model utilities, and release tooling. |
| `training_configs/` | MNPO hyperparameter YAMLs for each training stage (e.g., `gemma-2-9b-it-mnpo-iter*.yaml`). |
| `accelerate_configs/` | Launch configurations for Accelerate, DeepSpeed ZeRO, and FSDP setups. |
| `scripts/` | Auxiliary utilities and launch helpers. |
| `run.sh` | Example shell pipeline that ties together dataset splitting, on-policy data refresh, precomputation, and training loops. |

## Environment Setup
We separate environments for model training and large-scale decoding. Both assume **Python 3.10** and CUDA 12.8 builds of PyTorch/FlashAttention.

<details>
<summary><code>mnpo_train</code> (training & reward model inference)</summary>

```bash
conda create -n mnpo_train python=3.10 -y
conda activate mnpo_train
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  numpy==1.26.4 \
  accelerate==0.29.2 \
  deepspeed==0.15.4 \
  transformers==4.44.2 \
  trl==0.9.6 \
  datasets==2.18.0 \
  huggingface-hub==0.23.2 \
  peft==0.7.1 \
  wandb
```
</details>

<details>
<summary><code>mnpo_infer</code> (on-policy decoding)</summary>

```bash
conda create -n mnpo_infer python=3.10 -y
conda activate mnpo_infer
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install \
  vllm==0.9.0 \
  "transformers<4.54.0" \
  datasets==2.18.0 \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  numpy==1.26.4 \
  deepspeed==0.15.4 \
  https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl \
  more_itertools
```
</details>

Set `PYTHONPATH` to the repository root before running any module entrypoints:

```bash
export PYTHONPATH=$(pwd)
```

## End-to-End Workflow
The `run.sh` script demonstrates a three-iteration MNPO curriculum and can be adapted to your infrastructure.

1. **Initial dataset split** – `mnpo_scripts.split_dataset` shards the base preference dataset into per-iteration train/test JSONL files to avoid leakage between stages.
2. **On-policy generation (for iteration &gt; 1)** – `on_policy_data_gen.decode` samples multiple responses per prompt, `post_process` filters identical answers, and `reward_model_annotate` scores them with a reward model to produce MNPO-ready pairs. reward_model_annotate scores them with a reward model to produce MNPO-ready pairs. For reproduction and experiments, we included both reward model and preference model annotation.
3. **Precomputation** – `mnpo_scripts.precompute` computes log-probabilities, normalizers, and history buffers used by MNPO training. Previous stage checkpoints can be chained via the `--history_paths` argument.
4. **Training** – `mnpo_scripts.run_mnpo` launches the actual MNPO updates using Accelerate/DeepSpeed and the YAML config for the current iteration. Outputs are written under `outputs/` and fed into the next iteration.

Adjust the variables at the top of `run.sh` (base model, dataset name, GPU count, cache directories) to reflect your setup, then execute:

```bash
bash run.sh
```

## Support & Citation
If you build on this codebase in academic work, please cite the MNPO methodology and link back to this repository so others can reproduce your setup.
