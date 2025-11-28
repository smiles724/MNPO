# MNPO Training Pipeline

🔔 News

- (2025-11) The codebase has been updated to include full support for HT-MNPO 

This repository packages the full iterative **Multiplayer Nash Preference Optimization (MNPO)** workflow that we used to fine-tune instruction-following language models with on-policy preference data. It bundles scripts for dataset preparation, preference data generation, annotation, and multi-GPU MNPO training so you can reproduce or adapt our alignment pipeline end-to-end.

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
2. **On-policy generation (for iteration &gt; 1)** – `on_policy_data_gen.decode` samples multiple responses per prompt, `post_process` filters identical answers, and `reward_model_annotate` scores them with a reward model to produce MNPO-ready pairs. For reproduction and experiments, we included both reward model and preference model annotation.
3. **Precomputation** – `mnpo_scripts.precompute` computes log-probabilities, normalizers, and history buffers used by MNPO training. Previous stage checkpoints can be chained via the `--history_paths` argument.
4. **Training** – `mnpo_scripts.run_mnpo` launches the actual MNPO updates using Accelerate/DeepSpeed and the YAML config for the current iteration. Outputs are written under `outputs/` and fed into the next iteration.

Adjust the variables at the top of run.sh, customize the training and accelerate configurations to match your setup, then execute:
```bash
bash run_iter1/2/3.sh
```

`run_iter1/2/3.sh` demonstrates a pipeline using a single reward model. Other reward models can be substituted by modifying the corresponding reward-model-annotation section.


## Evaluation

We adopt [EvalScope](https://github.com/modelscope/evalscope/tree/main) for a unified evaluation pipeline to save time and ensure reproducibility.

### Installation

Follow the official GitHub instructions to set up EvalScope:

```bash
conda create -n evalscope python=3.10
conda activate evalscope
git clone https://github.com/modelscope/evalscope.git
cd evalscope/
pip install -e .
```

### Serving the Model with vLLM

Before evaluation, first serve your model via `vllm`:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-9b-it \
    --served-model-name google/gemma-2-9b-it \
    --trust_remote_code \
    --port 8801 \
    --tensor-parallel-size 8 # num of gpu
```

### Evaluating Rule-Based Datasets

EvalScope provides CLI support for multiple datasets at once. For example:

```bash
evalscope eval \
    --model gemma-2-9b-it \
    --api-url http://127.0.0.1:8801/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --limit 5 \ 
    --datasets humaneval arc  # add more datasets here
```

### Evaluating Datasets with LLM Judges

Some benchmarks require an LLM judge for evaluation. Here is an example script:

```python
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='gemma-2-9b-it',
    api_url='http://127.0.0.1:8801/v1',
    api_key='EMPTY',
    eval_type=EvalType.SERVICE,
    datasets=[
        'alpaca_eval',
    ],
    eval_batch_size=12,
    judge_worker_num=12,
    limit=5,  # optional for debugging
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'gpt-5-mini',
        'generation_config': {"reasoning_effort": "minimal"},
        'api_url': 'xx',
        'api_key': 'xx',
    },
)

run_task(task_cfg=task_cfg)
```

> 📌 **Notes**
>
> * The version of EvalScope used in the paper is **1.0.2**.
> * The LLM judge is `gpt-5-mini` (Aug 7, 2025), with `reasoning_effort="minimal"`.
> * All other judge parameters follow EvalScope defaults.

You can find the list of datasets supported by EvalScope at [the official documentation](https://evalscope.readthedocs.io/en/latest/get_started/supported_dataset/llm.html).

## MT-Bench Evaluation

For MT-Bench, we follow the [official FastChat repository](https://github.com/lm-sys/FastChat).

**Step 1: Generate model outputs**

```bash
export OPENAI_BASE_URL="http://127.0.0.1:8801/v1"
export OPENAI_API_KEY="EMPTY"

python -m fastchat.llm_judge.gen_api_answer \
    --model gemma-2-9b-it \
    --bench-name mt_bench \
    --parallel 12
```

**Step 2: Judge with an external model**

```bash
export OPENAI_API_KEY="xx"
export OPENAI_BASE_URL="xx"

python gen_judgment.py \
    --model-list gemma-2-9b-it \
    --parallel 12 \
    --judge-model gpt-5-mini
```

## Support & Citation
If you build on this codebase in academic work, please cite the MNPO methodology and link back to this repository so others can reproduce your setup.
