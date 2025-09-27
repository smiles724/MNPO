# Environment

sim
```shell
conda create -n sim python=3.10 -y
conda activate sim
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
  numpy==1.26.4 \
  accelerate==0.29.2 \
  deepspeed==0.15.4 \
  transformers==4.44.2 \
  trl==0.9.6 \
  datasets==2.18.0 \
  huggingface-hub==0.23.2 \
  peft==0.7.1 \
  wandb \


```
mnpo

```shell
conda create -n mnpo python=3.10 -y
conda activate mnpo
pip install \
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
  vllm==0.9.0 \
  pip install "transformers<4.54.0"
  datasets==2.18.0 \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl  
  numpy==1.26.4 \
  deepspeed==0.15.4 \
  pip install https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.5%2Bcu128torch2.7-cp38-abi3-linux_x86_64.whl \
  more_itertools

```