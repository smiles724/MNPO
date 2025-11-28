from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy
import os

# python evalscope/api_eval.py

api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL")

task_cfg = TaskConfig(
    model='allenai/olmo-2-0325-32b-instruct',
    generation_config={"max_tokens": 4096},
    api_url='xxx',
    api_key='xxx',
    eval_type=EvalType.SERVICE,
    datasets=[
        'arena_hard'
    ],
    eval_batch_size=12,
    judge_worker_num=12,
    # limit=5,
    judge_strategy=JudgeStrategy.AUTO,
    judge_model_args={
        'model_id': 'gpt-5-mini',
        'generation_config': {"reasoning_effort": "minimal"},
        'api_url': api_url,
        'api_key': api_key,
    },
)


run_task(task_cfg=task_cfg)