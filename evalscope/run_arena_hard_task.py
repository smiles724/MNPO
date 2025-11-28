import os
import argparse
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")

    task_cfg = TaskConfig(
        model=args.model_name,
        generation_config={"max_tokens": 4096},

        api_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="EMPTY",

        eval_type=EvalType.SERVICE,
        datasets=['arena_hard'],
        eval_batch_size=12,
        judge_worker_num=18,
        judge_strategy=JudgeStrategy.AUTO,

        judge_model_args={
            'model_id': 'gpt-5-mini',
            'generation_config': {"reasoning_effort": "minimal"},
            'api_url': api_url,
            'api_key': api_key,
        },
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
