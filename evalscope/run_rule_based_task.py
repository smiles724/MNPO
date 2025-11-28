# run_minerva_task.py
import os
import argparse
from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    task_cfg = TaskConfig(
        model=args.model_name,
        api_url=f"http://127.0.0.1:{args.port}/v1",
        api_key="EMPTY",
        eval_type=EvalType.SERVICE,
        datasets=['ifeval'],
        eval_batch_size=20,
    )

    run_task(task_cfg=task_cfg)


if __name__ == "__main__":
    main()
