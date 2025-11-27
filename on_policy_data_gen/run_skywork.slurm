#!/bin/bash
#SBATCH --account=yejin
#SBATCH -p yejin
#SBATCH --gres=gpu:2
#SBATCH --job-name=skywork
#SBATCH --output=output_skywork.log

export HF_HOME=/hai/scratch/fangwu97/hf_cache

#/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.reward_model_annotate_skywork \
#--generation_file /hai/scratch/fangwu97/xu/SimPO_slurm/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split1.jsonl \
#--output_dir /hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/gemma2_ufb_part1_split1_skywork \
#--cache_dir "/hai/scratch/fangwu97/xu/cache"

/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.rm_skywork_batch