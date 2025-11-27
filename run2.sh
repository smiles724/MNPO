
set -e

SCRATCH_ROOT=/hai/scratch/fangwu97
PROJECT_ROOT=$SCRATCH_ROOT/xu
SIMPO_ROOT=$PROJECT_ROOT/SimPO_slurm
MNPO_ROOT=$PROJECT_ROOT/MNPO
CACHE_DIR=$PROJECT_ROOT/cache

# === 实验相关（经常改的） ===
POLICY_MODEL=$MNPO_ROOT/outputs/gemma-2-9b-it_mnpo_stage_2_armo_puredpoloss_beta5
POLICY_MODEL_T_2=$MNPO_ROOT/outputs/gemma-2-9b-it_mnpo_stage_1_armo_inpo_iter1_20k
REF_MODEL=$CACHE_DIR/google/gemma-2-9b-it                                      # ref model 路径

DATA_SPLIT=$SIMPO_ROOT/data/gemma2_ufb_part1_20k/gemma2_ufb_part1_split3.jsonl
GEN_DIR=$SIMPO_ROOT/datasets/gemma2_ultrafeedback/mnpo_iter3_armo_dpo_abl         # decode & post_process 输出目录
SCORED_FILE=${GEN_DIR}_scored.jsonl                                            # armo 打分输出 + precompute 输入
PREF_DIR=$SIMPO_ROOT/data/mnpo_iter3_armo_dpo_abl/pref # precompute 输出目录

export HF_HOME=$SCRATCH_ROOT/hf_cache

# data gen
stdbuf -oL -eL /hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python -u -m on_policy_data_gen.decode \
    --data_dir $DATA_SPLIT \
    --model $POLICY_MODEL \
    --seeds 13 21 42 79 100 \
    --output_dir $GEN_DIR \
    --cache_dir $CACHE_DIR \
    --num_gpu 2

/hai/scratch/fangwu97/miniconda3/envs/inpo/bin/python on_policy_data_gen/post_process.py \
    --generation_file_dir $GEN_DIR

# reward model
# armo can not run on multi-gpus
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/python -u -m on_policy_data_gen.rm_armo_new \
    --cache_dir $CACHE_DIR \
    --input_file $GEN_DIR/all_outputs.json \
    --output_file $SCORED_FILE

# pre
/hai/scratch/fangwu97/miniconda3/envs/sim/bin/accelerate launch --num_processes=2 -m mnpo_scripts.precompute \
    --run_name "mnpo_iter3_precompute" \
    --model_name_or_path $POLICY_MODEL \
    --ref_model $REF_MODEL \
    --train_dir $SCORED_FILE \
    --output_dir $PREF_DIR \
    --history_paths $POLICY_MODEL_T_2 \
        $POLICY_MODEL \
    --cache_dir $CACHE_DIR \
    --sanity_check False