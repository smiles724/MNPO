#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

# --- Central Configuration ---
export PYTHONPATH=$(pwd)
NUM_GPUS=8
NUM_ITERATIONS=3
BASE_MODEL="google/gemma-2-9b-it"
DATASET_NAME="princeton-nlp/gemma2-ultrafeedback-armorm"
BASE_DATA_PATH="./data"
BASE_OUTPUT_PATH="./outputs"
CACHE_DIR="./cache"

# --- Function for On-Policy Data Generation ---
generate_on_policy_data() {
    local iter=$1
    local model_path=$2
    local data_part_path="${BASE_DATA_PATH}/gemma2_ufb_part${iter}.jsonl"
    local output_dir="./datasets/gemma2_ultrafeedback/mnpo_iter${iter}"

    echo "----------------------------------------------------"
    echo "    Iteration ${iter}: Starting On-Policy Data Generation"
    echo "----------------------------------------------------"

    conda activate mnpo_infer

    for SEED in 13 21 42 79 100; do
        echo "Running decode with seed $SEED..."
        stdbuf -oL -eL python -u -m on_policy_data_gen.decode \
            --data_dir "$data_part_path" \
            --model "$model_path" \
            --seed "$SEED" \
            --output_dir "$output_dir" \
            --batch_size 96 \
            --cache_dir "$CACHE_DIR" \
            --num_gpu "$NUM_GPUS"
    done

    python -m on_policy_data_gen.post_process \
        --generation_file_dir "$output_dir"

    conda activate mnpo_train

    python -m on_policy_data_gen.reward_model_annotate \
        --generation_file "${output_dir}/all_outputs.json" \
        --output_dir "$output_dir" \
        --cache_dir "$CACHE_DIR"

    echo "On-policy data generation for iteration ${iter} complete."
}

# --- 1. Initial Data Preparation (Run Once) ---
echo "===================================================="
echo "         Step 1: Splitting Initial Dataset"
echo "===================================================="
python -m mnpo_scripts.split_dataset \
    --dataset_name "$DATASET_NAME" \
    --num_splits "$NUM_ITERATIONS" \
    --seed 42 \
    --base_save_path "$BASE_DATA_PATH" \
    --json_save_path "$BASE_DATA_PATH"
echo "Dataset splitting complete."


# --- 2. Iterative Training Loop ---
history_paths=()

for i in $(seq 1 $NUM_ITERATIONS); do
    echo ""
    echo "===================================================="
    echo "               STARTING ITERATION $i"
    echo "===================================================="

    # --- Define paths and models for the current iteration ---
    i_prev=$((i - 1))

    # The model for precomputation and data generation is the output of the previous stage.
    # For iter 1, it's the base model.
    if [ "$i" -eq 1 ]; then
        current_policy_model="$BASE_MODEL"
        train_data_dir="${BASE_DATA_PATH}/gemma2_ufb_part1.jsonl"
    else
        current_policy_model="${BASE_OUTPUT_PATH}/gemma-2-9b-it_mnpo_stage_${i_prev}/"
        # Generate on-policy data for iterations > 1
        generate_on_policy_data "$i" "$current_policy_model"
        train_data_dir="./datasets/gemma2_ultrafeedback/mnpo_iter${i}"
    fi

    test_data_dir="${BASE_DATA_PATH}/gemma2_ufb_part${i}_test.jsonl"
    precompute_output_dir="${BASE_DATA_PATH}/mnpo_iter${i}/pref"

    # --- Precomputation Step ---
    echo "----------------------------------------------------"
    echo "         Iteration $i: Starting Precomputation"
    echo "----------------------------------------------------"

    history_args=""
    if [ ${#history_paths[@]} -gt 0 ]; then
        # The space at the end is important
        history_args="--history_paths ${history_paths[*]} "
    fi

    accelerate launch --num_processes=$NUM_GPUS -m mnpo_scripts.precompute \
        --run_name "mnpo_iter${i}_precompute" \
        --model_name_or_path "$current_policy_model" \
        --ref_model "$BASE_MODEL" \
        --train_dir "$train_data_dir" \
        --test_dir "$test_data_dir" \
        --output_dir "$precompute_output_dir" \
        ${history_args} \
        --cache_dir "$CACHE_DIR" \
        --sanity_check False
    echo "Precomputation for iteration $i complete."

    # --- Training Step ---
    echo "----------------------------------------------------"
    echo "            Iteration $i: Starting Training"
    echo "----------------------------------------------------"

    ACCELERATE_LOG_LEVEL=info accelerate launch \
        --config_file accelerate_configs/deepspeed_zero3.yaml \
        -m mnpo_scripts.run_mnpo \
        "training_configs/gemma-2-9b-it-mnpo-iter${i}.yaml"

    # --- Update History for Next Iteration ---
    history_paths+=("${BASE_OUTPUT_PATH}/gemma-2-9b-it_mnpo_stage_${i}/")

    echo ""
    echo "**************** COMPLETED ITERATION $i ****************"
done

echo ""
echo "===================================================="
echo "             ALL TRAINING ITERATIONS COMPLETE"
echo "===================================================="