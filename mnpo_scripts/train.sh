ACCELERATE_LOG_LEVEL=info /path/to/mnpo_train/bin/accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml \
    -m mnpo_scripts.run_mnpo \
    "training_configs/gemma-2-9b-it-mnpo-iter2-armo-td3.yaml"
