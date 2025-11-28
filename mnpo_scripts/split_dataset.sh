

/path/to/mnpo_train/bin/python -m mnpo_scripts.split_dataset \
    --dataset_name princeton-nlp/gemma2-ultrafeedback-armorm \
    --num_splits 3 \
    --seed 42 \
    --json_save_path "./data"