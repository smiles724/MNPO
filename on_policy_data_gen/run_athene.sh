

/path/to/mnpo_train/bin/python -u -m on_policy_data_gen.rm_athene \
    --input_file /datasets/gemma2_ultrafeedback/mnpo_iter3_athene/all_outputs.json \
    --output_file /datasets/gemma2_ultrafeedback/mnpo_iter3_athene_scored.jsonl \
    --cache_dir /another/cache/dir \
    --batch_size 256
