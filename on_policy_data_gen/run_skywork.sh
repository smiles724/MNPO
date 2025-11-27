

/path/to/mnpo_train/bin/python -u -m on_policy_data_gen.rm_skywork_batch \
    --cache_dir /another/cache/dir \
    --input_file /datasets/gemma2_ultrafeedback/mnpo_iter3_skywork/all_outputs.json \
    --output_file /datasets/gemma2_ultrafeedback/mnpo_iter3_skywork_scored.jsonl
