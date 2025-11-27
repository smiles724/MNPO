import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from tqdm import tqdm


MODEL_NAME = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
CACHE_DIR = "/hai/scratch/fangwu97/xu/cache"
INPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_skywork/all_outputs.json"
OUTPUT_FILE = "/hai/scratch/fangwu97/xu/SimPO_slurm/datasets/gemma2_ultrafeedback/mnpo_iter3_skywork_scored.jsonl"

MAX_SEQ_LENGTH = 4096


def load_data(file_path):
    suffix = os.path.splitext(file_path)[1].lower()

    if suffix in [".jsonl", ".jsonlines", ".ljson"]:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    elif suffix == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            for item in data:
                yield item

        elif isinstance(data, dict):
            yield data

        else:
            raise ValueError("JSON format has to be list or dict")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")




def main():
    print(f"Step 1: loading {MODEL_NAME}...")
    print(f"         - Cache Dir: {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        num_labels=1,
        cache_dir=CACHE_DIR
    )
    model.eval()
    print("         - model loaded.")

    bos_token = tokenizer.bos_token
    strip_bos = bos_token is not None
    if strip_bos:
        print(f"         - removing BOS token: '{bos_token}'")

    print(f"Step 2: processing {INPUT_FILE}...")
    dataset = load_data(INPUT_FILE)

    try:
        total_lines = sum(1 for _ in open(INPUT_FILE, 'r'))
        print(f"         - found {total_lines} samples ")
    except Exception:
        total_lines = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for sample in tqdm(dataset, total=total_lines, desc="Scoring responses"):
            prompt = sample['prompt']
            responses = sample['all_generated_responses']

            if not responses:
                sample['skywork_v2_scores'] = []
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue

            conversations_batch = []
            for resp in responses:
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}
                ]
                conversations_batch.append(conv)

            formatted_batch = tokenizer.apply_chat_template(
                conversations_batch,
                tokenize=False,
            )

            if strip_bos:
                formatted_batch_stripped = [
                    s[len(bos_token):] if s.startswith(bos_token) else s
                    for s in formatted_batch
                ]
            else:
                formatted_batch_stripped = formatted_batch

            inputs = tokenizer(
                formatted_batch_stripped,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH
            ).to(model.device)

            with torch.no_grad():
                logits = model(**inputs).logits

                scores = logits.squeeze(-1).float().cpu().tolist()

            max_idx = scores.index(max(scores))
            min_idx = scores.index(min(scores))

            if max_idx == min_idx:
                if len(scores) == 1:
                    chosen_response = responses[0]
                    rejected_response = responses[0]
                else:

                    chosen_response = responses[0]
                    rejected_response = responses[1]
            else:
                chosen_response = responses[max_idx]
                rejected_response = responses[min_idx]


            sample['all_rm_scores'] = scores

            sample['chosen'] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen_response}
            ]

            sample['rejected'] = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": rejected_response}
            ]

            f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"\nStep 3: all done！")
    print(f"         - results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()