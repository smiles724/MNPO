import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from transformers import HfArgumentParser


@dataclass
class ScriptArgs:
    cache_dir: str = field(metadata={"help": "cache directory"})
    input_file: str = field(metadata={"help": "input json/jsonl"})
    output_file: str = field(metadata={"help": "output jsonl"})

parser = HfArgumentParser(ScriptArgs)
args = parser.parse_args_into_dataclasses()[0]

# parameters configuration
MODEL_NAME = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
CACHE_DIR = args.cache_dir
INPUT_FILE = args.input_file
OUTPUT_FILE = args.output_file

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
            raise ValueError("JSON file has to be list or dict")

    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def main():

    print(f"Step 1: loading {MODEL_NAME}...")
    print(f"         - Cache Dir: {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        use_fast=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,  # must enable for armorm
        cache_dir=CACHE_DIR
    )
    model.eval()
    print("         - model loading completed")

    print(f"Step 2: processing {INPUT_FILE}...")
    dataset = load_data(INPUT_FILE)

    try:
        total_lines = sum(1 for _ in open(INPUT_FILE, 'r'))
        print(f"         - found {total_lines} samples")
    except Exception:
        total_lines = 0

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for sample in tqdm(dataset, total=total_lines, desc="Scoring and creating pairs"):
            prompt = sample['prompt']
            responses = sample['all_generated_responses']


            if not responses:
                sample['all_rm_scores'] = []
                sample['chosen'] = []
                sample['rejected'] = []
                f_out.write(json.dumps(sample, ensure_ascii=False) + '\n')
                continue


            conversations_batch = []
            for resp in responses:
                conv = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp}
                ]
                conversations_batch.append(conv)


            input_ids = tokenizer.apply_chat_template(
                conversations_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            ).to("cuda")

            with torch.no_grad():
                output = model(input_ids)
                scores = output.score.float().cpu().tolist()


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

    print(f"\nStep 3: all done!")
    print(f"         - ArmoRM formatted data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()