from vllm import LLM, SamplingParams
from datasets import load_dataset, load_from_disk
import os
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER" # this is recommended for gemma-2 models; otherwise it is not needed
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Decode with vllm')
parser.add_argument('--data_dir', type=str, default="HuggingFaceH4/ultrafeedback_binarized",
                    help='Directory containing the data')
parser.add_argument('--model', type=str, default="google/gemma-2-9b-it",
                    help='Path to the LLM model')
parser.add_argument('--temperature', type=float, default=0.8,
                    help='Temperature for sampling')
parser.add_argument('--top_p', type=float, default=0.95,
                    help='Top-p probability for sampling')
parser.add_argument('--max_tokens', type=int, default=4096,
                    help='Maximum number of tokens to generate')
parser.add_argument('--output_dir', type=str, default="datasets/gemma2_ultrafeedback",
                    help='Output directory')
parser.add_argument('--num_gpu', type=int, default=4)
parser.add_argument('--sanity_check', action='store_true', help="Enable sanity check (only use 100 samples)")
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--cache_dir', type=str, default=None,
                    help='Cache directory for model and dataset')
parser.add_argument('--seeds', type=int, nargs='+', default=[42],
                    help='A list of random seeds to run')
args = parser.parse_args()

print(args)

data_dir = args.data_dir
llm = LLM(
    model=args.model,
    tensor_parallel_size=args.num_gpu,
    download_dir=args.cache_dir,
    gpu_memory_utilization=0.9,  # Allow VLLM to use 90% of GPU memory
)
tokenizer = llm.get_tokenizer()

if os.path.exists(data_dir):
    # If the input is an existing local file path
    print("Detected local file path, loading local file...")
    # Use the 'json' loader, which supports both .json and .jsonl files
    train_dataset = load_dataset("json", data_files=data_dir, split="train")
else:
    # If not a local file path, assume it is a dataset name on Hugging Face Hub
    print("No local file detected, trying to load from Hugging Face Hub...")
    train_dataset = load_dataset(data_dir, split="train")

# If sanity check is enabled, only select a small number of samples
if args.sanity_check:
    print("Performing sanity check, using only 100 samples.")
    train_dataset = train_dataset.select(range(min(len(train_dataset), 100)))

prompts = sorted(list(set(train_dataset['prompt'])))

conversations = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]

for seed in args.seeds:
    print(f"\n--- Processing for seed {seed} ---")

    sampling_params = SamplingParams(temperature=args.temperature,
                                     top_p=args.top_p,
                                     max_tokens=args.max_tokens,
                                     seed=seed,)
    output_data = []

    print(f"Submitting {len(conversations)} prompts to vLLM in a single batch...")

    try:

        all_outputs = llm.generate(conversations, sampling_params)

        print("Generation complete. Processing outputs...")

        for i, output in enumerate(tqdm(all_outputs)):
            output_data.append({
                'prompt': prompts[i],
                "format_prompt": output.prompt,
                'generated_text': output.outputs[0].text,
            })

    except Exception as e:
        print(f"Generation failed with error: {e}")

    output_file = f'output_{seed}.json'
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, output_file), 'w') as f:
        json.dump(output_data, f, indent=4)

    print(f"Outputs saved to {os.path.join(args.output_dir, output_file)}")

print("\nAll seeds processed.")